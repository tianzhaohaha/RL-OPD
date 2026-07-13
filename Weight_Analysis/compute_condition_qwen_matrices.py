#!/usr/bin/env python3
import argparse
import csv
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from safetensors import safe_open
from tqdm import tqdm

from compute_euclidean_qwen_matrices import resolve_base_model_dir
from compute_hse_qwen_matrices import (
    CSV_FIELDS as HSE_CSV_FIELDS,
    DEFAULT_MATRICES,
    build_targets,
    checkpoint_model_dir,
    discover_checkpoints,
    discover_layers,
    load_weight_map,
    normalize_device,
    parse_int_spec,
    parse_matrix_spec,
)


CONDITION_TARGETS = ("ratio", "relative-change", "delta", "checkpoint")
CSV_FIELDS = tuple(field for field in HSE_CSV_FIELDS if field != "hse" and field != "s") + (
    "condition_number",
    "checkpoint_condition_number",
    "base_condition_number",
    "condition_number_ratio",
    "condition_number_relative_change",
    "condition_number_delta",
    "condition_target",
)


def existing_keys(jsonl_path: Path, condition_target: str) -> Set[Tuple[int, str, str]]:
    if not jsonl_path.exists():
        return set()

    keys: Set[Tuple[int, str, str]] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(record.get("condition_target", "ratio")) != condition_target:
                continue
            keys.add((int(record["step"]), str(record["param_name"]), condition_target))
    return keys


def write_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> None:
    if not jsonl_path.exists():
        return

    with jsonl_path.open("r", encoding="utf-8") as src, csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        writer = csv.DictWriter(dst, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            writer.writerow({field: record.get(field, "") for field in CSV_FIELDS})


def finite_or_none(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def divide_or_inf(numerator: float, denominator: float) -> float:
    if denominator == 0:
        if numerator == 0:
            return float("nan")
        return math.copysign(float("inf"), numerator)
    return numerator / denominator


def compute_condition_number(
    tensor: torch.Tensor,
    device: torch.device,
    eps: float,
) -> float:
    if tensor.ndim != 2:
        raise ValueError(f"Condition number expects a 2D tensor, got shape {list(tensor.shape)}")

    with torch.no_grad():
        matrix = tensor.to(dtype=torch.float32, device=device, non_blocking=True)
        singular_values = torch.linalg.svdvals(matrix)
        if singular_values.numel() == 0:
            return float("nan")
        max_singular = singular_values.max().item()
        min_singular = singular_values.min().item()
        if min_singular <= eps:
            if max_singular <= eps:
                return float("nan")
            return float("inf")
        return max_singular / min_singular


def choose_condition_value(
    checkpoint_condition: float,
    base_condition: float,
    condition_target: str,
) -> float:
    ratio = divide_or_inf(checkpoint_condition, base_condition)
    relative_change = divide_or_inf(checkpoint_condition - base_condition, base_condition)
    delta = checkpoint_condition - base_condition

    if condition_target == "ratio":
        return ratio
    if condition_target == "relative-change":
        return relative_change
    if condition_target == "delta":
        return delta
    if condition_target == "checkpoint":
        return checkpoint_condition
    raise ValueError(f"Unknown condition target: {condition_target}")


def resolve_base_shards(
    base_weight_map: Dict[str, str],
    targets: Sequence[Tuple[int, str, str, str, str]],
) -> Dict[str, str]:
    base_shards = {}
    missing = []
    for _, _, _, param_name, _ in targets:
        shard_name = base_weight_map.get(param_name)
        if shard_name is None:
            missing.append(param_name)
        else:
            base_shards[param_name] = shard_name

    if missing:
        raise ValueError(
            f"Base model is missing {len(missing)} requested tensors; first missing: {missing[0]}"
        )
    return base_shards


def rows_from_targets(
    checkpoint_model_dir: Path,
    base_model_dir: Path,
    checkpoint_name: str,
    step: int,
    targets: Sequence[Tuple[int, str, str, str, str]],
    base_shards: Dict[str, str],
    device: torch.device,
    eps: float,
    condition_target: str,
    skip_keys: Set[Tuple[int, str, str]],
) -> Iterable[Dict[str, object]]:
    shard_to_targets: Dict[str, List[Tuple[int, str, str, str]]] = {}
    for layer, block, matrix_name, param_name, shard_name in targets:
        if (step, param_name, condition_target) in skip_keys:
            continue
        shard_to_targets.setdefault(shard_name, []).append((layer, block, matrix_name, param_name))

    for shard_name in sorted(shard_to_targets):
        checkpoint_shard_path = checkpoint_model_dir / shard_name
        if not checkpoint_shard_path.exists():
            raise FileNotFoundError(f"Missing checkpoint shard: {checkpoint_shard_path}")

        with ExitStack() as stack:
            checkpoint_shard = stack.enter_context(
                safe_open(checkpoint_shard_path, framework="pt", device="cpu")
            )
            base_readers = {}

            for layer, block, matrix_name, param_name in tqdm(
                shard_to_targets[shard_name],
                desc=f"{checkpoint_name}/{shard_name}",
                leave=False,
            ):
                base_shard_name = base_shards[param_name]
                if base_shard_name not in base_readers:
                    base_shard_path = base_model_dir / base_shard_name
                    if not base_shard_path.exists():
                        raise FileNotFoundError(f"Missing base shard: {base_shard_path}")
                    base_readers[base_shard_name] = stack.enter_context(
                        safe_open(base_shard_path, framework="pt", device="cpu")
                    )

                checkpoint_tensor = checkpoint_shard.get_tensor(param_name)
                base_tensor = base_readers[base_shard_name].get_tensor(param_name)
                if tuple(checkpoint_tensor.shape) != tuple(base_tensor.shape):
                    raise ValueError(
                        f"Shape mismatch for {param_name}: checkpoint shape {list(checkpoint_tensor.shape)} "
                        f"vs base shape {list(base_tensor.shape)}"
                    )

                checkpoint_condition = compute_condition_number(
                    checkpoint_tensor,
                    device=device,
                    eps=eps,
                )
                base_condition = compute_condition_number(
                    base_tensor,
                    device=device,
                    eps=eps,
                )
                condition_ratio = divide_or_inf(checkpoint_condition, base_condition)
                condition_relative_change = divide_or_inf(
                    checkpoint_condition - base_condition,
                    base_condition,
                )
                condition_delta = checkpoint_condition - base_condition
                condition_number = choose_condition_value(
                    checkpoint_condition=checkpoint_condition,
                    base_condition=base_condition,
                    condition_target=condition_target,
                )
                shape = list(checkpoint_tensor.shape)
                del checkpoint_tensor, base_tensor
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                yield {
                    "checkpoint": checkpoint_name,
                    "step": step,
                    "layer": layer,
                    "block": block,
                    "matrix": matrix_name,
                    "param_name": param_name,
                    "shape": shape,
                    "condition_number": finite_or_none(condition_number),
                    "checkpoint_condition_number": finite_or_none(checkpoint_condition),
                    "base_condition_number": finite_or_none(base_condition),
                    "condition_number_ratio": finite_or_none(condition_ratio),
                    "condition_number_relative_change": finite_or_none(condition_relative_change),
                    "condition_number_delta": finite_or_none(condition_delta),
                    "condition_target": condition_target,
                }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute matrix-level condition numbers for Qwen safetensors checkpoints "
            "relative to a base model."
        )
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("output/train_ckpt/gp_l_sft-slight-qwen"),
        help="Directory containing checkpoint-* or global_step_* subdirectories.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model path or Hugging Face repo id used as the condition-number reference.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="When --base-model is a Hugging Face repo id, only use local HF cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <checkpoint-root>/condition_number_analysis.",
    )
    parser.add_argument(
        "--device",
        default="cuda:6",
        help="Torch device for condition-number computation. Default: cuda:6.",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Checkpoint steps to process, e.g. 500 or 500,1000-2000. Default: all.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Layer ids to process, e.g. 0 or 0,3-7. Default: all discovered layers.",
    )
    parser.add_argument(
        "--matrices",
        default=",".join(DEFAULT_MATRICES),
        help=(
            "Comma-separated matrix names. Default: "
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj."
        ),
    )
    parser.add_argument(
        "--condition-target",
        choices=CONDITION_TARGETS,
        default="ratio",
        help=(
            "Main value written to condition_number: ratio=cond(checkpoint)/cond(base); "
            "relative-change=(cond(checkpoint)-cond(base))/cond(base); "
            "delta=cond(checkpoint)-cond(base); checkpoint=cond(checkpoint). Default: ratio."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Smallest singular value threshold treated as zero. Default: 1e-12.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Append outputs and skip rows already present in "
            "condition_number_matrices.jsonl for the same step/param/condition-target."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eps < 0:
        raise ValueError("--eps must be non-negative.")

    checkpoint_root = args.checkpoint_root
    output_dir = args.output_dir or (checkpoint_root / "condition_number_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "condition_number_matrices.jsonl"
    csv_path = output_dir / "condition_number_matrices.csv"

    device = normalize_device(args.device)
    steps = parse_int_spec(args.steps, "steps")
    requested_layers = parse_int_spec(args.layers, "layers")
    matrices = parse_matrix_spec(args.matrices)
    checkpoints = discover_checkpoints(checkpoint_root, steps)
    base_model_dir = resolve_base_model_dir(args.base_model, local_files_only=args.local_files_only)
    base_weight_map = load_weight_map(base_model_dir)

    if args.skip_existing and jsonl_path.exists() and not csv_path.exists():
        write_csv_from_jsonl(jsonl_path, csv_path)

    print(f"[INFO] checkpoint_root: {checkpoint_root}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] base_model: {args.base_model}")
    print(f"[INFO] base_model_dir: {base_model_dir}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] matrices: {','.join(matrices)}")
    print(f"[INFO] checkpoints: {[step for step, _, _ in checkpoints]}")
    print(f"[INFO] condition_target: {args.condition_target}")
    print("[INFO] condition mode: exact SVD singular-value ratio")

    total_written = 0
    with jsonl_path.open("a" if args.skip_existing else "w", encoding="utf-8") as jsonl_file, csv_path.open(
        "a" if args.skip_existing else "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if not args.skip_existing or not csv_path.exists() or csv_path.stat().st_size == 0:
            writer.writeheader()
        skip_keys = existing_keys(jsonl_path, args.condition_target) if args.skip_existing else set()
        if args.skip_existing:
            print(f"[INFO] skip_existing: loaded {len(skip_keys)} existing row keys")

        for step, checkpoint_name, checkpoint_dir in tqdm(checkpoints, desc="checkpoints"):
            model_dir = checkpoint_model_dir(checkpoint_dir)
            weight_map = load_weight_map(model_dir)
            layers = sorted(requested_layers) if requested_layers is not None else discover_layers(
                weight_map,
                matrices,
            )
            targets = build_targets(weight_map, layers, matrices)
            base_shards = resolve_base_shards(base_weight_map, targets)
            print(f"[INFO] {checkpoint_name}: {len(targets)} target tensors")

            for row in rows_from_targets(
                checkpoint_model_dir=model_dir,
                base_model_dir=base_model_dir,
                checkpoint_name=checkpoint_name,
                step=step,
                targets=targets,
                base_shards=base_shards,
                device=device,
                eps=args.eps,
                condition_target=args.condition_target,
                skip_keys=skip_keys,
            ):
                jsonl_file.write(json.dumps(row, ensure_ascii=True) + "\n")
                writer.writerow(row)
                jsonl_file.flush()
                csv_file.flush()
                total_written += 1

    print(f"[DONE] wrote {total_written} new checkpoint rows")
    print(f"[DONE] jsonl: {jsonl_path}")
    print(f"[DONE] csv: {csv_path}")


if __name__ == "__main__":
    main()
