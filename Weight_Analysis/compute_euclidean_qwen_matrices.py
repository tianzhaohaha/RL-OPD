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
    resolve_model_dir as resolve_hf_or_local_model_dir,
)


CSV_FIELDS = tuple(field for field in HSE_CSV_FIELDS if field != "hse" and field != "s") + (
    "numel",
    "euclidean_distance",
    "rms_distance",
    "base_norm",
    "relative_euclidean_distance",
)


def is_safetensors_model_dir(path: Path) -> bool:
    return (path / "model.safetensors.index.json").exists() or (path / "model.safetensors").exists()


def resolve_base_model_dir(model_or_path: str, local_files_only: bool) -> Path:
    raw_path = Path(model_or_path).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        candidates.extend(
            [
                Path.cwd() / raw_path,
                Path.cwd() / "OPD" / raw_path,
                repo_root / raw_path,
                repo_root / "OPD" / raw_path,
            ]
        )

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            model_dir = checkpoint_model_dir(candidate)
            if is_safetensors_model_dir(model_dir):
                return model_dir

    return resolve_hf_or_local_model_dir(model_or_path, local_files_only=local_files_only)


def existing_keys(jsonl_path: Path) -> Set[Tuple[int, str]]:
    if not jsonl_path.exists():
        return set()

    keys: Set[Tuple[int, str]] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((int(record["step"]), str(record["param_name"])))
    return keys


def jsonl_has_required_fields(jsonl_path: Path) -> bool:
    if not jsonl_path.exists():
        return True

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            return set(CSV_FIELDS).issubset(record)
    return True


def csv_has_required_fields(csv_path: Path) -> bool:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return set(CSV_FIELDS).issubset(reader.fieldnames or [])


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


def compute_euclidean_stats_full(
    checkpoint_tensor: torch.Tensor,
    base_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float]:
    checkpoint_tensor = checkpoint_tensor.to(dtype=torch.float32, device=device, non_blocking=True)
    base_tensor = base_tensor.to(dtype=torch.float32, device=device, non_blocking=True)
    distance = torch.linalg.vector_norm(checkpoint_tensor - base_tensor).item()
    base_norm = torch.linalg.vector_norm(base_tensor).item()
    return distance, base_norm


def compute_euclidean_stats_chunked(
    checkpoint_tensor: torch.Tensor,
    base_tensor: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> Tuple[float, float]:
    checkpoint_flat = checkpoint_tensor.reshape(-1)
    base_flat = base_tensor.reshape(-1)
    diff_total = 0.0
    base_total = 0.0

    with torch.no_grad():
        for start in range(0, checkpoint_flat.numel(), chunk_size):
            end = min(start + chunk_size, checkpoint_flat.numel())
            checkpoint_chunk = checkpoint_flat[start:end].to(
                dtype=torch.float32,
                device=device,
                non_blocking=True,
            )
            base_chunk = base_flat[start:end].to(
                dtype=torch.float32,
                device=device,
                non_blocking=True,
            )
            diff = checkpoint_chunk - base_chunk
            diff_total += torch.dot(diff, diff).item()
            base_total += torch.dot(base_chunk, base_chunk).item()

    return math.sqrt(diff_total), math.sqrt(base_total)


def compute_euclidean_stats(
    checkpoint_tensor: torch.Tensor,
    base_tensor: torch.Tensor,
    device: torch.device,
    chunk_size: Optional[int],
) -> Dict[str, float]:
    if tuple(checkpoint_tensor.shape) != tuple(base_tensor.shape):
        raise ValueError(
            f"Shape mismatch: checkpoint shape {list(checkpoint_tensor.shape)} "
            f"vs base shape {list(base_tensor.shape)}"
        )

    numel = checkpoint_tensor.numel()
    with torch.no_grad():
        if chunk_size is None:
            distance, base_norm = compute_euclidean_stats_full(
                checkpoint_tensor=checkpoint_tensor,
                base_tensor=base_tensor,
                device=device,
            )
        else:
            distance, base_norm = compute_euclidean_stats_chunked(
                checkpoint_tensor=checkpoint_tensor,
                base_tensor=base_tensor,
                device=device,
                chunk_size=chunk_size,
            )

    if base_norm == 0.0:
        raise ValueError("Cannot compute relative Euclidean distance because base tensor norm is zero.")

    return {
        "numel": numel,
        "euclidean_distance": distance,
        "rms_distance": distance / math.sqrt(numel),
        "base_norm": base_norm,
        "relative_euclidean_distance": distance / base_norm,
    }


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
    chunk_size: Optional[int],
    skip_keys: Set[Tuple[int, str]],
) -> Iterable[Dict[str, object]]:
    shard_to_targets: Dict[str, List[Tuple[int, str, str, str]]] = {}
    for layer, block, matrix_name, param_name, shard_name in targets:
        if (step, param_name) in skip_keys:
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
                distance_stats = compute_euclidean_stats(
                    checkpoint_tensor=checkpoint_tensor,
                    base_tensor=base_tensor,
                    device=device,
                    chunk_size=chunk_size,
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
                    **distance_stats,
                }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute matrix-level Euclidean distance between Qwen safetensors "
            "checkpoints and a base model."
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
        help="Base model path or Hugging Face repo id used as the Euclidean-distance reference.",
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
        help="Output directory. Defaults to <checkpoint-root>/euclidean_distance_analysis.",
    )
    parser.add_argument(
        "--device",
        default="cuda:6",
        help="Torch device for distance computation. Default: cuda:6.",
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
        "--skip-existing",
        action="store_true",
        help=(
            "Append outputs and skip rows already present in "
            "euclidean_distance_matrices.jsonl for the same step/param."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional element chunk size for lower memory use. Default: disabled/full tensor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size is not None and args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive when provided.")

    checkpoint_root = args.checkpoint_root
    output_dir = args.output_dir or (checkpoint_root / "euclidean_distance_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "euclidean_distance_matrices.jsonl"
    csv_path = output_dir / "euclidean_distance_matrices.csv"

    device = normalize_device(args.device)
    steps = parse_int_spec(args.steps, "steps")
    requested_layers = parse_int_spec(args.layers, "layers")
    matrices = parse_matrix_spec(args.matrices)
    checkpoints = discover_checkpoints(checkpoint_root, steps)
    base_model_dir = resolve_base_model_dir(args.base_model, local_files_only=args.local_files_only)
    base_weight_map = load_weight_map(base_model_dir)

    if args.skip_existing:
        if jsonl_path.exists() and not jsonl_has_required_fields(jsonl_path):
            print(
                "[WARN] Existing JSONL is missing one or more current output columns; "
                "disabling --skip-existing and recomputing Euclidean rows."
            )
            args.skip_existing = False
        elif csv_path.exists() and not csv_has_required_fields(csv_path):
            if jsonl_path.exists():
                print("[INFO] Existing CSV schema is stale; rebuilding CSV from JSONL.")
                write_csv_from_jsonl(jsonl_path, csv_path)
            else:
                print(
                    "[WARN] Existing CSV is missing one or more current output columns; "
                    "disabling --skip-existing and recomputing Euclidean rows."
                )
                args.skip_existing = False
        elif jsonl_path.exists() and not csv_path.exists():
            write_csv_from_jsonl(jsonl_path, csv_path)

    print(f"[INFO] checkpoint_root: {checkpoint_root}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] base_model: {args.base_model}")
    print(f"[INFO] base_model_dir: {base_model_dir}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] matrices: {','.join(matrices)}")
    print(f"[INFO] checkpoints: {[step for step, _, _ in checkpoints]}")
    if args.chunk_size is None:
        print("[INFO] distance mode: full tensor")
    else:
        print(f"[INFO] distance mode: exact element chunks, chunk_size={args.chunk_size}")

    total_written = 0
    with jsonl_path.open("a" if args.skip_existing else "w", encoding="utf-8") as jsonl_file, csv_path.open(
        "a" if args.skip_existing else "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if not args.skip_existing or not csv_path.exists() or csv_path.stat().st_size == 0:
            writer.writeheader()
        skip_keys = existing_keys(jsonl_path) if args.skip_existing else set()
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
                chunk_size=args.chunk_size,
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