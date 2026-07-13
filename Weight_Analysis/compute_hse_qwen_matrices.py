#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from safetensors import safe_open
from tqdm import tqdm


ATTN_MATRICES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MATRICES = ("gate_proj", "up_proj", "down_proj")
DEFAULT_MATRICES = ATTN_MATRICES + MLP_MATRICES
CSV_FIELDS = (
    "checkpoint",
    "step",
    "layer",
    "block",
    "matrix",
    "param_name",
    "shape",
    "hse",
    "s",
)


def parse_int_spec(spec: Optional[str], label: str) -> Optional[Set[int]]:
    if spec is None or spec.strip().lower() in {"", "all"}:
        return None

    values: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid {label} range: {part}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    return values


def parse_matrix_spec(spec: str) -> Tuple[str, ...]:
    if spec.strip().lower() in {"", "all"}:
        return DEFAULT_MATRICES

    requested = tuple(part.strip() for part in spec.split(",") if part.strip())
    unknown = sorted(set(requested) - set(DEFAULT_MATRICES))
    if unknown:
        raise ValueError(
            f"Unknown matrix name(s): {unknown}. Valid names: {', '.join(DEFAULT_MATRICES)}"
        )
    return requested


def matrix_block(matrix_name: str) -> str:
    if matrix_name in ATTN_MATRICES:
        return "self_attn"
    if matrix_name in MLP_MATRICES:
        return "mlp"
    raise ValueError(f"Unknown matrix name: {matrix_name}")


def checkpoint_model_dir(checkpoint_dir: Path) -> Path:
    for candidate in (
        checkpoint_dir,
        checkpoint_dir / "actor" / "huggingface",
        checkpoint_dir / "huggingface",
    ):
        if (candidate / "model.safetensors.index.json").exists() or (candidate / "model.safetensors").exists():
            return candidate
    return checkpoint_dir


def discover_checkpoints(checkpoint_root: Path, steps: Optional[Set[int]]) -> List[Tuple[int, str, Path]]:
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")

    checkpoints: List[Tuple[int, str, Path]] = []
    for child in checkpoint_root.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)|global_step_(\d+)", child.name)
        if not match:
            continue
        step = int(match.group(1) or match.group(2))
        if steps is not None and step not in steps:
            continue
        checkpoints.append((step, child.name, checkpoint_model_dir(child)))

    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        step_desc = "all steps" if steps is None else f"steps {sorted(steps)}"
        raise FileNotFoundError(
            f"No checkpoint-* or global_step_* directories found for {step_desc} in {checkpoint_root}"
        )
    return checkpoints


def load_weight_map(model_dir: Path) -> Dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        single_safetensors = model_dir / "model.safetensors"
        if not single_safetensors.exists():
            raise FileNotFoundError(
                f"Missing safetensors index or single model.safetensors in: {model_dir}"
            )
        with safe_open(single_safetensors, framework="pt", device="cpu") as f:
            return {key: single_safetensors.name for key in f.keys()}

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid safetensors index, missing weight_map: {index_path}")
    return weight_map


def resolve_model_dir(model_or_path: str, local_files_only: bool) -> Path:
    local_path = Path(model_or_path)
    if local_path.exists():
        return local_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required when --base-model is a repo id instead of a local path."
        ) from exc

    snapshot_path = snapshot_download(
        repo_id=model_or_path,
        allow_patterns=["model.safetensors", "model.safetensors.index.json", "*.safetensors"],
        local_files_only=local_files_only,
    )
    return Path(snapshot_path)


def discover_layers(weight_map: Dict[str, str], matrices: Sequence[str]) -> List[int]:
    pattern = re.compile(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.([^.]+)\.weight$")
    matrix_set = set(matrices)
    layers = set()
    for name in weight_map:
        match = pattern.fullmatch(name)
        if match and match.group(3) in matrix_set:
            layers.add(int(match.group(1)))
    return sorted(layers)


def build_targets(
    weight_map: Dict[str, str],
    layers: Sequence[int],
    matrices: Sequence[str],
) -> List[Tuple[int, str, str, str, str]]:
    targets: List[Tuple[int, str, str, str, str]] = []
    missing: List[str] = []
    for layer in layers:
        for matrix_name in matrices:
            block = matrix_block(matrix_name)
            param_name = f"model.layers.{layer}.{block}.{matrix_name}.weight"
            shard_name = weight_map.get(param_name)
            if shard_name is None:
                missing.append(param_name)
                continue
            targets.append((layer, block, matrix_name, param_name, shard_name))

    if missing:
        print(f"[WARN] Missing {len(missing)} requested tensors; first missing: {missing[0]}")
    if not targets:
        raise ValueError("No target tensors found after applying layer/matrix filters.")
    return targets


def format_s(value: float) -> str:
    return f"{value:.12g}"


def existing_keys(jsonl_path: Path, s: float) -> Set[Tuple[int, str, str]]:
    if not jsonl_path.exists():
        return set()

    keys: Set[Tuple[int, str, str]] = set()
    s_key = format_s(s)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if format_s(float(record.get("s", "nan"))) != s_key:
                continue
            keys.add((int(record["step"]), str(record["param_name"]), s_key))
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


def hyperspherical_energy_riesz_full(
    matrix: torch.Tensor,
    device: torch.device,
    s: float,
) -> float:
    matrix = matrix.to(dtype=torch.float32, device=device, non_blocking=True)

    norm_matrix = matrix / (matrix.norm(dim=1, keepdim=True) + 1e-6)
    sim = torch.matmul(norm_matrix, norm_matrix.T).clamp(min=-1.0, max=1.0)
    dist = torch.sqrt((2 - 2 * sim).clamp(min=1e-6))
    energy = dist.pow(-s)
    mask = ~torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
    return energy[mask].sum().item()


def hyperspherical_energy_riesz_chunked(
    matrix: torch.Tensor,
    device: torch.device,
    s: float,
    chunk_size: int,
) -> float:
    matrix = matrix.to(dtype=torch.float32, device=device, non_blocking=True)
    norm_matrix = matrix / (matrix.norm(dim=1, keepdim=True) + 1e-6)

    total = 0.0
    n_rows = norm_matrix.size(0)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        sim = torch.matmul(norm_matrix[start:end], norm_matrix.T).clamp(min=-1.0, max=1.0)
        dist = torch.sqrt((2 - 2 * sim).clamp(min=1e-6))
        energy = dist.pow(-s)
        row_idx = torch.arange(end - start, device=device)
        col_idx = torch.arange(start, end, device=device)
        energy[row_idx, col_idx] = 0
        total += energy.sum().item()
    return total


def compute_hse(
    matrix: torch.Tensor,
    device: torch.device,
    s: float,
    chunk_size: Optional[int],
) -> float:
    with torch.no_grad():
        if chunk_size is None:
            return hyperspherical_energy_riesz_full(matrix, device=device, s=s)
        return hyperspherical_energy_riesz_chunked(
            matrix,
            device=device,
            s=s,
            chunk_size=chunk_size,
        )


def normalize_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def rows_from_targets(
    model_dir: Path,
    checkpoint_name: str,
    step: int,
    targets: Sequence[Tuple[int, str, str, str, str]],
    device: torch.device,
    s: float,
    chunk_size: Optional[int],
    skip_keys: Set[Tuple[int, str, str]],
) -> Iterable[Dict[str, object]]:
    shard_to_targets: Dict[str, List[Tuple[int, str, str, str]]] = {}
    for layer, block, matrix_name, param_name, shard_name in targets:
        key = (step, param_name, format_s(s))
        if key in skip_keys:
            continue
        shard_to_targets.setdefault(shard_name, []).append((layer, block, matrix_name, param_name))

    for shard_name in sorted(shard_to_targets):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard: {shard_path}")

        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            shard_targets = shard_to_targets[shard_name]
            for layer, block, matrix_name, param_name in tqdm(
                shard_targets,
                desc=f"{checkpoint_name}/{shard_name}",
                leave=False,
            ):
                tensor = shard.get_tensor(param_name)
                hse = compute_hse(tensor, device=device, s=s, chunk_size=chunk_size)
                shape = list(tensor.shape)
                del tensor
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
                    "hse": hse,
                    "s": s,
                }


def write_hse_rows(
    *,
    model_dir: Path,
    checkpoint_name: str,
    step: int,
    output_jsonl: Path,
    output_csv: Path,
    device: torch.device,
    s: float,
    chunk_size: Optional[int],
    matrices: Sequence[str],
    requested_layers: Optional[Set[int]],
    skip_existing: bool,
) -> int:
    if skip_existing and output_jsonl.exists() and not output_csv.exists():
        write_csv_from_jsonl(output_jsonl, output_csv)

    skip_keys = existing_keys(output_jsonl, s) if skip_existing else set()
    json_mode = "a" if skip_existing else "w"
    csv_mode = "a" if skip_existing else "w"
    csv_needs_header = not output_csv.exists() or output_csv.stat().st_size == 0 or not skip_existing

    weight_map = load_weight_map(model_dir)
    layers = sorted(requested_layers) if requested_layers is not None else discover_layers(
        weight_map,
        matrices,
    )
    targets = build_targets(weight_map, layers, matrices)
    print(f"[INFO] {checkpoint_name}: {len(targets)} target tensors")

    total_written = 0
    with output_jsonl.open(json_mode, encoding="utf-8") as jsonl_file, output_csv.open(
        csv_mode, encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if csv_needs_header:
            writer.writeheader()

        for row in rows_from_targets(
            model_dir=model_dir,
            checkpoint_name=checkpoint_name,
            step=step,
            targets=targets,
            device=device,
            s=s,
            chunk_size=chunk_size,
            skip_keys=skip_keys,
        ):
            jsonl_file.write(json.dumps(row, ensure_ascii=True) + "\n")
            writer.writerow(row)
            jsonl_file.flush()
            csv_file.flush()
            total_written += 1

    return total_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute matrix-level Riesz hyperspherical energy for Qwen safetensors checkpoints."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("output/train_ckpt/gp_l_sft-slight-qwen"),
        help="Directory containing checkpoint-* or global_step_* subdirectories.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Optional untrained baseline model path or Hugging Face repo id, "
            "for example Qwen/Qwen2.5-7B-Instruct. Writes hse_base_matrices.*."
        ),
    )
    parser.add_argument(
        "--base-name",
        default="base",
        help="Checkpoint label used in base HSE outputs. Default: base.",
    )
    parser.add_argument(
        "--base-output-prefix",
        default="hse_base_matrices",
        help="Output prefix for base HSE files. Default: hse_base_matrices.",
    )
    parser.add_argument(
        "--only-base",
        action="store_true",
        help="Only compute --base-model HSE and skip checkpoint-* directories.",
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
        help="Output directory. Defaults to <checkpoint-root>/hse_analysis.",
    )
    parser.add_argument(
        "--device",
        default="cuda:6",
        help="Torch device for HSE computation. Default: cuda:0.",
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
        help="Comma-separated matrix names. Default: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj.",
    )
    parser.add_argument("--s", type=float, default=1.0, help="Riesz energy exponent. Default: 1.0.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Append outputs and skip rows already present in hse_matrices.jsonl for the same step/param/s.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional exact row chunk size for lower memory use. Default: disabled/full matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size is not None and args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive when provided.")
    if args.only_base and args.base_model is None:
        raise ValueError("--only-base requires --base-model.")

    checkpoint_root = args.checkpoint_root
    output_dir = args.output_dir or (checkpoint_root / "hse_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "hse_matrices.jsonl"
    csv_path = output_dir / "hse_matrices.csv"

    device = normalize_device(args.device)
    steps = parse_int_spec(args.steps, "steps")
    requested_layers = parse_int_spec(args.layers, "layers")
    matrices = parse_matrix_spec(args.matrices)
    checkpoints = [] if args.only_base else discover_checkpoints(checkpoint_root, steps)

    print(f"[INFO] checkpoint_root: {checkpoint_root}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] matrices: {','.join(matrices)}")
    if not args.only_base:
        print(f"[INFO] checkpoints: {[step for step, _, _ in checkpoints]}")
    if args.chunk_size is None:
        print("[INFO] HSE mode: full matrix, aligned with reference implementation")
    else:
        print(f"[INFO] HSE mode: exact row chunks, chunk_size={args.chunk_size}")

    if args.base_model is not None:
        base_dir = resolve_model_dir(args.base_model, local_files_only=args.local_files_only)
        base_jsonl_path = output_dir / f"{args.base_output_prefix}.jsonl"
        base_csv_path = output_dir / f"{args.base_output_prefix}.csv"
        print(f"[INFO] base_model: {args.base_model}")
        print(f"[INFO] base_dir: {base_dir}")
        base_written = write_hse_rows(
            model_dir=base_dir,
            checkpoint_name=args.base_name,
            step=0,
            output_jsonl=base_jsonl_path,
            output_csv=base_csv_path,
            device=device,
            s=args.s,
            chunk_size=args.chunk_size,
            matrices=matrices,
            requested_layers=requested_layers,
            skip_existing=args.skip_existing,
        )
        print(f"[DONE] wrote {base_written} new base rows")
        print(f"[DONE] base jsonl: {base_jsonl_path}")
        print(f"[DONE] base csv: {base_csv_path}")

    total_written = 0
    if not args.only_base:
        with jsonl_path.open("a" if args.skip_existing else "w", encoding="utf-8") as jsonl_file, csv_path.open(
            "a" if args.skip_existing else "w", encoding="utf-8", newline=""
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            if not args.skip_existing or not csv_path.exists() or csv_path.stat().st_size == 0:
                writer.writeheader()
            skip_keys = existing_keys(jsonl_path, args.s) if args.skip_existing else set()
            if args.skip_existing:
                print(f"[INFO] skip_existing: loaded {len(skip_keys)} existing checkpoint row keys")

            for step, checkpoint_name, checkpoint_dir in tqdm(checkpoints, desc="checkpoints"):
                weight_map = load_weight_map(checkpoint_dir)
                layers = sorted(requested_layers) if requested_layers is not None else discover_layers(
                    weight_map,
                    matrices,
                )
                targets = build_targets(weight_map, layers, matrices)
                print(f"[INFO] {checkpoint_name}: {len(targets)} target tensors")

                for row in rows_from_targets(
                    model_dir=checkpoint_dir,
                    checkpoint_name=checkpoint_name,
                    step=step,
                    targets=targets,
                    device=device,
                    s=args.s,
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
