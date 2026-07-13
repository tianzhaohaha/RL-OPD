#!/usr/bin/env python3
"""Compute matrix-level spectral-preservation metrics for Qwen safetensors checkpoints.

Two metrics are computed for every targeted weight matrix, comparing the
checkpoint weight W_plus against the base weight W_0:

  1. Principal-angle rotation
         cos θ_i(U) = σ_i( U0_k^T U+_k )
         cos θ_i(V) = σ_i( V0_k^T V+_k )
     where U.,k and V.,k are the top-k left/right singular vectors.
     Reported in DEGREES. Smaller angles => stronger preservation of the
     pretrained dominant subspaces.

  2. Normalized spectral shift (spectral drift)
         NSS(W0, W+) = || σ(W+) - σ(W0) ||_2 / || σ(W0) ||_2
     where σ(·) are singular values in descending order. Smaller NSS =>
     stronger spectral preservation.

Both metrics share a single full SVD per matrix (svd vectors are needed for the
angles, the full singular-value spectrum for NSS), so we decompose each tensor
exactly once. This mirrors compute_condition_qwen_matrices.py in structure and
reuses the same checkpoint/target/IO helpers.
"""
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


METRIC_TARGETS = (
    "principal-angle-mean",
    "principal-angle-max",
    "principal-angle-u-mean",
    "principal-angle-v-mean",
    "nss",
)

# Identity columns are shared with the HSE schema (drop the HSE-specific ones),
# then append the spectral-preservation columns.
BASE_FIELDS = tuple(field for field in HSE_CSV_FIELDS if field not in ("hse", "s"))
CSV_FIELDS = BASE_FIELDS + (
    "top_k",
    "effective_k",
    "principal_angle_u_mean",
    "principal_angle_u_max",
    "principal_angle_v_mean",
    "principal_angle_v_max",
    "principal_angle_mean",
    "principal_angle_max",
    "nss",
    "metric_value",
    "metric_target",
)


def existing_keys(jsonl_path: Path) -> Set[Tuple[int, str]]:
    """Rows already computed, keyed by (step, param_name).

    One row carries every metric column, so the chosen metric_target does not
    gate computation; we skip a (step, param) pair only if it is already present.
    """
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
    return value if value is not None and math.isfinite(value) else None


def _adjoint(matrix: torch.Tensor) -> torch.Tensor:
    """Conjugate transpose of the last two dims; plain transpose for real tensors."""
    return matrix.transpose(-2, -1).conj()


def compute_spectral_preservation(
    checkpoint_tensor: torch.Tensor,
    base_tensor: torch.Tensor,
    device: torch.device,
    top_k: int,
    eps: float,
) -> Dict[str, float]:
    """Return principal-angle (degrees) and NSS metrics for one matrix pair.

    Uses one full SVD (full_matrices=False) per tensor. For very large matrices
    where only the angles are needed, torch.svd_lowrank(A, q=top_k) is a faster
    approximate alternative, but NSS requires the full spectrum, so we keep the
    exact full SVD here.
    """
    if checkpoint_tensor.ndim != 2 or base_tensor.ndim != 2:
        raise ValueError(
            "Spectral-preservation metrics expect 2D tensors, got shapes "
            f"{list(checkpoint_tensor.shape)} and {list(base_tensor.shape)}"
        )

    with torch.no_grad():
        w_ckpt = checkpoint_tensor.to(dtype=torch.float32, device=device, non_blocking=True)
        w_base = base_tensor.to(dtype=torch.float32, device=device, non_blocking=True)

        # U, S, Vh with Vh = V^H (rows are right singular vectors), S descending.
        u_ckpt, s_ckpt, vh_ckpt = torch.linalg.svd(w_ckpt, full_matrices=False)
        u_base, s_base, vh_base = torch.linalg.svd(w_base, full_matrices=False)

        # ---- Principal angles between top-k subspaces ----
        k_eff = int(min(top_k, u_base.shape[1], u_ckpt.shape[1]))
        if k_eff < 1:
            raise ValueError("Effective k collapsed below 1; check matrix rank.")

        # cos θ(U) = singular values of U0_k^H U+_k
        m_u = _adjoint(u_base[:, :k_eff]) @ u_ckpt[:, :k_eff]
        cos_u = torch.linalg.svdvals(m_u).clamp_(0.0, 1.0)
        # cos θ(V): right vectors are rows of Vh, so V0_k^H V+_k = Vh0[:k] @ Vh+[:k]^H
        m_v = vh_base[:k_eff, :] @ _adjoint(vh_ckpt[:k_eff, :])
        cos_v = torch.linalg.svdvals(m_v).clamp_(0.0, 1.0)

        theta_u = torch.rad2deg(torch.arccos(cos_u))
        theta_v = torch.rad2deg(torch.arccos(cos_v))
        theta_both = torch.cat([theta_u, theta_v])

        # ---- Normalized spectral shift ----
        denom = torch.linalg.vector_norm(s_base)
        if denom.item() <= eps:
            nss = float("nan")
        else:
            nss = float(torch.linalg.vector_norm(s_ckpt - s_base) / denom)

        metrics = {
            "effective_k": k_eff,
            "principal_angle_u_mean": float(theta_u.mean()),
            "principal_angle_u_max": float(theta_u.max()),
            "principal_angle_v_mean": float(theta_v.mean()),
            "principal_angle_v_max": float(theta_v.max()),
            "principal_angle_mean": float(theta_both.mean()),
            "principal_angle_max": float(theta_both.max()),
            "nss": nss,
        }

    return metrics


def choose_metric_value(metrics: Dict[str, float], metric_target: str) -> float:
    mapping = {
        "principal-angle-mean": "principal_angle_mean",
        "principal-angle-max": "principal_angle_max",
        "principal-angle-u-mean": "principal_angle_u_mean",
        "principal-angle-v-mean": "principal_angle_v_mean",
        "nss": "nss",
    }
    if metric_target not in mapping:
        raise ValueError(f"Unknown metric target: {metric_target}")
    return metrics[mapping[metric_target]]


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
    top_k: int,
    eps: float,
    metric_target: str,
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
                if tuple(checkpoint_tensor.shape) != tuple(base_tensor.shape):
                    raise ValueError(
                        f"Shape mismatch for {param_name}: checkpoint shape {list(checkpoint_tensor.shape)} "
                        f"vs base shape {list(base_tensor.shape)}"
                    )

                metrics = compute_spectral_preservation(
                    checkpoint_tensor,
                    base_tensor,
                    device=device,
                    top_k=top_k,
                    eps=eps,
                )
                metric_value = choose_metric_value(metrics, metric_target)
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
                    "top_k": top_k,
                    "effective_k": metrics["effective_k"],
                    "principal_angle_u_mean": finite_or_none(metrics["principal_angle_u_mean"]),
                    "principal_angle_u_max": finite_or_none(metrics["principal_angle_u_max"]),
                    "principal_angle_v_mean": finite_or_none(metrics["principal_angle_v_mean"]),
                    "principal_angle_v_max": finite_or_none(metrics["principal_angle_v_max"]),
                    "principal_angle_mean": finite_or_none(metrics["principal_angle_mean"]),
                    "principal_angle_max": finite_or_none(metrics["principal_angle_max"]),
                    "nss": finite_or_none(metrics["nss"]),
                    "metric_value": finite_or_none(metric_value),
                    "metric_target": metric_target,
                }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute matrix-level principal-angle rotation and normalized spectral "
            "shift (NSS) for Qwen safetensors checkpoints relative to a base model."
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
        help="Base model path or Hugging Face repo id used as the spectral reference.",
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
        help="Output directory. Defaults to <checkpoint-root>/spectral_preservation_analysis.",
    )
    parser.add_argument(
        "--device",
        default="cuda:6",
        help="Torch device for SVD computation. Default: cuda:6.",
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
        "--top-k",
        type=int,
        default=64,
        help=(
            "Number of top singular directions defining the dominant subspace for "
            "principal angles. Automatically clamped to the matrix rank. Default: 64."
        ),
    )
    parser.add_argument(
        "--metric-target",
        choices=METRIC_TARGETS,
        default="principal-angle-mean",
        help=(
            "Which metric is mirrored into the main metric_value column. All metric "
            "columns are always written, so you can re-plot any column without "
            "recomputing. Default: principal-angle-mean."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Threshold below which ||sigma(W0)||_2 is treated as zero. Default: 1e-12.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Append outputs and skip (step, param_name) pairs already present in "
            "spectral_preservation_matrices.jsonl."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eps < 0:
        raise ValueError("--eps must be non-negative.")
    if args.top_k < 1:
        raise ValueError("--top-k must be a positive integer.")

    checkpoint_root = args.checkpoint_root
    output_dir = args.output_dir or (checkpoint_root / "spectral_preservation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "spectral_preservation_matrices.jsonl"
    csv_path = output_dir / "spectral_preservation_matrices.csv"

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
    print(f"[INFO] top_k: {args.top_k}")
    print(f"[INFO] metric_target: {args.metric_target}")
    print("[INFO] mode: exact full SVD; principal angles (deg) + NSS")

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
                top_k=args.top_k,
                eps=args.eps,
                metric_target=args.metric_target,
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