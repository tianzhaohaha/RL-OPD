#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


METRICS = ("hse", "euclidean", "condition", "spectral")
METRIC_CHOICES = ("all",) + METRICS
PLOT_LEVELS_WITH_HEATMAP = {"heatmap", "all"}
CONDITION_COLUMNS = (
    "condition_number",
    "condition_number_ratio",
    "condition_number_relative_change",
    "condition_number_delta",
    "checkpoint_condition_number",
    "base_condition_number",
)
SPECTRAL_COLUMNS = (
    "principal_angle_mean",
    "principal_angle_max",
    "principal_angle_u_mean",
    "principal_angle_u_max",
    "principal_angle_v_mean",
    "principal_angle_v_max",
    "nss",
    "metric_value",
)
CHECKPOINT_DIR_RE = re.compile(r"^(?:checkpoint-(\d+)|global_step_(\d+))$")


def add_optional_arg(command: List[str], flag: str, value: Optional[object]) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def add_bool_arg(command: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def run_command(command: Sequence[str], *, cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    print("[RUN] " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True, cwd=str(cwd) if cwd is not None else None, env=env)


def python_executable(args: argparse.Namespace) -> str:
    return str(args.python_executable or sys.executable)


def parse_csv_list(value: str, *, all_values: Tuple[str, ...], label: str) -> Tuple[str, ...]:
    if value.strip().lower() == "all":
        return all_values

    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if not items:
        raise ValueError(f"--{label} must not be empty.")

    unknown = sorted(set(items) - set(all_values))
    if unknown:
        raise ValueError(f"Unknown --{label} value(s): {unknown}. Valid values: {', '.join(all_values)}")
    return items


def parse_metrics(values: Sequence[str]) -> Tuple[str, ...]:
    if "all" in values:
        if len(values) > 1:
            raise ValueError("--metrics all cannot be combined with specific metric names.")
        return METRICS
    return tuple(values)


def metric_requested(metrics: Iterable[str], metric: str) -> bool:
    return metric in set(metrics)


def has_hf_safetensors(model_dir: Path) -> bool:
    return (model_dir / "model.safetensors.index.json").exists() or (model_dir / "model.safetensors").exists()


def is_fsdp_checkpoint_dir(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "fsdp_config.json").exists() and any(checkpoint_dir.glob("model_world_size_*_rank_*.pt"))


def checkpoint_step(path: Path) -> Optional[int]:
    match = CHECKPOINT_DIR_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1) or match.group(2))


def parse_step_filter(spec: Optional[str]) -> Optional[set]:
    if spec is None or spec.strip().lower() in {"", "all"}:
        return None

    values = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid steps range: {part}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    return values


def discover_checkpoint_dirs(checkpoint_root: Path, steps: Optional[set]) -> Tuple[Path, ...]:
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")

    checkpoint_dirs = []
    for child in checkpoint_root.iterdir():
        if not child.is_dir():
            continue
        step = checkpoint_step(child)
        if step is None:
            continue
        if steps is not None and step not in steps:
            continue
        checkpoint_dirs.append((step, child))
    return tuple(path for _, path in sorted(checkpoint_dirs, key=lambda item: item[0]))


def actor_checkpoint_dir(checkpoint_dir: Path) -> Optional[Path]:
    actor_dir = checkpoint_dir / "actor"
    if is_fsdp_checkpoint_dir(actor_dir):
        return actor_dir
    if is_fsdp_checkpoint_dir(checkpoint_dir):
        return checkpoint_dir
    return None


def merged_hf_dir(local_dir: Path) -> Path:
    return local_dir / "huggingface"


def resolve_verl_root(args: argparse.Namespace, script_dir: Path) -> Optional[Path]:
    if args.verl_root is not None:
        return args.verl_root

    repo_root = script_dir.parent
    candidate = repo_root / "verl"
    if (candidate / "verl" / "model_merger" / "__main__.py").exists():
        return candidate
    return None


def merge_env(args: argparse.Namespace, verl_root: Optional[Path]) -> dict:
    env = os.environ.copy()
    if verl_root is None:
        return env
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(verl_root) if not existing else f"{verl_root}{os.pathsep}{existing}"
    return env


def ensure_merged_checkpoints(args: argparse.Namespace, script_dir: Path) -> None:
    if args.plot_only or args.no_auto_merge:
        return

    steps = parse_step_filter(args.steps)
    checkpoint_dirs = discover_checkpoint_dirs(args.checkpoint_root, steps)
    if not checkpoint_dirs:
        return

    verl_root = resolve_verl_root(args, script_dir)
    env = merge_env(args, verl_root)
    merge_cwd = verl_root or Path.cwd()
    for checkpoint_dir in checkpoint_dirs:
        local_dir = actor_checkpoint_dir(checkpoint_dir)
        if local_dir is None:
            continue

        target_dir = merged_hf_dir(local_dir)
        if has_hf_safetensors(target_dir):
            print(f"[INFO] merged checkpoint exists: {target_dir}")
            continue
        if args.dry_run_merge:
            print(f"[DRY-RUN] would merge {local_dir} -> {target_dir}")
            continue

        command = [
            python_executable(args),
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            args.merge_backend,
            "--local_dir",
            str(local_dir),
            "--target_dir",
            str(target_dir),
        ]
        run_command(command, cwd=merge_cwd, env=env)


def compute_hse(args: argparse.Namespace, script_dir: Path, output_dir: Path) -> None:
    command = [
        python_executable(args),
        str(script_dir / "compute_hse_qwen_matrices.py"),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--matrices",
        args.matrices,
        "--s",
        str(args.hse_s),
    ]
    add_optional_arg(command, "--steps", args.steps)
    add_optional_arg(command, "--layers", args.layers)
    add_optional_arg(command, "--base-model", args.base_model)
    add_optional_arg(command, "--base-name", args.hse_base_name)
    add_optional_arg(command, "--base-output-prefix", args.hse_base_output_prefix)
    add_optional_arg(command, "--chunk-size", args.chunk_size)
    add_bool_arg(command, "--local-files-only", args.local_files_only)
    add_bool_arg(command, "--skip-existing", args.skip_existing)
    run_command(command)


def compute_euclidean(args: argparse.Namespace, script_dir: Path, output_dir: Path) -> None:
    command = [
        python_executable(args),
        str(script_dir / "compute_euclidean_qwen_matrices.py"),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--base-model",
        str(args.base_model),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--matrices",
        args.matrices,
    ]
    add_optional_arg(command, "--steps", args.steps)
    add_optional_arg(command, "--layers", args.layers)
    add_optional_arg(command, "--chunk-size", args.chunk_size)
    add_bool_arg(command, "--local-files-only", args.local_files_only)
    add_bool_arg(command, "--skip-existing", args.skip_existing)
    run_command(command)


def compute_condition(args: argparse.Namespace, script_dir: Path, output_dir: Path) -> None:
    command = [
        python_executable(args),
        str(script_dir / "compute_condition_qwen_matrices.py"),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--base-model",
        str(args.base_model),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--matrices",
        args.matrices,
        "--condition-target",
        args.condition_target,
        "--eps",
        str(args.condition_eps),
    ]
    add_optional_arg(command, "--steps", args.steps)
    add_optional_arg(command, "--layers", args.layers)
    add_bool_arg(command, "--local-files-only", args.local_files_only)
    add_bool_arg(command, "--skip-existing", args.skip_existing)
    run_command(command)


def compute_spectral(args: argparse.Namespace, script_dir: Path, output_dir: Path) -> None:
    command = [
        python_executable(args),
        str(script_dir / "compute spectral preservation qwen matrices.py"),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--base-model",
        str(args.base_model),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--matrices",
        args.matrices,
        "--top-k",
        str(args.spectral_top_k),
        "--metric-target",
        args.spectral_metric_target,
        "--eps",
        str(args.spectral_eps),
    ]
    add_optional_arg(command, "--steps", args.steps)
    add_optional_arg(command, "--layers", args.layers)
    add_bool_arg(command, "--local-files-only", args.local_files_only)
    add_bool_arg(command, "--skip-existing", args.skip_existing)
    run_command(command)


def common_plot_command(args: argparse.Namespace, script_path: Path, csv_path: Path, output_dir: Path) -> List[str]:
    command = [
        python_executable(args),
        str(script_path),
        "--csv",
        str(csv_path),
        "--output-dir",
        str(output_dir),
        "--matrices",
        args.matrices,
        "--yscale",
        args.yscale,
        "--dpi",
        str(args.dpi),
        "--plot-level",
        args.plot_level,
    ]
    add_optional_arg(command, "--steps", args.steps)
    add_optional_arg(command, "--layers", args.layers)
    add_optional_arg(command, "--baseline-step", args.baseline_step)
    add_optional_arg(command, "--heatmap-vmin", args.heatmap_vmin)
    add_optional_arg(command, "--heatmap-vmax", args.heatmap_vmax)
    add_bool_arg(command, "--mean-only", args.mean_only)
    add_bool_arg(command, "--plot-correct-solution", args.plot_correct_solution)
    add_optional_arg(command, "--eval-checkpoint-root", args.eval_checkpoint_root)
    return command


def plot_hse(args: argparse.Namespace, script_dir: Path, analysis_dir: Path, output_dir: Path) -> None:
    variants = parse_csv_list(
        args.hse_plot_variants,
        all_values=("raw", "normalized"),
        label="hse-plot-variants",
    )
    for variant in variants:
        variant_output_dir = output_dir / variant if len(variants) > 1 else output_dir
        command = common_plot_command(
            args,
            script_dir / "plot_hse_qwen_matrices.py",
            analysis_dir / "hse_matrices.csv",
            variant_output_dir,
        )
        command.extend(["--heatmap-value", args.hse_heatmap_value])
        if variant == "normalized":
            command.extend(["--hse-transform", args.hse_normalize_method])
        else:
            command.extend(["--hse-transform", "raw"])
        add_optional_arg(command, "--baseline-csv", args.hse_baseline_csv or (analysis_dir / f"{args.hse_base_output_prefix}.csv"))
        run_command(command)


def plot_euclidean(args: argparse.Namespace, script_dir: Path, analysis_dir: Path, output_dir: Path) -> None:
    command = common_plot_command(
        args,
        script_dir / "plot_euclidean_qwen_matrices.py",
        analysis_dir / "euclidean_distance_matrices.csv",
        output_dir,
    )
    command.extend(["--distance-metric", args.euclidean_distance_metric])
    command.extend(["--heatmap-value", args.euclidean_heatmap_value])
    run_command(command)


def plot_condition(
    args: argparse.Namespace,
    script_dir: Path,
    analysis_dir: Path,
    output_dir: Path,
    value_columns: Tuple[str, ...],
) -> None:
    for value_column in value_columns:
        column_output_dir = output_dir / value_column if len(value_columns) > 1 else output_dir
        command = common_plot_command(
            args,
            script_dir / "plot_condition_qwen_matrices.py",
            analysis_dir / "condition_number_matrices.csv",
            column_output_dir,
        )
        command.extend(
            [
                "--value-column",
                value_column,
                "--condition-target",
                args.condition_target,
                "--heatmap-value",
                args.condition_heatmap_value,
            ]
        )
        run_command(command)


def plot_spectral(
    args: argparse.Namespace,
    script_dir: Path,
    analysis_dir: Path,
    output_dir: Path,
    value_columns: Tuple[str, ...],
) -> None:
    for value_column in value_columns:
        column_output_dir = output_dir / value_column if len(value_columns) > 1 else output_dir
        command = common_plot_command(
            args,
            script_dir / "plot spectral preservation qwen matrices.py",
            analysis_dir / "spectral_preservation_matrices.csv",
            column_output_dir,
        )
        command.extend(
            [
                "--value-column",
                value_column,
                "--metric-target",
                args.spectral_metric_target,
                "--heatmap-value",
                args.spectral_heatmap_value,
            ]
        )
        run_command(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot all Qwen matrix-level metrics with one command."
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
        help="Base model path or Hugging Face repo id for Euclidean/condition metrics and optional HSE base plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Unified output directory. Defaults to <checkpoint-root>/matrix_metric_analysis.",
    )
    parser.add_argument(
        "--python-executable",
        default=None,
        help="Python executable used for merge, compute, and plot subprocesses. Defaults to the current Python.",
    )
    parser.add_argument("--device", default="cuda:6", help="Torch device used by compute scripts. Default: cuda:6.")
    parser.add_argument("--steps", default=None, help="Checkpoint steps, e.g. 500 or 500,1000-2000. Default: all.")
    parser.add_argument("--layers", default=None, help="Layer ids, e.g. 0 or 0,3-7. Default: all discovered layers.")
    parser.add_argument(
        "--matrices",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated matrix names. Default: Qwen attention and MLP matrices.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_CHOICES,
        default=["all"],
        help="Metrics to process. Use all for every metric. Default: all.",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Use local Hugging Face cache only for repo ids.")
    parser.add_argument("--skip-existing", action="store_true", help="Append outputs and skip already-computed rows.")
    parser.add_argument("--compute-only", action="store_true", help="Only run metric computation.")
    parser.add_argument("--plot-only", action="store_true", help="Only run plotting from existing CSVs.")
    parser.add_argument("--merge-only", action="store_true", help="Only run automatic checkpoint merge, then exit.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Optional chunk size for HSE and Euclidean computation.")
    parser.add_argument(
        "--no-auto-merge",
        action="store_true",
        help="Disable automatic FSDP checkpoint merge before computation.",
    )
    parser.add_argument(
        "--dry-run-merge",
        action="store_true",
        help="Print FSDP checkpoint merge commands that would run, without executing them.",
    )
    parser.add_argument(
        "--merge-backend",
        choices=("fsdp", "megatron"),
        default="fsdp",
        help="Backend passed to verl.model_merger for automatic merge. Default: fsdp.",
    )
    parser.add_argument(
        "--verl-root",
        type=Path,
        default=None,
        help="Path to the outer verl repo directory used for python -m verl.model_merger. Default: auto-detect ./verl.",
    )

    parser.add_argument("--hse-s", type=float, default=1.0, help="Riesz energy exponent for HSE. Default: 1.0.")
    parser.add_argument("--hse-base-name", default="base", help="Checkpoint label used in HSE base outputs.")
    parser.add_argument(
        "--hse-base-output-prefix",
        default="hse_base_matrices",
        help="Output prefix for base HSE files. Default: hse_base_matrices.",
    )
    parser.add_argument(
        "--condition-target",
        choices=("ratio", "relative-change", "delta", "checkpoint"),
        default="ratio",
        help="Main condition_number value written by the condition compute script. Default: ratio.",
    )
    parser.add_argument("--condition-eps", type=float, default=1e-12, help="Smallest singular value threshold. Default: 1e-12.")
    parser.add_argument(
        "--spectral-top-k",
        type=int,
        default=64,
        help="Top singular directions used for spectral principal angles. Default: 64.",
    )
    parser.add_argument(
        "--spectral-metric-target",
        choices=(
            "principal-angle-mean",
            "principal-angle-max",
            "principal-angle-u-mean",
            "principal-angle-v-mean",
            "nss",
        ),
        default="principal-angle-mean",
        help="Metric mirrored into the spectral metric_value column. Default: principal-angle-mean.",
    )
    parser.add_argument(
        "--spectral-eps",
        type=float,
        default=1e-12,
        help="Threshold below which ||sigma(W0)||_2 is treated as zero for NSS. Default: 1e-12.",
    )

    parser.add_argument(
        "--plot-level",
        choices=("matrix", "matrix-layers", "param", "heatmap", "both", "all"),
        default="heatmap",
        help="Plot level passed to all plot scripts. Default: heatmap.",
    )
    parser.add_argument("--yscale", choices=("linear", "log"), default="linear", help="Y-axis scale. Default: linear.")
    parser.add_argument("--dpi", type=int, default=200, help="PNG DPI. Default: 200.")
    parser.add_argument("--mean-only", action="store_true", help="Only plot mean curves for matrix-level plots.")
    parser.add_argument("--baseline-step", type=int, default=None, help="Baseline step for relative-baseline heatmaps.")
    parser.add_argument("--heatmap-vmin", type=float, default=None, help="Optional heatmap color minimum.")
    parser.add_argument("--heatmap-vmax", type=float, default=None, help="Optional heatmap color maximum.")
    parser.add_argument(
        "--hse-heatmap-value",
        choices=("relative-base", "relative-layer-mean", "relative-baseline", "raw", "zscore-layer"),
        default="relative-base",
        help="HSE heatmap mode. Default: relative-base.",
    )
    parser.add_argument(
        "--hse-plot-variants",
        default="raw,normalized",
        help="Comma-separated HSE plot variants to write: raw, normalized, or all. Default: raw,normalized.",
    )
    parser.add_argument(
        "--hse-normalize-method",
        choices=("robust-log",),
        default="robust-log",
        help="Normalization used for the normalized HSE plot variant. Default: robust-log.",
    )
    parser.add_argument(
        "--euclidean-distance-metric",
        choices=("euclidean", "relative", "rms"),
        default="relative",
        help=(
            "Euclidean metric to plot. euclidean uses raw ||W_ckpt-W_base||_2; "
            "relative uses ||W_ckpt-W_base||_2 / ||W_base||_2; "
            "rms uses ||W_ckpt-W_base||_2 / sqrt(numel). Default: relative."
        ),
    )
    parser.add_argument(
        "--euclidean-heatmap-value",
        choices=("raw", "relative-layer-mean", "relative-baseline", "zscore-layer"),
        default="raw",
        help=(
            "Euclidean heatmap mode applied after --euclidean-distance-metric. "
            "Default: raw, which plots the selected Euclidean metric directly."
        ),
    )
    parser.add_argument(
        "--condition-heatmap-value",
        choices=("raw", "relative-layer-mean", "relative-baseline", "zscore-layer"),
        default="raw",
        help="Condition-number heatmap mode. Default: raw.",
    )
    parser.add_argument(
        "--spectral-heatmap-value",
        choices=("raw", "relative-layer-mean", "relative-baseline", "zscore-layer"),
        default="raw",
        help="Spectral-preservation heatmap mode. Default: raw.",
    )
    parser.add_argument(
        "--condition-value-columns",
        default="condition_number",
        help=(
            "Comma-separated condition columns to plot, or all. "
            f"Valid values: {', '.join(CONDITION_COLUMNS)}. Default: condition_number."
        ),
    )
    parser.add_argument(
        "--spectral-value-columns",
        default="principal_angle_mean,nss",
        help=(
            "Comma-separated spectral-preservation columns to plot, or all. "
            f"Valid values: {', '.join(SPECTRAL_COLUMNS)}. Default: principal_angle_mean,nss."
        ),
    )
    parser.add_argument("--hse-baseline-csv", type=Path, default=None, help="Optional HSE base CSV for relative-base heatmaps.")
    parser.add_argument("--plot-correct-solution", action="store_true", help="Overlay/annotate ID/OOD CORRECT_SOLUTION rates.")
    parser.add_argument(
        "--eval-checkpoint-root",
        type=Path,
        default=None,
        help="Checkpoint root containing checkpoint-*/eval_res/*.jsonl for CORRECT_SOLUTION plots.",
    )
    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
    condition_value_columns: Tuple[str, ...],
    spectral_value_columns: Tuple[str, ...],
) -> None:
    if args.compute_only and args.plot_only:
        raise ValueError("--compute-only and --plot-only cannot be used together.")
    if args.merge_only and args.plot_only:
        raise ValueError("--merge-only and --plot-only cannot be used together.")
    if args.chunk_size is not None and args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive when provided.")
    if (
        not args.plot_only
        and not args.merge_only
        and (
            metric_requested(args.metrics, "euclidean")
            or metric_requested(args.metrics, "condition")
            or metric_requested(args.metrics, "spectral")
        )
        and args.base_model is None
    ):
        raise ValueError("--base-model is required when computing euclidean/condition/spectral metrics.")
    if (
        metric_requested(args.metrics, "hse")
        and not args.compute_only
        and not args.merge_only
        and args.hse_heatmap_value == "relative-base"
        and args.plot_level in PLOT_LEVELS_WITH_HEATMAP
        and args.base_model is None
        and args.hse_baseline_csv is None
    ):
        raise ValueError("HSE relative-base heatmaps require --base-model or --hse-baseline-csv.")
    if not condition_value_columns:
        raise ValueError("No condition value columns selected.")
    if not spectral_value_columns:
        raise ValueError("No spectral value columns selected.")
    if args.spectral_top_k < 1:
        raise ValueError("--spectral-top-k must be a positive integer.")
    if args.spectral_eps < 0:
        raise ValueError("--spectral-eps must be non-negative.")


def main() -> None:
    args = parse_args()
    try:
        args.metrics = parse_metrics(args.metrics)
        condition_value_columns = parse_csv_list(
            args.condition_value_columns,
            all_values=CONDITION_COLUMNS,
            label="condition-value-columns",
        )
        spectral_value_columns = parse_csv_list(
            args.spectral_value_columns,
            all_values=SPECTRAL_COLUMNS,
            label="spectral-value-columns",
        )
        validate_args(args, condition_value_columns, spectral_value_columns)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    script_dir = Path(__file__).resolve().parent
    output_root = args.output_dir or (args.checkpoint_root / "matrix_metric_analysis")
    output_root.mkdir(parents=True, exist_ok=True)

    analysis_dirs = {
        "hse": output_root / "hse_analysis",
        "euclidean": output_root / "euclidean_distance_analysis",
        "condition": output_root / "condition_number_analysis",
        "spectral": output_root / "spectral_preservation_analysis",
    }
    plot_root = output_root / "plots"

    print(f"[INFO] output root: {output_root}")
    print(f"[INFO] metrics: {', '.join(args.metrics)}")

    if args.merge_only:
        ensure_merged_checkpoints(args, script_dir)
        print(f"[DONE] merge check complete: {args.checkpoint_root}")
        return

    if not args.plot_only:
        ensure_merged_checkpoints(args, script_dir)
        if metric_requested(args.metrics, "hse"):
            compute_hse(args, script_dir, analysis_dirs["hse"])
        if metric_requested(args.metrics, "euclidean"):
            compute_euclidean(args, script_dir, analysis_dirs["euclidean"])
        if metric_requested(args.metrics, "condition"):
            compute_condition(args, script_dir, analysis_dirs["condition"])
        if metric_requested(args.metrics, "spectral"):
            compute_spectral(args, script_dir, analysis_dirs["spectral"])

    if not args.compute_only:
        plot_root.mkdir(parents=True, exist_ok=True)
        if metric_requested(args.metrics, "hse"):
            plot_hse(args, script_dir, analysis_dirs["hse"], plot_root / "hse")
        if metric_requested(args.metrics, "euclidean"):
            plot_euclidean(args, script_dir, analysis_dirs["euclidean"], plot_root / "euclidean_distance")
        if metric_requested(args.metrics, "condition"):
            plot_condition(
                args,
                script_dir,
                analysis_dirs["condition"],
                plot_root / "condition_number",
                condition_value_columns,
            )
        if metric_requested(args.metrics, "spectral"):
            plot_spectral(
                args,
                script_dir,
                analysis_dirs["spectral"],
                plot_root / "spectral_preservation",
                spectral_value_columns,
            )

    print(f"[DONE] unified output: {output_root}")


if __name__ == "__main__":
    main()