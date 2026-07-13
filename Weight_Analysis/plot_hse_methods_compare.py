#!/usr/bin/env python3
"""Compare HSE across training methods (e.g. SFT / RL / OPD).

For each model component (fine-grained matrices such as W_up, W_gate, and
coarse blocks such as the whole MLP / self-attention), draw one line chart:
    x-axis: training step
    y-axis: HSE value (aggregated over layers)
    one line per training method.

Each training method points at an hse_analysis directory produced by
compute_hse_qwen_matrices.py (containing hse_matrices.csv and the optional
hse_base_matrices.csv). The origin of every curve (step 0) is taken from the
base model's HSE (hse_base_matrices.csv).

Edit METHOD_CONFIG below so you can simply run:

    python plot_hse_methods_compare.py --output-dir runs/compare_plots

or override on the command line with --input LABEL=PATH (repeatable).
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Edit this mapping: training method label -> hse_analysis directory (or a
# direct hse_matrices.csv path). Each directory is expected to also contain
# hse_base_matrices.csv, whose values are used as the step-0 origin.
# Relative paths are resolved against this script's directory.
# ---------------------------------------------------------------------------
METHOD_CONFIG: Dict[str, str] = {
    # "SFT": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_sft_gsm8k_teacher_gen/qwen3_1_7b_sft_gsm8k_1gpu_fsdp_20260623_2332/matrix_metric_analysis/hse_analysis",
    # "RL": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_grpo_gsm8k/qwen3_1_7b_grpo_gsm8k_2gpu_fsdp_20260610_0018/matrix_metric_analysis/hse_analysis",
    "OPD_F_P": "/home/jcgu/qyliu/RL-OPD/OPD/checkpoint/token_reward_direct_DAPO-Math-17k_Qwen3-1.7B_Qwen3-4B_7168-T_1.0-Tch_1.0-n_4-mbs_48-topk_16-topk_strategy_only_stu-rw_student_p-sp_False-2026-07-06_01-28-13/matrix_metric_analysis/hse_analysis",
    "OPD_F": "/home/jcgu/qyliu/RL-OPD/OPD/checkpoint/token_reward_direct_DAPO-Math-17k_Qwen3-1.7B_Qwen3-4B_7168-T_1.0-Tch_1.0-n_4-mbs_48-topk_16-topk_strategy_only_stu-rw_student_p-2026-07-02_18-57-59/matrix_metric_analysis/hse_analysis",
    # "OPD_4b_of": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_distill_gsm8k/qwen3_1_7b_opd_from_4b_grpo_overfit_teacher_gsm8k_s1_t1_fsdp_20260628_0543/matrix_metric_analysis/hse_analysis",
    # "OPD_hero_t": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_distill_gsm8k/qwen3_1_7b_opd_from_justrl_deepseek_1_5b_teacher_gsm8k_s1_t1_fsdp_20260628_0600/matrix_metric_analysis/hse_analysis",
}


# # Model name shown in figure titles and used for the default output directory
# # (hse_compare_plots/<MODEL_NAME>).
MODEL_NAME = "Qwen3-1_7B"


# METHOD_CONFIG: Dict[str, str] = {
#     "SFT": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_sft_gsm8k/qwen3_4b_sft_gsm8k_2gpu_fsdp_20260612_0340/matrix_metric_analysis/hse_analysis",
#     "RL": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_grpo_gsm8k/qwen3_4b_grpo_gsm8k_2gpu_fsdp_20260612_2342/matrix_metric_analysis/hse_analysis",
#     "OPD": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_distill_gsm8k/qwen3_4b_from_grpo_teacher_opd_gsm8k_s1_t1_fsdp_20260614_0306/matrix_metric_analysis/hse_analysis",
# }

# Model name shown in figure titles and used for the default output directory
# (hse_compare_plots/<MODEL_NAME>).
# MODEL_NAME = "Qwen3-4B"


# METHOD_CONFIG: Dict[str, str] = {
#     "SFT": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_sft_gsm8k/qwen2.5_1.5b_sft_gsm8k_1gpu_fsdp_20260617_0454/matrix_metric_analysis/hse_analysis",
#     "RL": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_grpo_gsm8k_math/qwen2_5_1_5b_grpo_gsm8k_math_2gpu_fsdp_20260616_0434/matrix_metric_analysis/hse_analysis",
#     "OPD": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_distill_gsm8k/qwen2_5_1_5b_opd_from_grpo_teacher_gsm8k_2gpu_s1_t1_fsdp_20260617_0120/matrix_metric_analysis/hse_analysis",
# }

# # Model name shown in figure titles and used for the default output directory
# # (hse_compare_plots/<MODEL_NAME>).
# MODEL_NAME = "Qwen2_5-1_5B"


# METHOD_CONFIG: Dict[str, str] = {
#     "SFT": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_sft_gsm8k/qwen2.5_3b_sft_gsm8k_1gpu_fsdp_20260617_0454/matrix_metric_analysis/hse_analysis",
#     "RL": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_grpo_gsm8k_math/qwen2_5_1_5b_grpo_gsm8k_math_2gpu_fsdp_20260616_0444/matrix_metric_analysis/hse_analysis",
#     "OPD": "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_distill_gsm8k/qwen2_5_3b_opd_from_grpo_teacher_gsm8k_2gpu_s1_t1_fsdp_20260617_0140/matrix_metric_analysis/hse_analysis",
# }

# # Model name shown in figure titles and used for the default output directory
# # (hse_compare_plots/<MODEL_NAME>).
# MODEL_NAME = "Qwen2_5-3B"


# Methods drawn on a separate right-hand y-axis. Use this for a method whose
# HSE is so large/volatile (e.g. SFT) that it flattens the other curves on a
# shared axis. Override on the command line with --secondary-axis LABEL[,LABEL].
SECONDARY_AXIS_METHODS: Tuple[str, ...] = ("SFT",)


SCRIPT_DIR = Path(__file__).resolve().parent
MATRICES_FILENAME = "hse_matrices.csv"
BASE_FILENAME = "hse_base_matrices.csv"


# Fine-grained matrices present in the CSV.
ATTN_MATRICES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MATRICES = ("gate_proj", "up_proj", "down_proj")
FINE_MATRICES = ATTN_MATRICES + MLP_MATRICES

# Coarse components built by aggregating several fine matrices.
BLOCK_COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "self_attn": ATTN_MATRICES,
    "mlp": MLP_MATRICES,
    "all": FINE_MATRICES,
}

# Display labels for figure titles / filenames.
COMPONENT_LABELS = {
    "q_proj": r"$W_Q$",
    "k_proj": r"$W_K$",
    "v_proj": r"$W_V$",
    "o_proj": r"$W_O$",
    "gate_proj": r"$W_{\mathrm{gate}}$",
    "up_proj": r"$W_{\mathrm{up}}$",
    "down_proj": r"$W_{\mathrm{down}}$",
    "self_attn": "Self-Attention",
    "mlp": "MLP",
    "all": "All matrices",
}

# Default order in which components are plotted / listed.
COMPONENT_ORDER = FINE_MATRICES + ("self_attn", "mlp", "all")

# Distinct colors / markers per method, assigned in input order.
METHOD_STYLES = (
    {"color": "#1f77b4", "marker": "o"},
    {"color": "#d62728", "marker": "s"},
    {"color": "#2ca02c", "marker": "^"},
    {"color": "#9467bd", "marker": "D"},
    {"color": "#ff7f0e", "marker": "v"},
    {"color": "#8c564b", "marker": "P"},
)


def parse_int_spec(spec: Optional[str], label: str) -> Optional[Set[int]]:
    if spec is None or spec.strip().lower() in {"", "all"}:
        return None

    values: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid {label} range: {part}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    return values


def parse_component_spec(spec: str) -> Tuple[str, ...]:
    if spec.strip().lower() in {"", "all-components"}:
        return COMPONENT_ORDER
    requested = tuple(part.strip() for part in spec.split(",") if part.strip())
    valid = set(COMPONENT_ORDER)
    unknown = [name for name in requested if name not in valid]
    if unknown:
        raise ValueError(
            f"Unknown component(s): {unknown}. Valid: {', '.join(COMPONENT_ORDER)}"
        )
    return requested


def parse_input_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--input must be LABEL=PATH, got: {value!r}"
        )
    label, path_str = value.split("=", 1)
    label = label.strip()
    path = Path(path_str.strip())
    if not label:
        raise argparse.ArgumentTypeError(f"Empty label in --input {value!r}")
    return label, path


def resolve_path(raw: Path) -> Path:
    return raw if raw.is_absolute() else (SCRIPT_DIR / raw)


def resolve_csv_paths(raw: Path) -> Tuple[Path, Optional[Path]]:
    """Map a config entry to (matrices_csv, base_csv).

    The entry may be an hse_analysis directory or a direct hse_matrices.csv.
    """
    path = resolve_path(raw)
    if path.is_dir():
        return path / MATRICES_FILENAME, path / BASE_FILENAME
    base_csv = path.parent / BASE_FILENAME
    return path, (base_csv if base_csv.exists() else None)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def load_hse_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"HSE CSV does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"step", "layer", "matrix", "hse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    df = df.copy()
    df["step"] = pd.to_numeric(df["step"], errors="raise").astype(int)
    df["layer"] = pd.to_numeric(df["layer"], errors="raise").astype(int)
    df["hse"] = pd.to_numeric(df["hse"], errors="raise")
    return df


def apply_filters(
    df: pd.DataFrame,
    steps: Optional[Set[int]],
    layers: Optional[Set[int]],
) -> pd.DataFrame:
    filtered = df
    if steps is not None:
        filtered = filtered[filtered["step"].isin(steps)]
    if layers is not None:
        filtered = filtered[filtered["layer"].isin(layers)]
    return filtered


def component_series(
    df: pd.DataFrame,
    component: str,
    agg: str,
    base_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return a step -> aggregated HSE series for one component.

    When base_df is provided, its aggregated value is prepended as the step-0
    origin (the base model's HSE).
    """
    if component in BLOCK_COMPONENTS:
        matrices = BLOCK_COMPONENTS[component]
    else:
        matrices = (component,)
    comp_df = df[df["matrix"].isin(matrices)]
    if comp_df.empty:
        return pd.DataFrame(columns=["step", "hse"])
    # Aggregate over all layers (and over matrices for block components).
    series = (
        comp_df.groupby("step", as_index=False)["hse"]
        .agg(agg)
        .sort_values("step")
    )

    if base_df is not None:
        base_comp = base_df[base_df["matrix"].isin(matrices)]
        if not base_comp.empty:
            base_value = float(base_comp["hse"].agg(agg))
            origin = pd.DataFrame({"step": [0], "hse": [base_value]})
            series = series[series["step"] != 0]
            series = pd.concat([origin, series], ignore_index=True).sort_values("step")

    return series


def plot_component(
    method_frames: Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]],
    component: str,
    output_dir: Path,
    agg: str,
    yscale: str,
    dpi: int,
    normalize: bool,
    secondary_methods: Set[str] = frozenset(),
    model_name: str = MODEL_NAME,
) -> Optional[Path]:
    label = COMPONENT_LABELS.get(component, component)
    fig, ax = plt.subplots(figsize=(8, 5))
    # Separate right-hand axis is only created when a secondary method has data.
    secondary_ax = None
    plotted = False
    legend_handles: List = []
    legend_labels: List[str] = []

    for idx, (method, (df, base_df)) in enumerate(method_frames.items()):
        series = component_series(df, component, agg, base_df=base_df)
        if series.empty:
            print(f"[WARN] {method}: no data for component {component}; skipping line.")
            continue
        y_values = series["hse"].to_numpy(dtype=float)
        if normalize:
            base = y_values[0]
            if base != 0:
                y_values = y_values / base
        style = METHOD_STYLES[idx % len(METHOD_STYLES)]
        on_secondary = method in secondary_methods
        if on_secondary:
            if secondary_ax is None:
                secondary_ax = ax.twinx()
            target_ax = secondary_ax
            legend_name = f"{method} (right axis)"
        else:
            target_ax = ax
            legend_name = method
        (line,) = target_ax.plot(
            series["step"].to_numpy(),
            y_values,
            linewidth=2.0,
            markersize=4.5,
            linestyle="--" if on_secondary else "-",
            label=legend_name,
            **style,
        )
        legend_handles.append(line)
        legend_labels.append(legend_name)
        plotted = True

    if not plotted:
        plt.close(fig)
        print(f"[WARN] No method had data for component {component}; figure skipped.")
        return None

    ylabel = "HSE"
    if agg == "mean":
        ylabel = "Mean HSE (over layers)"
    elif agg == "sum":
        ylabel = "Total HSE (over layers)"
    if normalize:
        ylabel = f"{ylabel} / step-0 value"

    ax.set_title(f"{model_name} HSE vs training step: {label}")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    if secondary_ax is not None:
        secondary_label = ", ".join(
            method for method in method_frames if method in secondary_methods
        )
        secondary_ax.set_ylabel(f"{ylabel} [{secondary_label}]")
        secondary_ax.set_yscale(yscale)
    ax.legend(legend_handles, legend_labels, loc="best")

    output_path = output_dir / f"hse_compare_{safe_filename(component)}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay HSE curves of multiple training methods per component."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        type=parse_input_arg,
        help=(
            "Override METHOD_CONFIG: training method and its hse_analysis dir or "
            "hse_matrices.csv, e.g. SFT=runs/sft/hse_analysis. Repeat per method. "
            "If omitted, the METHOD_CONFIG dict in this script is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the output figures. "
            f"Default: hse_compare_plots/<MODEL_NAME> (currently {MODEL_NAME})."
        ),
    )
    parser.add_argument(
        "--components",
        default="all-components",
        help=(
            "Comma-separated components to plot, or all-components. "
            f"Valid: {', '.join(COMPONENT_ORDER)}. Default: all-components."
        ),
    )
    parser.add_argument(
        "--agg",
        choices=("mean", "sum"),
        default="mean",
        help="How to aggregate HSE over layers (and matrices in a block). Default: mean.",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Steps to include, e.g. 100 or 100,500-1000. Default: all.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Layer ids to include, e.g. 0 or 0,3-7. Default: all.",
    )
    parser.add_argument(
        "--yscale",
        choices=("linear", "log"),
        default="linear",
        help="Y-axis scale. Default: linear.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Divide each method's curve by its first plotted step value (relative change).",
    )
    parser.add_argument(
        "--secondary-axis",
        default=None,
        help=(
            "Comma-separated method labels drawn on a separate right-hand y-axis "
            "(for very large/volatile curves like SFT). 'none' disables it. "
            f"Default: {','.join(SECONDARY_AXIS_METHODS) or 'none'} (from SECONDARY_AXIS_METHODS)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG DPI. Default: 200.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        SCRIPT_DIR / "hse_compare_plots" / safe_filename(MODEL_NAME)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = parse_int_spec(args.steps, "steps")
    layers = parse_int_spec(args.layers, "layers")
    components = parse_component_spec(args.components)

    if args.secondary_axis is None:
        secondary_methods: Set[str] = set(SECONDARY_AXIS_METHODS)
    elif args.secondary_axis.strip().lower() in {"", "none"}:
        secondary_methods = set()
    else:
        secondary_methods = {
            part.strip() for part in args.secondary_axis.split(",") if part.strip()
        }

    if args.input:
        config_items: List[Tuple[str, Path]] = list(args.input)
    else:
        if not METHOD_CONFIG:
            raise ValueError(
                "METHOD_CONFIG is empty and no --input given. Edit METHOD_CONFIG "
                "at the top of this script or pass --input LABEL=PATH."
            )
        config_items = [(label, Path(path)) for label, path in METHOD_CONFIG.items()]

    method_frames: Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]] = {}
    for label, raw_path in config_items:
        if label in method_frames:
            raise ValueError(f"Duplicate method label: {label}")
        matrices_csv, base_csv = resolve_csv_paths(raw_path)
        df = apply_filters(load_hse_csv(matrices_csv), steps=steps, layers=layers)
        if df.empty:
            raise ValueError(f"No rows remain for {label} after filtering: {matrices_csv}")
        base_df = None
        if base_csv is not None and base_csv.exists():
            base_df = apply_filters(load_hse_csv(base_csv), steps=None, layers=layers)
            if base_df.empty:
                base_df = None
        if base_df is None:
            print(f"[WARN] {label}: no base model CSV found; step-0 origin omitted.")
        method_frames[label] = (df, base_df)
        base_note = base_csv if base_df is not None else "none"
        print(f"[INFO] {label}: {len(df)} rows from {matrices_csv} (base: {base_note})")

    print(f"[INFO] output dir: {output_dir}")
    print(f"[INFO] components: {', '.join(components)}")
    print(f"[INFO] aggregation: {args.agg}")
    if secondary_methods:
        print(f"[INFO] secondary (right) axis: {', '.join(sorted(secondary_methods))}")

    written: List[Path] = []
    for component in components:
        output_path = plot_component(
            method_frames=method_frames,
            component=component,
            output_dir=output_dir,
            agg=args.agg,
            yscale=args.yscale,
            dpi=args.dpi,
            normalize=args.normalize,
            secondary_methods=secondary_methods,
        )
        if output_path is not None:
            written.append(output_path)
            print(f"[DONE] {output_path}")

    if not written:
        raise RuntimeError("No plots were written.")


if __name__ == "__main__":
    main()
