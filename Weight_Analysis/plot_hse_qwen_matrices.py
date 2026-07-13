#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


LANGUAGE_MATRICES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
EXTENDED_MATRICES = (
    "cross_attn_q_proj",
    "cross_attn_k_proj",
    "cross_attn_v_proj",
    "cross_attn_o_proj",
    "vision_self_attn_q_proj",
    "vision_self_attn_k_proj",
    "vision_self_attn_v_proj",
    "vision_self_attn_o_proj",
    "vision_mlp_fc1",
    "vision_mlp_fc2",
    "vision_global_attn_q_proj",
    "vision_global_attn_k_proj",
    "vision_global_attn_v_proj",
    "vision_global_attn_o_proj",
    "vision_global_mlp_fc1",
    "vision_global_mlp_fc2",
    "multi_modal_projector",
)
DEFAULT_MATRICES = LANGUAGE_MATRICES
ALL_MATRICES = DEFAULT_MATRICES + EXTENDED_MATRICES
MATRIX_ORDER = {matrix_name: idx for idx, matrix_name in enumerate(ALL_MATRICES)}
MATRIX_LABELS = {
    "q_proj": r"$W_Q$",
    "k_proj": r"$W_K$",
    "v_proj": r"$W_V$",
    "o_proj": r"$W_O$",
    "gate_proj": r"$W_{\mathrm{gate}}$",
    "up_proj": r"$W_{\mathrm{up}}$",
    "down_proj": r"$W_{\mathrm{down}}$",
    "cross_attn_q_proj": r"$W_Q^{cross}$",
    "cross_attn_k_proj": r"$W_K^{cross}$",
    "cross_attn_v_proj": r"$W_V^{cross}$",
    "cross_attn_o_proj": r"$W_O^{cross}$",
    "vision_self_attn_q_proj": r"$W_Q^{vision}$",
    "vision_self_attn_k_proj": r"$W_K^{vision}$",
    "vision_self_attn_v_proj": r"$W_V^{vision}$",
    "vision_self_attn_o_proj": r"$W_O^{vision}$",
    "vision_mlp_fc1": r"$W_{fc1}^{vision}$",
    "vision_mlp_fc2": r"$W_{fc2}^{vision}$",
    "vision_global_attn_q_proj": r"$W_Q^{vision-global}$",
    "vision_global_attn_k_proj": r"$W_K^{vision-global}$",
    "vision_global_attn_v_proj": r"$W_V^{vision-global}$",
    "vision_global_attn_o_proj": r"$W_O^{vision-global}$",
    "vision_global_mlp_fc1": r"$W_{fc1}^{vision-global}$",
    "vision_global_mlp_fc2": r"$W_{fc2}^{vision-global}$",
    "multi_modal_projector": r"$W_{projector}$",
}
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
EVAL_RES_JSONL_RE = re.compile(r"^(?:gp_l|virl_l)_(indist|ind|ood).*\.jsonl$")
SPLIT_LABELS = {
    "ind": "ID",
    "ood": "OOD",
}
HEATMAP_TICK_UNIT = 1e-6
CorrectSolutionRates = Dict[int, Dict[str, float]]
HSE_TRANSFORMS = ("raw", "robust-log")


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


def matrix_sort_key(matrix_name: str) -> Tuple[int, str]:
    return MATRIX_ORDER.get(matrix_name, len(MATRIX_ORDER)), matrix_name


def infer_matrices(df: pd.DataFrame) -> Tuple[str, ...]:
    return tuple(sorted((str(value) for value in df["matrix"].unique()), key=matrix_sort_key))


def parse_matrix_spec(spec: str) -> Optional[Tuple[str, ...]]:
    if spec.strip().lower() in {"", "all"}:
        return None

    return tuple(part.strip() for part in spec.split(",") if part.strip())


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def load_hse_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"HSE CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"step", "layer", "block", "matrix", "param_name", "hse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["step"] = pd.to_numeric(df["step"], errors="raise").astype(int)
    df["layer"] = pd.to_numeric(df["layer"], errors="raise").astype(int)
    df["hse"] = pd.to_numeric(df["hse"], errors="raise")
    return df


def robust_hse_scale(df: pd.DataFrame) -> float:
    finite = df["hse"].dropna()
    finite = finite[finite > 0]
    if finite.empty:
        return 1.0
    scale = float(finite.median())
    return scale if scale > 0 else 1.0


def transform_hse_df(df: pd.DataFrame, transform: str, scale: float) -> pd.DataFrame:
    if transform == "raw":
        return df
    if transform == "robust-log":
        transformed = df.copy()
        transformed["hse"] = transformed["hse"].clip(lower=0).div(scale).map(math.log1p)
        return transformed
    raise ValueError(f"Unknown HSE transform: {transform}")


def hse_label(transform: str) -> str:
    if transform == "raw":
        return "HSE"
    if transform == "robust-log":
        return "log1p(HSE / median HSE)"
    return transform


def apply_filters(
    df: pd.DataFrame,
    steps: Optional[Set[int]],
    layers: Optional[Set[int]],
    matrices: Tuple[str, ...],
) -> pd.DataFrame:
    filtered = df[df["matrix"].isin(matrices)].copy()
    if steps is not None:
        filtered = filtered[filtered["step"].isin(steps)]
    if layers is not None:
        filtered = filtered[filtered["layer"].isin(layers)]
    if filtered.empty:
        raise ValueError("No rows remain after applying filters.")
    return filtered.sort_values(["matrix", "layer", "step"])


def filter_baseline(
    df: pd.DataFrame,
    layers: Optional[Set[int]],
    matrices: Tuple[str, ...],
) -> pd.DataFrame:
    filtered = df[df["matrix"].isin(matrices)].copy()
    if layers is not None:
        filtered = filtered[filtered["layer"].isin(layers)]
    if filtered.empty:
        raise ValueError("No baseline rows remain after applying layer/matrix filters.")
    return filtered.sort_values(["matrix", "layer"])


def to_int_list(values) -> list:
    return [int(value) for value in values]


def parse_correct_solution_rate(jsonl_path: Path) -> Optional[float]:
    try:
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"[WARN] Could not read {jsonl_path}: {exc}")
        return None

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        rates = record.get("Rates")
        if not isinstance(rates, dict) or "CORRECT_SOLUTION" not in rates:
            continue

        try:
            return float(rates["CORRECT_SOLUTION"])
        except (TypeError, ValueError):
            print(f"[WARN] Invalid CORRECT_SOLUTION in {jsonl_path}: {rates['CORRECT_SOLUTION']!r}")
            return None

    print(f"[WARN] No Rates.CORRECT_SOLUTION found in {jsonl_path}")
    return None


def latest_path(paths: list) -> Path:
    return max(paths, key=lambda path: (path.stat().st_mtime, str(path)))


def split_from_eval_res_jsonl(jsonl_path: Path) -> Optional[str]:
    match = EVAL_RES_JSONL_RE.fullmatch(jsonl_path.name)
    if not match:
        return None
    split_name = match.group(1)
    return "ind" if split_name in {"indist", "ind"} else split_name


def collect_checkpoint_eval_res_candidates(
    checkpoint_root: Path,
    plotted_steps: Set[int],
) -> Dict[Tuple[int, str], list]:
    if not checkpoint_root.exists():
        print(f"[WARN] Evaluation checkpoint root does not exist: {checkpoint_root}")
        return {}

    candidates: Dict[Tuple[int, str], list] = {}
    for checkpoint_dir in checkpoint_root.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        match = CHECKPOINT_DIR_RE.fullmatch(checkpoint_dir.name)
        if not match:
            continue
        step = int(match.group(1))
        if step not in plotted_steps:
            continue

        eval_res_dir = checkpoint_dir / "eval_res"
        if not eval_res_dir.exists():
            continue
        for jsonl_path in eval_res_dir.glob("*.jsonl"):
            split = split_from_eval_res_jsonl(jsonl_path)
            if split is None:
                continue
            candidates.setdefault((step, split), []).append(jsonl_path)
    return candidates


def load_correct_solution_rates(
    checkpoint_root: Path,
    plotted_steps: Set[int],
) -> CorrectSolutionRates:
    candidates = collect_checkpoint_eval_res_candidates(
        checkpoint_root=checkpoint_root,
        plotted_steps=plotted_steps,
    )

    rates: CorrectSolutionRates = {}
    for (step, split), paths in sorted(candidates.items()):
        chosen = latest_path(paths)
        if len(paths) > 1:
            print(
                f"[WARN] Multiple {SPLIT_LABELS[split]} logs for step {step}; "
                f"using latest: {chosen}"
            )
        rate = parse_correct_solution_rate(chosen)
        if rate is not None:
            rates.setdefault(step, {})[split] = rate

    for step in sorted(plotted_steps):
        missing = [
            SPLIT_LABELS[split]
            for split in ("ind", "ood")
            if split not in rates.get(step, {})
        ]
        if missing:
            print(f"[WARN] Missing CORRECT_SOLUTION for step {step}: {', '.join(missing)}")

    return rates


def correct_solution_xy(
    steps: list,
    correct_solution_rates: Optional[CorrectSolutionRates],
    split: str,
) -> Tuple[list, list]:
    if not correct_solution_rates:
        return [], []

    x_values = []
    y_values = []
    for step in steps:
        rate = correct_solution_rates.get(step, {}).get(split)
        if rate is None:
            continue
        x_values.append(step)
        y_values.append(rate)
    return x_values, y_values


def correct_solution_percent_axis_upper(values: list) -> float:
    if not values:
        return 1.0
    max_value = max(values)
    for upper in (5, 10, 20, 30, 50, 75, 100):
        if max_value <= upper:
            return float(upper)
    return float(((int(max_value) // 25) + 1) * 25)


def add_correct_solution_axis(
    ax,
    steps: list,
    correct_solution_rates: Optional[CorrectSolutionRates],
    linewidth: float = 1.8,
    markersize: float = 4,
    show_axis: bool = True,
    tick_labelsize: Optional[float] = None,
):
    if not correct_solution_rates:
        return None

    styles = {
        "ind": {"color": "#2ca02c", "marker": "s"},
        "ood": {"color": "#d62728", "marker": "^"},
    }
    metric_axes = []
    for split in ("ind", "ood"):
        x_values, y_values = correct_solution_xy(steps, correct_solution_rates, split)
        if not x_values:
            continue
        y_values = [value * 100.0 for value in y_values]
        color = styles[split]["color"]
        metric_ax = ax.twinx()
        if len(metric_axes) > 0:
            metric_ax.spines["right"].set_position(("axes", 1.14))
        metric_ax.plot(
            x_values,
            y_values,
            linestyle="--",
            linewidth=linewidth,
            markersize=markersize,
            label=f"{SPLIT_LABELS[split]} CORRECT_SOLUTION",
            zorder=20,
            **styles[split],
        )
        metric_ax.set_ylim(0.0, correct_solution_percent_axis_upper(y_values))
        metric_ax.grid(False)
        metric_ax.spines["right"].set_color(color)
        metric_ax.tick_params(axis="y", colors=color)
        metric_ax.yaxis.label.set_color(color)
        metric_ax.set_ylabel(f"{SPLIT_LABELS[split]} CS (%)")
        if tick_labelsize is not None:
            metric_ax.tick_params(axis="y", labelsize=tick_labelsize)
            metric_ax.yaxis.label.set_size(tick_labelsize)
        if not show_axis:
            metric_ax.set_ylabel("")
            metric_ax.tick_params(axis="y", labelright=False, right=False)
            metric_ax.spines["right"].set_visible(False)
        metric_axes.append(metric_ax)

    if not metric_axes:
        return None

    return metric_axes


def set_combined_legend(ax, secondary_ax=None, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if secondary_ax is not None:
        if isinstance(secondary_ax, (list, tuple)):
            secondary_axes = secondary_ax
        else:
            secondary_axes = [secondary_ax]
        for item in secondary_axes:
            if item is None:
                continue
            secondary_handles, secondary_labels = item.get_legend_handles_labels()
            handles.extend(secondary_handles)
            labels.extend(secondary_labels)
    if handles:
        ax.legend(handles, labels, **kwargs)


def correct_solution_label(
    step: int,
    correct_solution_rates: Optional[CorrectSolutionRates],
) -> Optional[str]:
    if not correct_solution_rates:
        return None

    parts = []
    for split in ("ind", "ood"):
        rate = correct_solution_rates.get(step, {}).get(split)
        if rate is not None:
            parts.append(f"{SPLIT_LABELS[split]}={rate:.4f}")
    return ", ".join(parts) if parts else None


def annotate_correct_solution(
    ax,
    step: int,
    correct_solution_rates: Optional[CorrectSolutionRates],
) -> None:
    label = correct_solution_label(step, correct_solution_rates)
    if label is None:
        return

    ax.text(
        0.99,
        0.02,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
        },
    )


def plot_matrix_group(
    df: pd.DataFrame,
    matrix_name: str,
    output_dir: Path,
    yscale: str,
    dpi: int,
    mean_only: bool,
    correct_solution_rates: Optional[CorrectSolutionRates],
    hse_axis_label: str = "HSE",
) -> Optional[Path]:
    matrix_df = df[df["matrix"] == matrix_name].copy()
    if matrix_df.empty:
        print(f"[WARN] No rows for matrix={matrix_name}; skipping.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    layer_count = matrix_df["layer"].nunique()

    if not mean_only:
        cmap = plt.get_cmap("tab20")
        for idx, (layer, layer_df) in enumerate(matrix_df.groupby("layer", sort=True)):
            layer_df = layer_df.sort_values("step")
            ax.plot(
                layer_df["step"],
                layer_df["hse"],
                marker="o",
                linewidth=1.2,
                markersize=3.5,
                alpha=0.72,
                color=cmap(idx % cmap.N),
                label=f"layer {layer}",
            )

    mean_df = (
        matrix_df.groupby("step", as_index=False)["hse"]
        .mean()
        .sort_values("step")
    )
    ax.plot(
        mean_df["step"],
        mean_df["hse"],
        color="black",
        marker="o",
        linewidth=2.4,
        markersize=4.5,
        label="mean",
        zorder=10,
    )

    steps = to_int_list(sorted(matrix_df["step"].unique()))
    ax.set_title(f"{hse_axis_label} vs checkpoint step: {matrix_name}")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(hse_axis_label)
    ax.set_yscale(yscale)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_xticks(steps)
    if len(steps) > 8:
        ax.tick_params(axis="x", labelrotation=45)

    metric_ax = add_correct_solution_axis(ax, steps, correct_solution_rates)
    if mean_only:
        set_combined_legend(ax, metric_ax, loc="best")
    elif layer_count <= 14:
        set_combined_legend(ax, metric_ax, loc="best", ncol=2, fontsize=8)
    else:
        set_combined_legend(
            ax,
            metric_ax,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=2,
            fontsize=7,
            frameon=False,
        )

    output_path = output_dir / f"hse_{safe_filename(matrix_name)}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_param(
    param_df: pd.DataFrame,
    output_dir: Path,
    yscale: str,
    dpi: int,
    correct_solution_rates: Optional[CorrectSolutionRates],
    hse_axis_label: str = "HSE",
) -> Path:
    param_df = param_df.sort_values("step")
    first = param_df.iloc[0]
    layer = int(first["layer"])
    block = str(first["block"])
    matrix_name = str(first["matrix"])
    param_name = str(first["param_name"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        param_df["step"],
        param_df["hse"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        markersize=4.5,
        label=hse_axis_label,
    )

    steps = to_int_list(sorted(param_df["step"].unique()))
    ax.set_title(f"{hse_axis_label} vs checkpoint step: layer {layer} {matrix_name}")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(hse_axis_label)
    ax.set_yscale(yscale)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_xticks(steps)
    if len(steps) > 8:
        ax.tick_params(axis="x", labelrotation=45)
    metric_ax = add_correct_solution_axis(ax, steps, correct_solution_rates)
    if metric_ax is not None:
        set_combined_legend(ax, metric_ax, loc="best")
    ax.text(
        0.01,
        0.99,
        param_name,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#444444",
    )

    output_path = output_dir / f"layer_{layer:02d}_{safe_filename(block)}_{safe_filename(matrix_name)}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_matrix_layers(
    df: pd.DataFrame,
    matrix_name: str,
    output_dir: Path,
    yscale: str,
    dpi: int,
    correct_solution_rates: Optional[CorrectSolutionRates],
    hse_axis_label: str = "HSE",
) -> Optional[Path]:
    matrix_df = df[df["matrix"] == matrix_name].copy()
    if matrix_df.empty:
        print(f"[WARN] No rows for matrix={matrix_name}; skipping.")
        return None

    layers = to_int_list(sorted(matrix_df["layer"].unique()))
    steps = to_int_list(sorted(matrix_df["step"].unique()))
    ncols = 4
    nrows = (len(layers) + ncols - 1) // ncols
    fig_width = 4.0 * ncols
    fig_height = max(2.4 * nrows, 3.2)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        sharex=True,
        squeeze=False,
    )
    metric_legend_axes = None

    for idx, layer in enumerate(layers):
        ax = axes[idx // ncols][idx % ncols]
        layer_df = matrix_df[matrix_df["layer"] == layer].sort_values("step")
        ax.plot(
            layer_df["step"],
            layer_df["hse"],
            color="#1f77b4",
            marker="o",
            linewidth=1.6,
            markersize=3.5,
        )
        ax.set_title(f"layer {layer:02d}", fontsize=9)
        ax.set_yscale(yscale)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_xticks(steps)
        if len(steps) > 8:
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

        metric_ax = add_correct_solution_axis(
            ax,
            steps,
            correct_solution_rates,
            linewidth=1.0,
            markersize=2.5,
            show_axis=(idx % ncols == ncols - 1),
            tick_labelsize=6,
        )
        if metric_ax is not None:
            if metric_legend_axes is None:
                metric_legend_axes = metric_ax

    for idx in range(len(layers), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if metric_legend_axes is not None:
        handles = []
        labels = []
        for metric_ax in metric_legend_axes:
            metric_handles, metric_labels = metric_ax.get_legend_handles_labels()
            handles.extend(metric_handles)
            labels.extend(metric_labels)
        fig.legend(
            handles,
            labels,
            loc="upper right",
            ncol=2,
            fontsize=8,
            frameon=True,
            bbox_to_anchor=(0.99, 0.995),
        )

    fig.suptitle(f"{hse_axis_label} vs checkpoint step by layer: {matrix_name}", fontsize=14)
    fig.supxlabel("Checkpoint step")
    fig.supylabel(hse_axis_label)
    output_path = output_dir / f"hse_layers_{safe_filename(matrix_name)}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def heatmap_values(
    df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame],
    step: int,
    matrices: Tuple[str, ...],
    layers: list,
    mode: str,
    baseline_step: int,
) -> pd.DataFrame:
    step_df = df[df["step"] == step]
    values = step_df.pivot_table(index="matrix", columns="layer", values="hse", aggfunc="mean")
    values = values.reindex(index=list(matrices), columns=layers)

    if mode == "raw":
        return values

    if mode == "relative-layer-mean":
        row_mean = values.mean(axis=1).replace(0, float("nan"))
        return values.sub(row_mean, axis=0).div(row_mean, axis=0)

    if mode == "zscore-layer":
        row_mean = values.mean(axis=1)
        row_std = values.std(axis=1).replace(0, float("nan"))
        return values.sub(row_mean, axis=0).div(row_std, axis=0)

    if mode == "relative-baseline":
        base_df = df[df["step"] == baseline_step]
        baseline = base_df.pivot_table(index="matrix", columns="layer", values="hse", aggfunc="mean")
        baseline = baseline.reindex(index=list(matrices), columns=layers).replace(0, float("nan"))
        return values.sub(baseline).div(baseline)

    if mode == "relative-base":
        if baseline_df is None:
            raise ValueError("--heatmap-value relative-base requires --baseline-csv.")
        baseline = baseline_df.pivot_table(index="matrix", columns="layer", values="hse", aggfunc="mean")
        baseline = baseline.reindex(index=list(matrices), columns=layers).replace(0, float("nan"))
        return values.sub(baseline).div(baseline)

    raise ValueError(f"Unknown heatmap mode: {mode}")


def heatmap_limits(values: pd.DataFrame, mode: str, vmin: Optional[float], vmax: Optional[float]):
    if vmin is not None or vmax is not None:
        return vmin, vmax
    if mode == "raw":
        return None, None

    max_abs = values.abs().max().max()
    if pd.isna(max_abs) or float(max_abs) == 0:
        max_abs = 1.0
    return -float(max_abs), float(max_abs)


def heatmap_colorbar_label(mode: str, baseline_step: int, hse_axis_label: str) -> str:
    if mode == "raw":
        return hse_axis_label
    if mode == "relative-layer-mean":
        return f"({hse_axis_label} - row mean) / row mean"
    if mode == "relative-baseline":
        return f"({hse_axis_label} - step {baseline_step}) / step {baseline_step}"
    if mode == "relative-base":
        return f"({hse_axis_label} - base) / base"
    if mode == "zscore-layer":
        return "Layer z-score"
    return mode


def set_heatmap_colorbar_unit(cbar) -> None:
    cbar.formatter = FuncFormatter(lambda value, _: f"{value / HEATMAP_TICK_UNIT:g}")
    cbar.update_ticks()


def plot_heatmap(
    df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame],
    step: int,
    matrices: Tuple[str, ...],
    layers: list,
    output_dir: Path,
    mode: str,
    baseline_step: int,
    vmin: Optional[float],
    vmax: Optional[float],
    dpi: int,
    correct_solution_rates: Optional[CorrectSolutionRates],
    hse_axis_label: str = "HSE",
    hse_transform: str = "raw",
) -> Path:
    values = heatmap_values(
        df=df,
        baseline_df=baseline_df,
        step=step,
        matrices=matrices,
        layers=layers,
        mode=mode,
        baseline_step=baseline_step,
    )
    plot_vmin, plot_vmax = heatmap_limits(values, mode=mode, vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")

    fig_width = max(6.4, 0.32 * len(layers) + 2.0)
    fig_height = max(3.8, 0.42 * len(matrices) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(
        values.to_numpy(dtype=float),
        aspect="auto",
        cmap=cmap,
        vmin=plot_vmin,
        vmax=plot_vmax,
        interpolation="nearest",
    )

    x_positions = list(range(len(layers)))
    if len(layers) <= 12:
        tick_positions = x_positions
    else:
        tick_positions = [idx for idx, layer in enumerate(layers) if layer % 4 == 0]
        if x_positions[-1] not in tick_positions:
            tick_positions.append(x_positions[-1])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(layers[idx]) for idx in tick_positions])
    ax.set_yticks(range(len(matrices)))
    ax.set_yticklabels([MATRIX_LABELS.get(matrix, matrix) for matrix in matrices], fontsize=12)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("Layer Depth", fontsize=13, fontweight="bold")
    ax.set_title(f"{hse_axis_label} heatmap, checkpoint step {step}", fontsize=12)
    annotate_correct_solution(ax, step, correct_solution_rates)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.3, fraction=0.08)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    if hse_transform == "raw":
        set_heatmap_colorbar_unit(cbar)
        colorbar_label = f"{heatmap_colorbar_label(mode, baseline_step, hse_axis_label)} (1e-6)"
    else:
        colorbar_label = heatmap_colorbar_label(mode, baseline_step, hse_axis_label)
    cbar.set_label(colorbar_label, labelpad=8)

    output_path = output_dir / f"hse_heatmap_step_{step}_{safe_filename(mode)}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Qwen matrix-level HSE curves from hse_matrices.csv."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("output/train_ckpt/gp_l_sft-slight-qwen/hse_analysis/hse_matrices.csv"),
        help="Input CSV generated by compute_hse_qwen_matrices.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output directory. Defaults to <csv parent>/plots.",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help=(
            "Base model HSE CSV for --heatmap-value relative-base. "
            "Defaults to <csv parent>/hse_base_matrices.csv."
        ),
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Checkpoint steps to plot, e.g. 500 or 500,1000-2000. Default: all.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Layer ids to plot, e.g. 0 or 0,3-7. Default: all.",
    )
    parser.add_argument(
        "--matrices",
        default="all",
        help="Comma-separated matrix labels, or all. Default: all matrices present in the CSV.",
    )
    parser.add_argument(
        "--yscale",
        choices=("linear", "log"),
        default="linear",
        help="Y-axis scale. Default: linear.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG DPI. Default: 200.")
    parser.add_argument(
        "--plot-level",
        choices=("matrix", "matrix-layers", "param", "heatmap", "both", "all"),
        default="all",
        help=(
            "matrix: one overview per matrix type with layer curves; "
            "matrix-layers: one multi-subplot figure per matrix type with one subplot per layer; "
            "param: one plot per concrete layer matrix; "
            "heatmap: one matrix-by-layer heatmap per checkpoint step; "
            "both: matrix and param plots; all: generate all plot types. Default: all."
        ),
    )
    parser.add_argument(
        "--heatmap-value",
        choices=("relative-base", "relative-layer-mean", "relative-baseline", "raw", "zscore-layer"),
        default="relative-base",
        help=(
            "Value shown in heatmaps. relative-base compares checkpoints to the untrained base model; "
            "relative-layer-mean centers each matrix row within each step; "
            "relative-baseline compares each step to --baseline-step; raw uses HSE directly. "
            "Default: relative-base."
        ),
    )
    parser.add_argument(
        "--hse-transform",
        choices=HSE_TRANSFORMS,
        default="raw",
        help=(
            "Transform used only for plotting HSE values. raw preserves original values; "
            "robust-log plots log1p(HSE / median HSE) to reduce extreme-value dominance "
            "without dropping outliers. Default: raw."
        ),
    )
    parser.add_argument(
        "--baseline-step",
        type=int,
        default=None,
        help="Baseline checkpoint step for --heatmap-value relative-baseline. Default: first plotted step.",
    )
    parser.add_argument("--heatmap-vmin", type=float, default=None, help="Optional heatmap color minimum.")
    parser.add_argument("--heatmap-vmax", type=float, default=None, help="Optional heatmap color maximum.")
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="Only plot the mean across selected layers for each matrix.",
    )
    parser.add_argument(
        "--plot-correct-solution",
        action="store_true",
        help=(
            "Overlay or annotate ID/OOD CORRECT_SOLUTION rates parsed from "
            "--eval-checkpoint-root. Default: disabled."
        ),
    )
    parser.add_argument(
        "--eval-checkpoint-root",
        type=Path,
        default=None,
        help=(
            "Checkpoint root containing checkpoint-*/eval_res/*_ind*.jsonl "
            "and *_ood*.jsonl. Defaults to <csv parent>/.. when "
            "--plot-correct-solution is enabled."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.csv.parent / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_hse_csv(args.csv)
    steps = parse_int_spec(args.steps, "steps")
    layers = parse_int_spec(args.layers, "layers")
    matrix_spec = parse_matrix_spec(args.matrices)
    matrices = matrix_spec if matrix_spec is not None else infer_matrices(raw_df)
    df = apply_filters(raw_df, steps=steps, layers=layers, matrices=matrices)
    baseline_df = None
    baseline_csv = args.baseline_csv or (args.csv.parent / "hse_base_matrices.csv")
    if args.heatmap_value == "relative-base" and args.plot_level in {"heatmap", "all"}:
        baseline_df = filter_baseline(
            load_hse_csv(baseline_csv),
            layers=layers,
            matrices=matrices,
        )

    hse_axis_label = hse_label(args.hse_transform)
    if args.hse_transform != "raw":
        scale_source = pd.concat([df, baseline_df], ignore_index=True) if baseline_df is not None else df
        transform_scale = robust_hse_scale(scale_source)
        df = transform_hse_df(df, transform=args.hse_transform, scale=transform_scale)
        if baseline_df is not None:
            baseline_df = transform_hse_df(baseline_df, transform=args.hse_transform, scale=transform_scale)
        print(f"[INFO] HSE transform: {args.hse_transform}, scale={transform_scale:.12g}")
    else:
        print(f"[INFO] HSE transform: {args.hse_transform}")

    print(f"[INFO] input csv: {args.csv}")
    print(f"[INFO] output dir: {output_dir}")
    print(f"[INFO] rows: {len(df)}")
    if baseline_df is not None:
        print(f"[INFO] baseline csv: {baseline_csv}")
    plotted_steps = to_int_list(sorted(df["step"].unique()))
    plotted_layers = to_int_list(sorted(df["layer"].unique()))
    baseline_step = args.baseline_step if args.baseline_step is not None else plotted_steps[0]
    if args.heatmap_value == "relative-baseline" and baseline_step not in plotted_steps:
        raise ValueError(f"--baseline-step {baseline_step} is not included in plotted steps {plotted_steps}")
    print(f"[INFO] steps: {plotted_steps}")
    print(f"[INFO] layers: {plotted_layers}")
    correct_solution_rates = None
    if args.plot_correct_solution:
        eval_checkpoint_root = args.eval_checkpoint_root or args.csv.parent.parent
        correct_solution_rates = load_correct_solution_rates(
            checkpoint_root=eval_checkpoint_root,
            plotted_steps=set(plotted_steps),
        )
        loaded_count = sum(len(split_rates) for split_rates in correct_solution_rates.values())
        print(f"[INFO] eval checkpoint root: {eval_checkpoint_root}")
        print(f"[INFO] loaded CORRECT_SOLUTION values: {loaded_count}")

    written = []
    if args.plot_level in {"matrix", "both", "all"}:
        matrix_output_dir = output_dir / "by_matrix"
        matrix_output_dir.mkdir(parents=True, exist_ok=True)
        for matrix_name in matrices:
            output_path = plot_matrix_group(
                df=df,
                matrix_name=matrix_name,
                output_dir=matrix_output_dir,
                yscale=args.yscale,
                dpi=args.dpi,
                mean_only=args.mean_only,
                correct_solution_rates=correct_solution_rates,
                hse_axis_label=hse_axis_label,
            )
            if output_path is not None:
                written.append(output_path)
                print(f"[DONE] {output_path}")

    if args.plot_level in {"matrix-layers", "all"}:
        matrix_layers_output_dir = output_dir / "by_matrix_layers"
        matrix_layers_output_dir.mkdir(parents=True, exist_ok=True)
        for matrix_name in matrices:
            output_path = plot_matrix_layers(
                df=df,
                matrix_name=matrix_name,
                output_dir=matrix_layers_output_dir,
                yscale=args.yscale,
                dpi=args.dpi,
                correct_solution_rates=correct_solution_rates,
                hse_axis_label=hse_axis_label,
            )
            if output_path is not None:
                written.append(output_path)
                print(f"[DONE] {output_path}")

    if args.plot_level in {"param", "both", "all"}:
        param_output_dir = output_dir / "by_param"
        param_output_dir.mkdir(parents=True, exist_ok=True)
        for _, param_df in df.groupby("param_name", sort=True):
            output_path = plot_param(
                param_df=param_df,
                output_dir=param_output_dir,
                yscale=args.yscale,
                dpi=args.dpi,
                correct_solution_rates=correct_solution_rates,
                hse_axis_label=hse_axis_label,
            )
            written.append(output_path)
            print(f"[DONE] {output_path}")

    if args.plot_level in {"heatmap", "all"}:
        heatmap_output_dir = output_dir / "heatmaps"
        heatmap_output_dir.mkdir(parents=True, exist_ok=True)
        for step in plotted_steps:
            output_path = plot_heatmap(
                df=df,
                baseline_df=baseline_df,
                step=step,
                matrices=matrices,
                layers=plotted_layers,
                output_dir=heatmap_output_dir,
                mode=args.heatmap_value,
                baseline_step=baseline_step,
                vmin=args.heatmap_vmin,
                vmax=args.heatmap_vmax,
                dpi=args.dpi,
                correct_solution_rates=correct_solution_rates,
                hse_axis_label=hse_axis_label,
                hse_transform=args.hse_transform,
            )
            written.append(output_path)
            print(f"[DONE] {output_path}")

    if not written:
        raise RuntimeError("No plots were written.")


if __name__ == "__main__":
    main()
