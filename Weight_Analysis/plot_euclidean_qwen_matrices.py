#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_hse_qwen_matrices import (
    ALL_MATRICES,
    CorrectSolutionRates,
    DEFAULT_MATRICES,
    MATRIX_LABELS,
    MATRIX_ORDER,
    add_correct_solution_axis,
    annotate_correct_solution,
    load_correct_solution_rates,
    matrix_sort_key,
    parse_int_spec,
    parse_matrix_spec,
    safe_filename,
    set_combined_legend,
    to_int_list,
)


DISTANCE_METRICS = {
    "euclidean": {
        "column": "euclidean_distance",
        "label": "Euclidean distance to base",
        "prefix": "euclidean_distance",
    },
    "relative": {
        "column": "relative_euclidean_distance",
        "label": "Relative Euclidean distance to base",
        "prefix": "relative_euclidean_distance",
    },
    "rms": {
        "column": "rms_distance",
        "label": "RMS Euclidean distance to base",
        "prefix": "rms_euclidean_distance",
    },
}


def infer_matrices(df: pd.DataFrame) -> Tuple[str, ...]:
    return tuple(sorted((str(value) for value in df["matrix"].unique()), key=matrix_sort_key))


def load_distance_csv(csv_path: Path, distance_column: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Euclidean-distance CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"step", "layer", "block", "matrix", "param_name", distance_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["step"] = pd.to_numeric(df["step"], errors="raise").astype(int)
    df["layer"] = pd.to_numeric(df["layer"], errors="raise").astype(int)
    df[distance_column] = pd.to_numeric(df[distance_column], errors="raise")
    return df


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


def plot_matrix_group(
    df: pd.DataFrame,
    matrix_name: str,
    output_dir: Path,
    yscale: str,
    dpi: int,
    mean_only: bool,
    correct_solution_rates: Optional[CorrectSolutionRates],
    distance_column: str,
    distance_label: str,
    output_prefix: str,
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
                layer_df[distance_column],
                marker="o",
                linewidth=1.2,
                markersize=3.5,
                alpha=0.72,
                color=cmap(idx % cmap.N),
                label=f"layer {layer}",
            )

    mean_df = (
        matrix_df.groupby("step", as_index=False)[distance_column]
        .mean()
        .sort_values("step")
    )
    ax.plot(
        mean_df["step"],
        mean_df[distance_column],
        color="black",
        marker="o",
        linewidth=2.4,
        markersize=4.5,
        label="mean",
        zorder=10,
    )

    steps = to_int_list(sorted(matrix_df["step"].unique()))
    ax.set_title(f"{distance_label} vs checkpoint step: {matrix_name}")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(distance_label)
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

    output_path = output_dir / f"{output_prefix}_{safe_filename(matrix_name)}.png"
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
    distance_column: str,
    distance_label: str,
    output_prefix: str,
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
        param_df[distance_column],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        markersize=4.5,
        label=distance_label,
    )

    steps = to_int_list(sorted(param_df["step"].unique()))
    ax.set_title(f"{distance_label}: layer {layer} {matrix_name}")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(distance_label)
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

    output_path = output_dir / f"{output_prefix}_layer_{layer:02d}_{safe_filename(block)}_{safe_filename(matrix_name)}.png"
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
    distance_column: str,
    distance_label: str,
    output_prefix: str,
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
            layer_df[distance_column],
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
        if metric_ax is not None and metric_legend_axes is None:
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

    fig.suptitle(f"{distance_label} by layer: {matrix_name}", fontsize=14)
    fig.supxlabel("Checkpoint step")
    fig.supylabel(distance_label)
    output_path = output_dir / f"{output_prefix}_layers_{safe_filename(matrix_name)}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def heatmap_values(
    df: pd.DataFrame,
    step: int,
    matrices: Tuple[str, ...],
    layers: list,
    mode: str,
    baseline_step: int,
    distance_column: str,
) -> pd.DataFrame:
    step_df = df[df["step"] == step]
    values = step_df.pivot_table(index="matrix", columns="layer", values=distance_column, aggfunc="mean")
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
        baseline = base_df.pivot_table(index="matrix", columns="layer", values=distance_column, aggfunc="mean")
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


def heatmap_colorbar_label(mode: str, baseline_step: int, distance_label: str) -> str:
    if mode == "raw":
        return distance_label
    if mode == "relative-layer-mean":
        return f"({distance_label} - row mean) / row mean"
    if mode == "relative-baseline":
        return f"({distance_label} - step {baseline_step}) / step {baseline_step}"
    if mode == "zscore-layer":
        return "Layer z-score"
    return mode


def plot_heatmap(
    df: pd.DataFrame,
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
    distance_column: str,
    distance_label: str,
    output_prefix: str,
) -> Path:
    values = heatmap_values(
        df=df,
        step=step,
        matrices=matrices,
        layers=layers,
        mode=mode,
        baseline_step=baseline_step,
        distance_column=distance_column,
    )
    plot_vmin, plot_vmax = heatmap_limits(values, mode=mode, vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("RdBu_r").copy() if mode == "raw" else plt.get_cmap("viridis").copy()
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
    ax.set_title(f"{distance_label} heatmap, checkpoint step {step}", fontsize=12)
    annotate_correct_solution(ax, step, correct_solution_rates)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.3, fraction=0.08)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label(heatmap_colorbar_label(mode, baseline_step, distance_label), labelpad=8)

    output_path = output_dir / f"{output_prefix}_heatmap_step_{step}_{safe_filename(mode)}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Qwen matrix-level Euclidean distance to base from euclidean_distance_matrices.csv."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("output/train_ckpt/gp_l_sft-slight-qwen/euclidean_distance_analysis/euclidean_distance_matrices.csv"),
        help="Input CSV generated by compute_euclidean_qwen_matrices.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output directory. Defaults to <csv parent>/plots.",
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
    parser.add_argument(
        "--distance-metric",
        choices=tuple(DISTANCE_METRICS),
        default="euclidean",
        help=(
            "Distance metric to plot. euclidean uses raw ||W_ckpt-W_base||_2; "
            "relative uses ||W_ckpt-W_base||_2 / ||W_base||_2; "
            "rms uses ||W_ckpt-W_base||_2 / sqrt(numel). Default: euclidean."
        ),
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
        choices=("raw", "relative-layer-mean", "relative-baseline", "zscore-layer"),
        default="raw",
        help=(
            "Value shown in heatmaps. raw uses Euclidean distance to base directly; "
            "relative-layer-mean centers each matrix row within each step; "
            "relative-baseline compares each step to --baseline-step. Default: raw."
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

    metric_config = DISTANCE_METRICS[args.distance_metric]
    distance_column = metric_config["column"]
    distance_label = metric_config["label"]
    output_prefix = metric_config["prefix"]

    df = load_distance_csv(args.csv, distance_column=distance_column)
    steps = parse_int_spec(args.steps, "steps")
    layers = parse_int_spec(args.layers, "layers")
    matrix_spec = parse_matrix_spec(args.matrices)
    matrices = matrix_spec if matrix_spec is not None else infer_matrices(df)
    df = apply_filters(df, steps=steps, layers=layers, matrices=matrices)

    print(f"[INFO] input csv: {args.csv}")
    print(f"[INFO] output dir: {output_dir}")
    print(f"[INFO] distance metric: {args.distance_metric} ({distance_column})")
    print(f"[INFO] rows: {len(df)}")
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
                distance_column=distance_column,
                distance_label=distance_label,
                output_prefix=output_prefix,
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
                distance_column=distance_column,
                distance_label=distance_label,
                output_prefix=output_prefix,
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
                distance_column=distance_column,
                distance_label=distance_label,
                output_prefix=output_prefix,
            )
            written.append(output_path)
            print(f"[DONE] {output_path}")

    if args.plot_level in {"heatmap", "all"}:
        heatmap_output_dir = output_dir / "heatmaps"
        heatmap_output_dir.mkdir(parents=True, exist_ok=True)
        for step in plotted_steps:
            output_path = plot_heatmap(
                df=df,
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
                distance_column=distance_column,
                distance_label=distance_label,
                output_prefix=output_prefix,
            )
            written.append(output_path)
            print(f"[DONE] {output_path}")

    if not written:
        raise RuntimeError("No plots were written.")


if __name__ == "__main__":
    main()