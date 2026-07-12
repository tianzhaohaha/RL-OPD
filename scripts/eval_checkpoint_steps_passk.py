#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Evaluate pass@K for every global_step checkpoint in an OPD training archive.

The script stays inside the OPD project tree. It uses OPD/verl for generation
and OPD/scripts/val/eval/utils.py for math answer grading.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


SCRIPT_PATH = Path(__file__).resolve()
OPD_ROOT = SCRIPT_PATH.parents[1]
VERL_ROOT = OPD_ROOT / "verl"
VAL_UTILS_PATH = OPD_ROOT / "scripts" / "val" / "eval" / "utils.py"

sys.path.insert(0, str(VERL_ROOT))


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


@contextmanager
def timed_stage(name: str):
    start = time.perf_counter()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {name}...", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished {name} in {format_duration(elapsed)}", flush=True)


def progress_iter(iterable, total: int, desc: str, disable: bool = False):
    if disable:
        yield from iterable
        return
    try:
        from tqdm import tqdm

        yield from tqdm(iterable, total=total, desc=desc)
        return
    except ImportError:
        pass

    start = time.perf_counter()
    print_every = max(1, total // 20) if total > 0 else 1
    for index, item in enumerate(iterable, start=1):
        yield item
        if index == 1 or index == total or index % print_every == 0:
            elapsed = time.perf_counter() - start
            rate = index / elapsed if elapsed > 0 else 0.0
            eta = (total - index) / rate if rate > 0 else 0.0
            print(
                f"{desc}: {index}/{total} ({index / total:.1%}), "
                f"elapsed {format_duration(elapsed)}, eta {format_duration(eta)}",
                flush=True,
            )


def path_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def require_inside_opd(path: Path, label: str) -> Path:
    if not path.is_absolute():
        path = (OPD_ROOT / path) if not str(path).startswith("OPD/") else (OPD_ROOT.parent / path)
    resolved = path.expanduser().resolve()
    if not path_inside(resolved, OPD_ROOT):
        raise ValueError(f"{label} must be inside OPD root: {OPD_ROOT}\nGot: {resolved}")
    return resolved


def parse_k_values(raw_k: str) -> list[int]:
    k_values = sorted({int(item.strip()) for item in raw_k.split(",") if item.strip()})
    if not k_values or k_values[0] < 1:
        raise ValueError("--k must contain positive integers, for example: 1,2,4,8,16")
    return k_values


def parse_step_filter(raw_steps: str | None) -> set[int] | None:
    if raw_steps is None:
        return None
    steps: set[int] = set()
    for item in raw_steps.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if end < start:
                raise ValueError(f"Invalid step range: {item}")
            steps.update(range(start, end + 1))
        else:
            steps.add(int(item))
    return steps


def step_number(step_dir: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", step_dir.name)
    if match is None:
        raise ValueError(f"Not a global_step directory: {step_dir}")
    return int(match.group(1))


def discover_step_dirs(checkpoint_root: Path, step_filter: set[int] | None) -> list[Path]:
    step_dirs = [path for path in checkpoint_root.iterdir() if path.is_dir() and re.fullmatch(r"global_step_\d+", path.name)]
    step_dirs.sort(key=step_number)
    if step_filter is not None:
        step_dirs = [path for path in step_dirs if step_number(path) in step_filter]
    if not step_dirs:
        raise FileNotFoundError(f"No matching global_step_* directories found under {checkpoint_root}")
    return step_dirs


def has_hf_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def resolve_actor_dir(step_dir: Path) -> Path | None:
    actor_dir = step_dir / "actor"
    if (actor_dir / "fsdp_config.json").is_file():
        return actor_dir
    if (step_dir / "fsdp_config.json").is_file():
        return step_dir
    return None


def resolve_hf_model_dir(step_dir: Path) -> Path | None:
    candidates = [
        step_dir / "actor" / "huggingface",
        step_dir / "huggingface",
        step_dir / "actor",
        step_dir,
    ]
    for candidate in candidates:
        if has_hf_weights(candidate):
            return candidate.resolve()
    return None


def load_grade_answer() -> Callable[[str, str], bool]:
    if not VAL_UTILS_PATH.is_file():
        raise FileNotFoundError(f"Cannot find OPD validation utils: {VAL_UTILS_PATH}")
    spec = importlib.util.spec_from_file_location("opd_val_utils", VAL_UTILS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OPD validation utils: {VAL_UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grade_answer_verl


def data_task_name(data_file: Path) -> str:
    if data_file.parent.name:
        return data_file.parent.name
    return data_file.stem


def default_data_files() -> list[Path]:
    return [
        OPD_ROOT / "datasets" / "test_data" / "AIME25" / "test.parquet",
        OPD_ROOT / "datasets" / "test_data" / "AMC23" / "test.parquet",
        OPD_ROOT / "datasets" / "test_data" / "AIME24" / "test.parquet",
    ]


def read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def subset_data_file(args: argparse.Namespace, data_file: Path, task_name: str) -> Path:
    if args.num_samples is None:
        return data_file
    if args.num_samples < 1:
        raise ValueError("--num-samples must be a positive integer")

    dataset = read_parquet(data_file)
    if args.num_samples >= len(dataset):
        print(f"{task_name}: --num-samples={args.num_samples} >= dataset size {len(dataset)}; using all prompts.")
        return data_file

    if args.sample_seed is not None:
        subset = dataset.sample(n=args.num_samples, random_state=args.sample_seed).reset_index(drop=True)
        selection = f"random sample seed={args.sample_seed}"
    else:
        subset = dataset.head(args.num_samples).reset_index(drop=True)
        selection = "first N"

    subset_dir = args.output_dir / "data_subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = f"_seed{args.sample_seed}" if args.sample_seed is not None else ""
    subset_path = subset_dir / f"{task_name}__n{args.num_samples}{seed_tag}.parquet"
    if subset_path.is_file() and not args.overwrite:
        print(f"{task_name}: reusing data subset {subset_path}")
        return subset_path.resolve()
    subset.to_parquet(subset_path, index=False)
    print(f"{task_name}: using {args.num_samples} prompts ({selection}) -> {subset_path}")
    return subset_path.resolve()


def run_command(command: list[str], env: dict[str, str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(command)
    print(f"$ {command_text}", flush=True)
    if dry_run:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"DRY RUN: {command_text}\n")
        return

    start = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {command_text}\n")
        log_file.flush()
        subprocess.run(command, cwd=OPD_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] finished in {format_duration(time.perf_counter() - start)}\n")


def prepare_model(args: argparse.Namespace, step_dir: Path, run_name: str, env: dict[str, str], log_path: Path) -> Path:
    hf_model_dir = resolve_hf_model_dir(step_dir)
    if hf_model_dir is not None and not args.force_merge:
        return hf_model_dir

    actor_dir = resolve_actor_dir(step_dir)
    if actor_dir is None:
        raise FileNotFoundError(f"Cannot find actor FSDP checkpoint or HF model under {step_dir}")

    merged_dir = args.output_dir / "merged_hf" / run_name
    if has_hf_weights(merged_dir) and not args.overwrite:
        return merged_dir.resolve()

    merged_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(merged_dir),
        "--use_cpu_initialization",
    ]
    run_command(command, env, log_path, args.dry_run)
    return merged_dir.resolve()


def generate_responses(
    args: argparse.Namespace,
    model_path: Path,
    data_file: Path,
    generated_path: Path,
    max_k: int,
    env: dict[str, str],
    log_path: Path,
) -> None:
    if generated_path.is_file() and not args.overwrite and not args.dry_run:
        print(f"Reusing generated parquet: {generated_path}")
        return
    if max_k > 1 and args.temperature <= 0:
        raise ValueError("Generation with K > 1 requires --temperature > 0")

    generated_path.parent.mkdir(parents=True, exist_ok=True)
    max_model_len = args.max_model_len or (args.max_prompt_length + args.max_response_length)
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_generation_server",
        f"data.train_files=['{data_file}']",
        f"+data.output_path={generated_path}",
        f"data.prompt_key={args.prompt_key}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"actor_rollout_ref.model.path={model_path}",
        f"actor_rollout_ref.rollout.name={args.backend}",
        f"actor_rollout_ref.rollout.n={max_k}",
        f"actor_rollout_ref.rollout.temperature={args.temperature}",
        f"actor_rollout_ref.rollout.top_p={args.top_p}",
        f"actor_rollout_ref.rollout.response_length={args.max_response_length}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        f"actor_rollout_ref.rollout.enforce_eager={args.enforce_eager}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tp}",
        f"trainer.n_gpus_per_node={args.gpus}",
        "trainer.nnodes=1",
    ]
    if args.num_cpus is not None:
        command.append(f"+ray_kwargs.ray_init.num_cpus={args.num_cpus}")
    command.extend(args.extra_generate_arg)
    run_command(command, env, log_path, args.dry_run)


def score_response(grade_answer: Callable[[str, str], bool], response: str, reward_data: Any) -> float:
    if isinstance(reward_data, dict):
        ground_truth = reward_data.get("ground_truth")
    else:
        ground_truth = reward_data["ground_truth"]
    return 1.0 if grade_answer(str(response), str(ground_truth)) else 0.0


def compute_passk(
    args: argparse.Namespace,
    generated_path: Path,
    metrics_path: Path,
    k_values: list[int],
    step: int,
    task_name: str,
    data_file: Path,
    model_path: Path,
    grade_answer: Callable[[str, str], bool],
) -> dict[str, Any]:
    import numpy as np

    if metrics_path.is_file() and not args.overwrite:
        with metrics_path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    dataset = read_parquet(generated_path)
    required_columns = [args.response_key, args.reward_model_key]
    missing = [column for column in required_columns if column not in dataset.columns]
    if missing:
        raise KeyError(f"Generated parquet is missing required columns: {missing}")

    max_k = max(k_values)
    score_rows: list[list[float]] = []
    missing_boxed = 0
    total_responses = 0

    row_iter = progress_iter(
        dataset.iterrows(),
        total=len(dataset),
        desc=f"Scoring step {step} {task_name}",
        disable=args.no_progress,
    )
    for row_index, row in row_iter:
        responses = row[args.response_key]
        if isinstance(responses, np.ndarray):
            responses = responses.tolist()
        if not isinstance(responses, list):
            responses = list(responses)
        if len(responses) < max_k:
            raise ValueError(f"Row {row_index} has {len(responses)} responses, but max K is {max_k}")

        selected_responses = [str(response) for response in responses[:max_k]]
        total_responses += len(selected_responses)
        missing_boxed += sum("\\boxed" not in response for response in selected_responses)
        scores = [score_response(grade_answer, response, row[args.reward_model_key]) for response in selected_responses]
        score_rows.append(scores)

    score_array = np.asarray(score_rows, dtype=np.float32)
    overall: dict[str, float] = {}
    for k in k_values:
        pass_values = np.max(score_array[:, :k], axis=1)
        mean_values = np.mean(score_array[:, :k], axis=1)
        overall[f"pass@{k}"] = float(np.mean(pass_values))
        overall[f"mean@{k}"] = float(np.mean(mean_values))

    metrics = {
        "step": step,
        "task": task_name,
        "data_file": str(data_file),
        "model_path": str(model_path),
        "generated_path": str(generated_path),
        "num_prompts": int(len(dataset)),
        "num_responses": int(total_responses),
        "missing_boxed_responses": int(missing_boxed),
        "k_values": k_values,
        "overall": overall,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def aggregate_results(results: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    by_step: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") != "ok":
            continue
        by_step.setdefault(int(result["step"]), []).append(result)

    for step, step_results in sorted(by_step.items()):
        total_prompts = sum(int(item.get("num_prompts", 0)) for item in step_results)
        metrics: dict[str, float | int] = {"num_prompts": total_prompts, "num_tasks": len(step_results)}
        if total_prompts > 0:
            for k in k_values:
                for metric_name in (f"pass@{k}", f"mean@{k}"):
                    weighted_sum = sum(float(item.get(metric_name, 0.0)) * int(item.get("num_prompts", 0)) for item in step_results)
                    metrics[metric_name] = weighted_sum / total_prompts
        aggregate[str(step)] = metrics
    return aggregate


def sanitize_plot_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def metric_names_for_plot(k_values: list[int]) -> list[str]:
    return [f"pass@{k}" for k in k_values] + [f"mean@{k}" for k in k_values]


def write_plot_data_csv(
    output_path: Path,
    results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    metric_names: list[str],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["row_type", "step", "task", "metric", "value"])
        writer.writeheader()
        for step, metrics in sorted(aggregate.items(), key=lambda item: int(item[0])):
            for metric_name in metric_names:
                if metric_name in metrics:
                    writer.writerow(
                        {
                            "row_type": "aggregate",
                            "step": step,
                            "task": "ALL",
                            "metric": metric_name,
                            "value": metrics[metric_name],
                        }
                    )
        for result in sorted(results, key=lambda item: (int(item.get("step", 0)), str(item.get("task", "")))):
            if result.get("status") != "ok":
                continue
            for metric_name in metric_names:
                if metric_name in result:
                    writer.writerow(
                        {
                            "row_type": "task",
                            "step": result["step"],
                            "task": result.get("task", ""),
                            "metric": metric_name,
                            "value": result[metric_name],
                        }
                    )


def plot_metric_curves(args: argparse.Namespace, results: list[dict[str, Any]], k_values: list[int]) -> list[str]:
    if args.no_plot:
        print("Plotting disabled by --no-plot")
        return []

    aggregate = aggregate_results(results, k_values)
    if not aggregate:
        print("No successful metrics to plot.")
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Plotting skipped because matplotlib is unavailable: {exc}", file=sys.stderr)
        return []

    metric_names = metric_names_for_plot(k_values)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_data_csv = args.output_dir / "metrics_over_steps_plot_data.csv"
    write_plot_data_csv(plot_data_csv, results, aggregate, metric_names)

    saved_paths: list[Path] = [plot_data_csv]
    aggregate_steps = sorted(int(step) for step in aggregate)
    task_names = sorted({str(result.get("task")) for result in results if result.get("status") == "ok"})

    def aggregate_values(metric_name: str) -> tuple[list[int], list[float]]:
        steps: list[int] = []
        values: list[float] = []
        for step in aggregate_steps:
            metrics = aggregate[str(step)]
            if metric_name in metrics:
                steps.append(step)
                values.append(float(metrics[metric_name]))
        return steps, values

    def task_values(task_name: str, metric_name: str) -> tuple[list[int], list[float]]:
        rows = [
            result
            for result in results
            if result.get("status") == "ok" and str(result.get("task")) == task_name and metric_name in result
        ]
        rows.sort(key=lambda item: int(item["step"]))
        return [int(row["step"]) for row in rows], [float(row[metric_name]) for row in rows]

    for metric_name in metric_names:
        fig, ax = plt.subplots(figsize=(9, 5))
        steps, values = aggregate_values(metric_name)
        if steps:
            ax.plot(steps, values, marker="o", linewidth=2.4, label="ALL")
        for task_name in task_names:
            task_steps, task_metric_values = task_values(task_name, metric_name)
            if task_steps:
                ax.plot(task_steps, task_metric_values, marker="o", linewidth=1.6, label=task_name)
        ax.set_title(f"{metric_name} over training steps")
        ax.set_xlabel("global step")
        ax.set_ylabel(metric_name)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        output_path = plots_dir / f"{sanitize_plot_name(metric_name)}_over_steps.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        saved_paths.append(output_path)

    ncols = 2
    nrows = (len(metric_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, 3.4 * nrows)))
    axes_flat = list(getattr(axes, "flat", [axes]))
    for ax, metric_name in zip(axes_flat, metric_names):
        steps, values = aggregate_values(metric_name)
        ax.plot(steps, values, marker="o", linewidth=2.0)
        ax.set_title(metric_name)
        ax.set_xlabel("global step")
        ax.set_ylabel(metric_name)
        ax.grid(True, linestyle="--", alpha=0.35)
    for ax in axes_flat[len(metric_names) :]:
        ax.axis("off")
    fig.suptitle("Aggregate metrics over training steps", y=1.0)
    fig.tight_layout()
    aggregate_plot = args.output_dir / "aggregate_metrics_over_steps.png"
    fig.savefig(aggregate_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(aggregate_plot)

    if task_names:
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, 3.4 * nrows)))
        axes_flat = list(getattr(axes, "flat", [axes]))
        for ax, metric_name in zip(axes_flat, metric_names):
            for task_name in task_names:
                task_steps, task_metric_values = task_values(task_name, metric_name)
                if task_steps:
                    ax.plot(task_steps, task_metric_values, marker="o", linewidth=1.6, label=task_name)
            ax.set_title(metric_name)
            ax.set_xlabel("global step")
            ax.set_ylabel(metric_name)
            ax.grid(True, linestyle="--", alpha=0.35)
        for ax in axes_flat[len(metric_names) :]:
            ax.axis("off")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.02))
        fig.suptitle("Per-task metrics over training steps", y=1.08)
        fig.tight_layout()
        task_plot = args.output_dir / "task_metrics_over_steps.png"
        fig.savefig(task_plot, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(task_plot)

    plot_index = args.output_dir / "plots.json"
    plot_index.write_text(
        json.dumps({"plot_files": [str(path) for path in saved_paths]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    saved_paths.append(plot_index)

    for path in saved_paths:
        print(f"Saved plot artifact: {path}")
    return [str(path) for path in saved_paths]


def write_summaries(args: argparse.Namespace, results: list[dict[str, Any]], k_values: list[int]) -> None:
    aggregate = aggregate_results(results, k_values)
    summary = {
        "checkpoint_root": str(args.checkpoint_root),
        "output_dir": str(args.output_dir),
        "data_files": [str(path) for path in args.data_files],
        "k_values": k_values,
        "num_results": len(results),
        "num_failures": sum(1 for result in results if result.get("status") == "failed"),
        "num_dry_run": sum(1 for result in results if result.get("status") == "dry_run"),
        "aggregate_by_step": aggregate,
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_dir / "summary.json"
    summary_csv = args.output_dir / "summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metric_columns = [f"pass@{k}" for k in k_values] + [f"mean@{k}" for k in k_values]
    fieldnames = [
        "step",
        "row_type",
        "task",
        "status",
        "num_prompts",
        *metric_columns,
        "metrics_path",
        "generated_path",
        "log_path",
        "model_path",
        "error",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key, "") for key in fieldnames}
            row["row_type"] = "task"
            writer.writerow(row)
        for step, metrics in aggregate.items():
            row = {key: "" for key in fieldnames}
            row.update({"step": step, "row_type": "aggregate", "task": "ALL", "status": "ok"})
            row["num_prompts"] = metrics.get("num_prompts", "")
            for metric_name in metric_columns:
                row[metric_name] = metrics.get(metric_name, "")
            writer.writerow(row)

    print(f"\nSaved summary JSON: {summary_json}")
    print(f"Saved summary CSV: {summary_csv}")


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(VERL_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        visible_gpu_count = len([item for item in args.cuda_visible_devices.split(",") if item.strip()])
        if visible_gpu_count > 0 and args.gpus > visible_gpu_count:
            raise ValueError(f"--gpus={args.gpus} exceeds --cuda-visible-devices={args.cuda_visible_devices}")
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pass@K for all OPD global_step checkpoints.")
    parser.add_argument("checkpoint_root", type=Path, help="OPD training archive containing global_step_* directories.")
    parser.add_argument(
        "--data-file",
        dest="data_files",
        type=Path,
        action="append",
        default=None,
        help="Validation parquet file. Can be repeated. Default: AIME25, AMC23, AIME24 under OPD/datasets/test_data.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <checkpoint_root>/passk_eval_all_steps")
    parser.add_argument("--steps", type=str, default=None, help="Comma/range filter, for example: 50,100,200-300. Default: all steps.")
    parser.add_argument("--k", type=str, default="1,2,4,8,16", help="Comma-separated pass@K values.")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "sglang", "hf", "trtllm"])
    parser.add_argument("--cuda-visible-devices", type=str, default=None, help="Physical GPU ids to expose, for example: 5 or 5,6.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of visible GPUs used by generation.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size for generation.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=7168)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-key", type=str, default="prompt")
    parser.add_argument("--response-key", type=str, default="responses")
    parser.add_argument("--reward-model-key", type=str, default="reward_model")
    parser.add_argument("--num-samples", type=int, default=None, help="Evaluate only N prompts per dataset.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Random sample seed when --num-samples is set.")
    parser.add_argument("--num-cpus", type=int, default=None, help="Ray num_cpus override for generation.")
    parser.add_argument("--force-merge", action="store_true", help="Merge FSDP actor even if actor/huggingface already exists.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs and rewrite per-task metrics.")
    parser.add_argument("--dry-run", action="store_true", help="Write planned commands and summary without running generation.")
    parser.add_argument("--no-plot", action="store_true", help="Skip metric-over-step plot generation after evaluation.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars and periodic progress messages.")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extra-generate-arg", action="append", default=[], help="Extra Hydra override passed to generation.")
    args = parser.parse_args()

    args.checkpoint_root = require_inside_opd(args.checkpoint_root, "checkpoint_root")
    if not args.checkpoint_root.is_dir():
        raise FileNotFoundError(f"checkpoint_root does not exist: {args.checkpoint_root}")
    args.output_dir = require_inside_opd(args.output_dir or (args.checkpoint_root / "passk_eval_all_steps"), "output_dir")
    raw_data_files = args.data_files or default_data_files()
    args.data_files = [require_inside_opd(path, "data_file") for path in raw_data_files]
    for data_file in args.data_files:
        if not data_file.is_file():
            raise FileNotFoundError(f"Validation data file does not exist: {data_file}")
    return args


def evaluate_one(args: argparse.Namespace, step_dir: Path, data_file: Path, k_values: list[int], env: dict[str, str], grade_answer):
    step = step_number(step_dir)
    task_name = data_task_name(data_file)
    run_name = f"global_step_{step}_{task_name}"
    max_k = max(k_values)
    log_path = args.output_dir / "logs" / f"{run_name}.log"
    generated_path = args.output_dir / "generations" / f"{run_name}__k{max_k}.parquet"
    metrics_path = args.output_dir / "metrics" / f"{run_name}__k{max_k}_passk.json"

    result: dict[str, Any] = {
        "step": step,
        "task": task_name,
        "step_dir": str(step_dir),
        "data_file": str(data_file),
        "generated_path": str(generated_path),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
    }

    with timed_stage(f"step {step} {task_name}: model preparation"):
        model_path = prepare_model(args, step_dir, run_name, env, log_path)
    result["model_path"] = str(model_path)

    data_for_generation = data_file if args.dry_run else subset_data_file(args, data_file, task_name)
    with timed_stage(f"step {step} {task_name}: generation"):
        generate_responses(args, model_path, data_for_generation, generated_path, max_k, env, log_path)
    if args.dry_run:
        result.update({"status": "dry_run"})
        return result

    with timed_stage(f"step {step} {task_name}: pass@K scoring"):
        metrics = compute_passk(args, generated_path, metrics_path, k_values, step, task_name, data_file, model_path, grade_answer)

    result.update({"status": "ok", "num_prompts": metrics.get("num_prompts", 0)})
    for metric_name, metric_value in metrics.get("overall", {}).items():
        result[metric_name] = metric_value
    return result


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    k_values = parse_k_values(args.k)
    step_filter = parse_step_filter(args.steps)
    step_dirs = discover_step_dirs(args.checkpoint_root, step_filter)
    env = build_env(args)
    grade_answer = load_grade_answer()

    print(f"OPD root: {OPD_ROOT}")
    print(f"Checkpoint root: {args.checkpoint_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Steps: {[step_number(path) for path in step_dirs]}")
    print(f"Data files: {[str(path) for path in args.data_files]}")
    print(f"K values: {k_values}")
    if args.cuda_visible_devices:
        print(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    results: list[dict[str, Any]] = []
    eval_jobs = [(step_dir, data_file) for step_dir in step_dirs for data_file in args.data_files]
    for step_dir, data_file in progress_iter(
        eval_jobs,
        total=len(eval_jobs),
        desc="Evaluating checkpoints",
        disable=args.no_progress,
    ):
        try:
            result = evaluate_one(args, step_dir, data_file, k_values, env, grade_answer)
        except Exception as exc:
            result = {
                "step": step_number(step_dir),
                "task": data_task_name(data_file),
                "step_dir": str(step_dir),
                "data_file": str(data_file),
                "status": "failed",
                "error": repr(exc),
            }
            print(f"FAILED step {result['step']} {result['task']}: {exc}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                results.append(result)
                write_summaries(args, results, k_values)
                raise
        results.append(result)
        write_summaries(args, results, k_values)

    with timed_stage("metric plotting"):
        plot_metric_curves(args, results, k_values)

    success_count = sum(1 for result in results if result.get("status") == "ok")
    failure_count = sum(1 for result in results if result.get("status") == "failed")
    print(f"Evaluation tasks: {success_count} succeeded, {failure_count} failed")
    if success_count == 0 and not args.dry_run:
        print(f"No successful evaluation metrics were produced. See logs under {args.output_dir / 'logs'}", file=sys.stderr)
        raise SystemExit(1)

    print(f"\nDone in {format_duration(time.perf_counter() - total_start)}")
    print(f"Summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
