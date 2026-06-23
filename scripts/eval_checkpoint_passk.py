#!/usr/bin/env python3
"""Evaluate pass@K for a verl checkpoint.

The script accepts either a verl FSDP actor checkpoint directory
(`.../global_step_xxx/actor`), a global step directory containing `actor`, or
an already merged HuggingFace model directory. It then:

1. Merges FSDP actor checkpoints to HuggingFace format when needed.
2. Generates max(K) responses per prompt with `verl.trainer.main_generation_server`.
3. Scores every response with verl's default reward function.
4. Reports pass@K as mean(max(score_1, ..., score_K)).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
VERL_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = VERL_ROOT.parent
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


def progress_iter(iterable, total: int, desc: str):
    try:
        from tqdm import tqdm

        yield from tqdm(iterable, total=total, desc=desc)
        return
    except ImportError:
        pass

    start = time.perf_counter()
    print_every = max(1, total // 20)
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline pass@K evaluation for verl checkpoints.")
    parser.add_argument(
        "checkpoint_path",
        type=Path,
        help="Path to .../global_step_xxx/actor, .../global_step_xxx, or a merged HuggingFace model directory.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=WORKSPACE_ROOT / "data/gsm8k/test.parquet",
        help="Evaluation parquet file containing prompt, data_source, and reward_model columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for merged models, generations, and metrics. Defaults to <checkpoint_experiment_dir>/passk_eval for checkpoint inputs.",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="1,2,4,8",
        help="Comma-separated K values to report. The script generates max(K) samples once.",
    )
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "sglang", "hf", "trtllm"])
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size for generation.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs visible to the generation job.")
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids to expose to merge/generation, for example: 0 or 4,5.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. Must be > 0 for K > 1.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top_p.")
    parser.add_argument("--max-response-length", type=int, default=2048, help="Maximum generated tokens per response.")
    parser.add_argument("--max-prompt-length", type=int, default=1024, help="Maximum prompt tokens.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum vLLM model context length. Defaults to max_prompt_length + max_response_length.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable CUDA graph capture for faster startup. Use --no-enforce-eager for repeated high-throughput runs.",
    )
    parser.add_argument("--prompt-key", type=str, default="prompt")
    parser.add_argument("--response-key", type=str, default="responses")
    parser.add_argument("--data-source-key", type=str, default="data_source")
    parser.add_argument("--reward-model-key", type=str, default="reward_model")
    parser.add_argument("--name", type=str, default=None, help="Run name used for output paths.")
    parser.add_argument("--skip-merge", action="store_true", help="Treat checkpoint_path as a merged HF model directory.")
    parser.add_argument("--skip-generate", action="store_true", help="Reuse an existing generated parquet.")
    parser.add_argument("--generated-path", type=Path, default=None, help="Existing/generated parquet path.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate or overwrite existing outputs.")
    parser.add_argument("--num-cpus", type=int, default=None, help="Ray num_cpus override for generation.")
    parser.add_argument("--extra-generate-arg", action="append", default=[], help="Extra Hydra override for generation.")
    return parser.parse_args()


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    start = time.perf_counter()
    try:
        subprocess.run(command, cwd=WORKSPACE_ROOT, env=env, check=True)
    finally:
        elapsed = time.perf_counter() - start
        print(f"Command finished in {format_duration(elapsed)}", flush=True)


def has_hf_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def resolve_actor_dir(path: Path) -> Path | None:
    path = path.resolve()
    if (path / "fsdp_config.json").is_file():
        return path
    if (path / "actor" / "fsdp_config.json").is_file():
        return path / "actor"
    return None


def parse_k_values(raw_k: str) -> list[int]:
    k_values = sorted({int(item.strip()) for item in raw_k.split(",") if item.strip()})
    if not k_values or k_values[0] < 1:
        raise ValueError("--k must contain positive integers, for example: 1,2,4,8")
    return k_values


def safe_name(path: Path) -> str:
    parts = [part for part in path.resolve().parts[-4:] if part not in {"actor"}]
    return "__".join(parts).replace("/", "_")


def infer_default_output_dir(checkpoint_path: Path) -> Path:
    path = checkpoint_path.resolve()
    parts = path.parts
    for index, part in enumerate(parts):
        if part.startswith("global_step_"):
            return Path(*parts[:index]) / "passk_eval"
    if path.name == "actor" and path.parent.name.startswith("global_step_"):
        return path.parent.parent / "passk_eval"
    return WORKSPACE_ROOT / "outputs/passk_eval"


def prepare_model(args: argparse.Namespace, run_name: str, env: dict[str, str]) -> Path:
    checkpoint_path = args.checkpoint_path.resolve()
    merged_dir = args.output_dir.resolve() / "merged_hf" / run_name

    if args.skip_merge or has_hf_weights(checkpoint_path):
        print(f"Using HuggingFace model: {checkpoint_path}")
        return checkpoint_path

    actor_dir = resolve_actor_dir(checkpoint_path)
    if actor_dir is None:
        raise FileNotFoundError(
            "checkpoint_path is neither a merged HF model nor a verl FSDP actor checkpoint. "
            "Pass .../global_step_xxx/actor, .../global_step_xxx, or use --skip-merge for HF models."
        )

    if has_hf_weights(merged_dir) and not args.overwrite:
        print(f"Reusing merged HuggingFace model: {merged_dir}")
        return merged_dir

    merged_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
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
        ],
        env,
    )
    return merged_dir


def generate_responses(
    args: argparse.Namespace, model_path: Path, generated_path: Path, max_k: int, env: dict[str, str]
) -> None:
    if generated_path.is_file() and not args.overwrite:
        print(f"Reusing generated parquet: {generated_path}")
        return

    if max_k > 1 and args.temperature <= 0:
        raise ValueError("Generation with K > 1 requires --temperature > 0.")

    generated_path.parent.mkdir(parents=True, exist_ok=True)
    max_model_len = args.max_model_len or (args.max_prompt_length + args.max_response_length)

    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_generation_server",
        f"data.train_files=['{args.data_file.resolve()}']",
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
    run_command(command, env)


def score_response(data_source: str, response: str, reward_data: Any) -> float:
    from verl.utils.reward_score import default_compute_score

    ground_truth = reward_data["ground_truth"] if isinstance(reward_data, dict) else reward_data.get("ground_truth")
    result = default_compute_score(data_source=data_source, solution_str=response, ground_truth=ground_truth)
    if isinstance(result, dict):
        for key in ("score", "acc", "reward"):
            if key in result:
                return float(result[key])
        raise ValueError(f"Reward function returned a dict without score/acc/reward: {result}")
    return float(result)


def compute_passk(args: argparse.Namespace, generated_path: Path, k_values: list[int], metrics_path: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    print(f"Reading generated parquet: {generated_path}", flush=True)
    dataset = pd.read_parquet(generated_path)
    required_columns = [args.response_key, args.data_source_key, args.reward_model_key]
    missing = [column for column in required_columns if column not in dataset.columns]
    if missing:
        raise KeyError(f"Generated parquet is missing required columns: {missing}")

    max_k = max(k_values)
    grouped_scores: dict[str, list[list[float]]] = {}

    total = len(dataset)
    print(f"Scoring {total} prompts with max K={max_k} ({total * max_k} responses)", flush=True)
    for row_index, row in progress_iter(dataset.iterrows(), total=total, desc="Scoring responses"):
        responses = row[args.response_key]
        if isinstance(responses, np.ndarray):
            responses = responses.tolist()
        if not isinstance(responses, list):
            responses = list(responses)
        if len(responses) < max_k:
            raise ValueError(f"Row {row_index} has {len(responses)} responses, but max K is {max_k}.")

        data_source = row[args.data_source_key]
        reward_data = row[args.reward_model_key]
        scores = [score_response(data_source, response, reward_data) for response in responses[:max_k]]
        grouped_scores.setdefault(data_source, []).append(scores)

    metrics: dict[str, Any] = {"num_prompts": int(len(dataset)), "k_values": k_values, "data_sources": {}}
    overall_by_k: dict[int, list[float]] = {k: [] for k in k_values}

    for data_source, score_rows in grouped_scores.items():
        score_array = np.asarray(score_rows, dtype=np.float32)
        source_metrics: dict[str, float | int] = {"num_prompts": int(score_array.shape[0])}
        for k in k_values:
            pass_values = np.max(score_array[:, :k], axis=1)
            mean_values = np.mean(score_array[:, :k], axis=1)
            source_metrics[f"pass@{k}"] = float(np.mean(pass_values))
            source_metrics[f"mean@{k}"] = float(np.mean(mean_values))
            overall_by_k[k].extend(pass_values.tolist())
        metrics["data_sources"][data_source] = source_metrics

    metrics["overall"] = {f"pass@{k}": float(np.mean(overall_by_k[k])) for k in k_values}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote metrics JSON: {metrics_path}", flush=True)
    return metrics


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    k_values = parse_k_values(args.k)
    max_k = max(k_values)
    run_name = args.name or safe_name(args.checkpoint_path)
    args.output_dir = (args.output_dir or infer_default_output_dir(args.checkpoint_path)).resolve()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(VERL_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        visible_gpu_count = len([item for item in args.cuda_visible_devices.split(",") if item.strip()])
        if visible_gpu_count > 0 and args.gpus > visible_gpu_count:
            raise ValueError(
                f"--gpus={args.gpus} exceeds the number of GPUs in --cuda-visible-devices={args.cuda_visible_devices}."
            )
        print(f"Using CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    print(f"Run name: {run_name}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"K values: {k_values}", flush=True)

    with timed_stage("model preparation"):
        model_path = prepare_model(args, run_name, env)
    generated_path = args.generated_path or args.output_dir / "generations" / f"{run_name}__k{max_k}.parquet"

    with timed_stage("response generation"):
        if not args.skip_generate:
            generate_responses(args, model_path, generated_path.resolve(), max_k, env)
        elif not generated_path.is_file():
            raise FileNotFoundError(f"--skip-generate was set, but generated parquet does not exist: {generated_path}")
        else:
            print(f"Reusing generated parquet: {generated_path}", flush=True)

    metrics_path = args.output_dir / "metrics" / f"{run_name}__k{max_k}_passk.json"
    with timed_stage("pass@K scoring"):
        metrics = compute_passk(args, generated_path.resolve(), k_values, metrics_path)

    print("\nPass@K results")
    for key, value in metrics["overall"].items():
        print(f"{key}: {value:.6f}")
    for data_source, source_metrics in metrics["data_sources"].items():
        print(f"\n{data_source} ({source_metrics['num_prompts']} prompts)")
        for k in k_values:
            print(f"pass@{k}: {source_metrics[f'pass@{k}']:.6f}  mean@{k}: {source_metrics[f'mean@{k}']:.6f}")
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Generated parquet: {generated_path}")
    print(f"Model path: {model_path}")
    print(f"Total runtime: {format_duration(time.perf_counter() - total_start)}")


if __name__ == "__main__":
    main()