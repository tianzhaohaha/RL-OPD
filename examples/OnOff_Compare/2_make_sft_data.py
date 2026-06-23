# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Step 2 | Convert teacher generations into SFT data (messages format).

Reads the parquet produced by ``1_teacher_generate.sh`` (original prompt columns
plus a ``responses`` column = list of N teacher samples) and emits a parquet with
a single ``messages`` column [user, assistant] that ``verl.trainer.sft_trainer``
consumes directly (``data.messages_key=messages``).

This is the OFF-POLICY supervised target used by experiment A: the student only
ever sees states from the teacher's trajectory distribution.

Optionally filter to keep only correct teacher samples (rejection-sampling SFT),
scored with verl's task reward so the SFT target matches the RL reward signal.

Usage:
    # both datasets, keep only correct teacher samples
    python3 2_make_sft_data.py \
        --inputs ~/data/gsm8k/teacher_gen_train.parquet ~/data/math/teacher_gen_train.parquet \
        --output ~/data/onoff_sft/train.parquet \
        --filter-correct

    # keep everything (pure distillation-by-SFT, no correctness filter)
    python3 2_make_sft_data.py \
        --inputs ~/data/gsm8k/teacher_gen_train.parquet \
        --output ~/data/onoff_sft/train.parquet
"""
import sys
sys.path.insert(0, "/home/jcgu/qyliu/RL-OPD/verl")

import argparse
import os

import pandas as pd


def _to_list(x):
    """parquet may return numpy arrays; normalise to python list."""
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _extract_ground_truth(reward_model):
    if reward_model is None:
        return None
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")
    # numpy/object fallbacks
    try:
        return reward_model["ground_truth"]
    except Exception:
        return None


def build_messages(df: pd.DataFrame, filter_correct: bool) -> list[dict]:
    if filter_correct:
        from verl.utils.reward_score import default_compute_score

    records = []
    kept, total = 0, 0
    for _, row in df.iterrows():
        prompt = _to_list(row["prompt"])
        # prompt entries may themselves be numpy structs; coerce to plain dict
        prompt = [dict(m) if not isinstance(m, dict) else m for m in prompt]
        responses = _to_list(row["responses"])
        data_source = row.get("data_source")
        ground_truth = _extract_ground_truth(row.get("reward_model"))

        for resp in responses:
            total += 1
            if filter_correct:
                try:
                    score = default_compute_score(data_source, resp, ground_truth)
                    score = score["score"] if isinstance(score, dict) else float(score)
                except Exception:
                    score = 0.0
                if score <= 0:
                    continue
            kept += 1
            messages = list(prompt) + [{"role": "assistant", "content": resp}]
            records.append({"messages": messages})

    print(f"Kept {kept}/{total} teacher samples (filter_correct={filter_correct}).")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="teacher_gen_*.parquet files")
    parser.add_argument("--output", required=True, help="output SFT parquet path")
    parser.add_argument(
        "--filter-correct",
        action="store_true",
        help="keep only teacher samples that solve the task (rejection-sampling SFT)",
    )
    args = parser.parse_args()

    frames = [pd.read_parquet(os.path.expanduser(p)) for p in args.inputs]
    df = pd.concat(frames, axis=0, ignore_index=True)

    records = build_messages(df, args.filter_correct)
    out_df = pd.DataFrame(records)

    out_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_parquet(out_path)
    print(f"Wrote {len(out_df)} SFT rows to {out_path}")


if __name__ == "__main__":
    main()
