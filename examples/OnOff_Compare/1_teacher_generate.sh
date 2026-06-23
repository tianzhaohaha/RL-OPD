#!/usr/bin/env bash
# Step 1 | Teacher generates trajectories on the SAME prompts (for experiment A: off-policy SFT).
#
# The teacher (default: the 4B GRPO checkpoint) rolls out responses on the GSM8K
# train prompts. Output parquet keeps every original column (prompt,
# reward_model, data_source, extra_info) and adds a `responses` column
# (list of N strings).
#
# Run this ONCE per dataset, then build SFT data with 2_make_sft_data.py.
#
# Usage:
#   bash 1_teacher_generate.sh                # both gsm8k + math, default teacher
#   TEACHER_MODEL=Qwen/Qwen3-32B N_SAMPLES=1 bash 1_teacher_generate.sh
ray stop

set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

# ---- user-adjustable ----
TEACHER_MODEL=${TEACHER_MODEL:-/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_grpo_gsm8k/qwen3_4b_grpo_gsm8k_2gpu_fsdp_20260612_2342/global_step_200/actor/huggingface_merged}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

PROMPT_LENGTH=${PROMPT_LENGTH:-1024}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-2048}
# Must satisfy: (NNODES * NGPUS_PER_NODE) % ROLLOUT_TP == 0, and TP <= total GPUs.
# A 4B teacher fits on 1 GPU; TP=1 with 2 GPUs => 2 parallel replicas (faster).
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}
N_SAMPLES=${N_SAMPLES:-1}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}

# Which datasets to generate over (space separated keys: gsm8k math)
DATASETS=${DATASETS:-"gsm8k"}
# ---- end user-adjustable ----

for ds in ${DATASETS}; do
    DATA_PATH="$HOME/qyliu/RL-OPD/data/${ds}/train.parquet"
    OUTPUT_PATH="$HOME/qyliu/RL-OPD/data/${ds}/teacher_gen_train.parquet"

    python3 -m verl.trainer.main_generation_server \
        trainer.nnodes="${NNODES}" \
        trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
        data.train_files="${DATA_PATH}" \
        data.prompt_key=prompt \
        +data.output_path="${OUTPUT_PATH}" \
        actor_rollout_ref.model.path="${TEACHER_MODEL}" \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
        actor_rollout_ref.rollout.top_p="${TOP_P}" \
        actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
        actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
        actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
        actor_rollout_ref.rollout.n="${N_SAMPLES}" "$@"
done

echo "Teacher generations written to \$HOME/data/<ds>/teacher_gen_train.parquet"
