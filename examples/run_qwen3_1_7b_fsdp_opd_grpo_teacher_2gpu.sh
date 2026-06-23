#!/usr/bin/env bash
# Run Qwen3-1.7B GSM8K OPD with a GRPO-trained checkpoint as the teacher.
# If the GRPO checkpoint is still saved as FSDP shards, this script merges it
# into a HuggingFace model first and then uses that merged directory as teacher.

set -xeuo pipefail

########################### paths ###########################
EXAMPLES_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${EXAMPLES_DIR}/../.." && pwd)
VERL_ROOT=${VERL_ROOT:-${REPO_ROOT}/verl}

OPD_SCRIPT=${OPD_SCRIPT:-${EXAMPLES_DIR}/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh}

########################### shared defaults ###########################
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M)}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
NNODES=${NNODES:-1}
DEVICE=${DEVICE:-gpu}

DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}}

# Student starts from the base/SFT model you want to train with OPD.
STUDENT_MODEL=${STUDENT_MODEL:-/home/jcgu/qyliu/RL-OPD/OPD/model/Qwen3-1.7B}
MODEL_PATH=${MODEL_PATH:-${STUDENT_MODEL}}

# Teacher defaults to the latest visible GRPO checkpoint in this workspace.
# For another GRPO checkpoint, set GRPO_TEACHER_ACTOR_DIR to .../global_step_xxx/actor.
GRPO_TEACHER_ACTOR_DIR=${GRPO_TEACHER_ACTOR_DIR:-${REPO_ROOT}/verl/checkpoints/verl_grpo_gsm8k/qwen3_1_7b_grpo_gsm8k_2gpu_fsdp_20260610_0018/global_step_300/actor}
MERGED_TEACHER_MODEL=${MERGED_TEACHER_MODEL:-${GRPO_TEACHER_ACTOR_DIR}/huggingface_merged}
TEACHER_MODEL=${TEACHER_MODEL:-${MERGED_TEACHER_MODEL}}
AUTO_MERGE_TEACHER=${AUTO_MERGE_TEACHER:-True}

OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/qwen3_1_7b_opd_from_grpo_teacher_gsm8k_2gpu_${RUN_ID}}

########################### experiment names ###########################
OPD_PROJECT_NAME=${OPD_PROJECT_NAME:-verl_distill_gsm8k}
OPD_EXPERIMENT_NAME=${OPD_EXPERIMENT_NAME:-qwen3_1_7b_opd_from_grpo_teacher_gsm8k_2gpu_s${OPD_NGPUS_PER_NODE:-1}_t${OPD_TEACHER_WORLD_SIZE:-1}_fsdp_${RUN_ID}}

########################### 2-GPU OPD resource split (1 student + 1 teacher) ###########################
OPD_NGPUS_PER_NODE=${OPD_NGPUS_PER_NODE:-1}
OPD_TEACHER_WORLD_SIZE=${OPD_TEACHER_WORLD_SIZE:-1}
OPD_ROLLOUT_TP=${OPD_ROLLOUT_TP:-1}
OPD_TEACHER_TP=${OPD_TEACHER_TP:-1}

########################### memory-safe defaults ###########################
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.4}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}

SP_SIZE=${SP_SIZE:-1}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}

export CUDA_VISIBLE_DEVICES DATA_ROOT MODEL_PATH STUDENT_MODEL TEACHER_MODEL
export DEVICE NNODES
export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH PPO_MAX_TOKEN_LEN_PER_GPU
export ROLLOUT_GPU_MEM_UTIL TEACHER_GPU_MEM_UTIL DISTILLATION_TOPK
export SP_SIZE MICRO_BATCH_SIZE_PER_GPU
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}"
cd "${VERL_ROOT}"

has_hf_weights() {
    [[ -d "$1" ]] && [[ -n "$(find "$1" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print -quit)" ]]
}

case "${AUTO_MERGE_TEACHER}" in
    True | true | 1 | yes | YES)
        if ! has_hf_weights "${TEACHER_MODEL}"; then
            if [[ ! -d "${GRPO_TEACHER_ACTOR_DIR}" ]]; then
                echo "GRPO_TEACHER_ACTOR_DIR does not exist: ${GRPO_TEACHER_ACTOR_DIR}" >&2
                exit 1
            fi
            if [[ ! -f "${GRPO_TEACHER_ACTOR_DIR}/fsdp_config.json" ]]; then
                echo "Missing FSDP config: ${GRPO_TEACHER_ACTOR_DIR}/fsdp_config.json" >&2
                echo "Set TEACHER_MODEL to an already-merged HuggingFace model, or set GRPO_TEACHER_ACTOR_DIR to .../global_step_xxx/actor." >&2
                exit 1
            fi

            python3 -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "${GRPO_TEACHER_ACTOR_DIR}" \
                --target_dir "${TEACHER_MODEL}" \
                --use_cpu_initialization
        fi
        ;;
esac

if ! has_hf_weights "${TEACHER_MODEL}"; then
    echo "TEACHER_MODEL is not a complete HuggingFace model directory: ${TEACHER_MODEL}" >&2
    echo "It must contain model weights such as *.safetensors or *.bin." >&2
    echo "For FSDP GRPO checkpoints, set GRPO_TEACHER_ACTOR_DIR to .../global_step_xxx/actor and keep AUTO_MERGE_TEACHER=True." >&2
    exit 1
fi

PROJECT_NAME=${OPD_PROJECT_NAME} \
EXPERIMENT_NAME=${OPD_EXPERIMENT_NAME} \
NGPUS_PER_NODE=${OPD_NGPUS_PER_NODE} \
TEACHER_WORLD_SIZE=${OPD_TEACHER_WORLD_SIZE} \
ROLLOUT_TP=${OPD_ROLLOUT_TP} \
TEACHER_TP=${OPD_TEACHER_TP} \
"${OPD_SCRIPT}" "$@"