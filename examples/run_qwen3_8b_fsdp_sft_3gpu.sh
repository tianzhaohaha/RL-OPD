#!/usr/bin/env bash
# Run Qwen3-4B/32B GSM8K OPD and SFT experiments sequentially.
# Default GPU set: 4,5,6,7.
# OPD has separate student and teacher resource pools, so it defaults to 2+2 GPUs.

set -xeuo pipefail

########################### paths ###########################
EXAMPLES_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${EXAMPLES_DIR}/../.." && pwd)
VERL_ROOT=${VERL_ROOT:-${REPO_ROOT}/verl}

GRPO_SCRIPT=${GRPO_SCRIPT:-${EXAMPLES_DIR}/grpo_trainer/run_qwen3_8b_fsdp.sh}
OPD_SCRIPT=${OPD_SCRIPT:-${EXAMPLES_DIR}/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh}
SFT_SCRIPT=${SFT_SCRIPT:-${EXAMPLES_DIR}/sft/gsm8k/run_qwen3_8b_fsdp.sh}

########################### shared defaults ###########################
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M)}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
N_GPUS=${N_GPUS:-1}
NNODES=${NNODES:-1}
DEVICE=${DEVICE:-gpu}

DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}}
MODEL_PATH=${MODEL_PATH:-/home/jcgu/qyliu/RL-OPD/OPD/model/Qwen3-1.7B}
STUDENT_MODEL=${STUDENT_MODEL:-${MODEL_PATH}}
TEACHER_MODEL=${TEACHER_MODEL:-/home/jcgu/qyliu/RL-OPD/OPD/model/Qwen3-32B}

OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/qwen3_1.7b_from_32b_gsm8k_opd_sft_${RUN_ID}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${VERL_ROOT}/checkpoints}

########################### experiment names ###########################
GRPO_PROJECT_NAME=${GRPO_PROJECT_NAME:-verl_grpo_gsm8k}
OPD_PROJECT_NAME=${OPD_PROJECT_NAME:-verl_distill_gsm8k}
SFT_PROJECT_NAME=${SFT_PROJECT_NAME:-verl_sft_gsm8k}

GRPO_EXPERIMENT_NAME=${GRPO_EXPERIMENT_NAME:-qwen3_1.7b_grpo_gsm8k_${N_GPUS}gpu_fsdp_${RUN_ID}}
OPD_EXPERIMENT_NAME=${OPD_EXPERIMENT_NAME:-qwen3_1.7b_from_32b_opd_gsm8k_s${OPD_NGPUS_PER_NODE:-2}_t${OPD_TEACHER_WORLD_SIZE:-2}_fsdp_${RUN_ID}}
SFT_EXPERIMENT_NAME=${SFT_EXPERIMENT_NAME:-qwen3_1.7b_sft_gsm8k_${N_GPUS}gpu_fsdp_${RUN_ID}}
SFT_SAVE_PATH=${SFT_SAVE_PATH:-${CHECKPOINT_ROOT}/${SFT_PROJECT_NAME}/${SFT_EXPERIMENT_NAME}}

########################### 4-GPU rollout/resource split (2+2) ###########################
GRPO_ROLLOUT_TP=${GRPO_ROLLOUT_TP:-1}
OPD_NGPUS_PER_NODE=${OPD_NGPUS_PER_NODE:-1}
OPD_TEACHER_WORLD_SIZE=${OPD_TEACHER_WORLD_SIZE:-2}
OPD_ROLLOUT_TP=${OPD_ROLLOUT_TP:-1}
OPD_TEACHER_TP=${OPD_TEACHER_TP:-2}

########################### memory-safe defaults ###########################
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}

ROLLOUT_N=${ROLLOUT_N:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.50}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}

SP_SIZE=${SP_SIZE:-1}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}

export CUDA_VISIBLE_DEVICES DATA_ROOT MODEL_PATH STUDENT_MODEL TEACHER_MODEL
export DEVICE NNODES
export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH PPO_MAX_TOKEN_LEN_PER_GPU
export ROLLOUT_N ROLLOUT_GPU_MEM_UTIL TEACHER_GPU_MEM_UTIL DISTILLATION_TOPK
export SP_SIZE MICRO_BATCH_SIZE_PER_GPU
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}"
cd "${VERL_ROOT}"

run_grpo() {
	PROJECT_NAME=${GRPO_PROJECT_NAME} \
	EXPERIMENT_NAME=${GRPO_EXPERIMENT_NAME} \
	NGPUS_PER_NODE=${N_GPUS} \
	ROLLOUT_TP=${GRPO_ROLLOUT_TP} \
	"${GRPO_SCRIPT}" "$@"
}

run_opd() {
	PROJECT_NAME=${OPD_PROJECT_NAME} \
	EXPERIMENT_NAME=${OPD_EXPERIMENT_NAME} \
	NGPUS_PER_NODE=${OPD_NGPUS_PER_NODE} \
	TEACHER_WORLD_SIZE=${OPD_TEACHER_WORLD_SIZE} \
	ROLLOUT_TP=${OPD_ROLLOUT_TP} \
	TEACHER_TP=${OPD_TEACHER_TP} \
	"${OPD_SCRIPT}" "$@"
}

run_sft() {
	PROJECT_NAME=${SFT_PROJECT_NAME} \
	EXPERIMENT_NAME=${SFT_EXPERIMENT_NAME} \
	"${SFT_SCRIPT}" "${N_GPUS}" "${SFT_SAVE_PATH}" "$@"
}

########################### sequential launch ###########################
run_sft "$@"
# run_opd "$@"
# run_grpo "$@"
