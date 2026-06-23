#!/usr/bin/env bash
# SFT | Qwen3-1.7B | GSM8K | FSDP engine
# Toggle Ulysses sequence parallel and LoRA/PEFT via env vars.
#
# Examples:
#   # plain SFT
#   bash run_qwen3_8b_fsdp.sh 8 /tmp/sft-ckpt
#
#   # sequence-parallel (Ulysses) = 2 + LoRA (the default demo)
#   SP_SIZE=2 USE_PEFT=1 bash run_qwen3_8b_fsdp.sh 8 /tmp/sft-ckpt

set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_8b_fsdp.sh <nproc_per_node> <save_path> [other_configs...]"
    echo "  Env: SP_SIZE (default 2), USE_PEFT (0|1, default 1)"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-/home/jcgu/qyliu/RL-OPD/OPD/model/Qwen3-1.7B}
SP_SIZE=${SP_SIZE:-2}
USE_PEFT=${USE_PEFT:-0}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGETS=${LORA_TARGETS:-all-linear}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-64}
LR=${LR:-1e-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-35}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-5}
PROJECT_NAME=${PROJECT_NAME:-gsm8k-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gsm8k-sft-qwen3-1-7b}
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}}
SFT_DATA_DIR=${SFT_DATA_DIR:-${DATA_ROOT}/data/gsm8k_sft}
SFT_TRAIN_FILE=${SFT_TRAIN_FILE:-${SFT_DATA_DIR}/train.parquet}
SFT_VAL_FILE=${SFT_VAL_FILE:-${SFT_DATA_DIR}/test.parquet}
# ---- end user-adjustable ----

extra_args=()
if [ "${USE_PEFT}" = "1" ]; then
    extra_args+=(
        "model.lora_rank=${LORA_RANK}"
        "model.lora_alpha=${LORA_ALPHA}"
        "model.target_modules=${LORA_TARGETS}"
    )
fi

torchrun --standalone --nnodes=1 --nproc_per_node=${nproc_per_node} \
    -m verl.trainer.sft_trainer \
    data.train_files=${SFT_TRAIN_FILE} \
    data.val_files=${SFT_VAL_FILE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    optim.lr=${LR} \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    "${extra_args[@]}" "$@"
