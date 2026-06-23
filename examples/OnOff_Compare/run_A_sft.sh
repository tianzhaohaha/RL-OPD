#!/usr/bin/env bash
# Experiment A | OFF-POLICY SFT on teacher-generated trajectories.
#
# State distribution: teacher's (off-policy).  Signal: token-level CE (dense).
# This is the off-policy + dense-supervision baseline of the comparison.
#
# Prereqs:
#   bash 0_prepare_data.sh
#   bash 1_teacher_generate.sh
#   python3 2_make_sft_data.py --inputs ~/data/gsm8k/teacher_gen_train.parquet \
#       ~/data/math/teacher_gen_train.parquet --output ~/data/onoff_sft/train.parquet [--filter-correct]
#
# Usage:
#   bash run_A_sft.sh                # auto save_path under <verl>/checkpoints/...
#   bash run_A_sft.sh 2              # set nproc, auto save_path
#   bash run_A_sft.sh 2 /path/ckpt  # set nproc and save_path explicitly

set -xeuo pipefail

nproc_per_node=${1:-2}
shift || true

EXAMPLES_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${EXAMPLES_DIR}/../.." && pwd)
VERL_ROOT=${VERL_ROOT:-${REPO_ROOT}}
cd "${VERL_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

# ---- user-adjustable (keep aligned with D/E for a fair comparison) ----
STUDENT_MODEL=${STUDENT_MODEL:-/home/jcgu/qyliu/RL-OPD/OPD/model/Qwen3-4B}

TRAIN_FILE=${TRAIN_FILE:-$HOME/qyliu/RL-OPD/data/onoff_sft/train.parquet}
# Validation uses the held-out GSM8K test prompts+answers in messages form.
# Build it once with gsm8k_multiturn_sft.py if you want SFT-style val loss; the
# real comparison metric is downstream task accuracy (see 9_eval.sh below).
VAL_FILE=${VAL_FILE:-$HOME/qyliu/RL-OPD/data/gsm8k_sft/test.parquet}

MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4}
LR=${LR:-1e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-30}
# Teacher gen = PROMPT(1024)+RESPONSE(2048)=3072; add headroom for chat-template
# special tokens so truncation=error does not abort on borderline samples.
MAX_LENGTH=${MAX_LENGTH:-3584}
SP_SIZE=${SP_SIZE:-1}
# Save + eval periodically (in steps) so we can plot accuracy vs. epoch and
# catch off-policy SFT overfitting. -1 disables.
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-50}
# Bound disk usage: keep only the most recent N checkpoints (null = keep all).
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-5}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M)}
N_GPUS=${N_GPUS:-${nproc_per_node}}
PROJECT_NAME=${PROJECT_NAME:-verl_sft_gsm8k_teacher_gen}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_sft_gsm8k_${N_GPUS}gpu_fsdp_${RUN_ID}}

# wandb: default to offline to avoid SSL/network retry loops on isolated nodes.
# Set WANDB_MODE=online (and a reachable network) to stream live; LOGGER controls
# which backends are enabled.
export WANDB_MODE=${WANDB_MODE:-offline}
LOGGER=${LOGGER:-'["console","wandb"]'}

# ckpt path follows: <verl>/checkpoints/<PROJECT_NAME>/<EXPERIMENT_NAME>
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${VERL_ROOT}/checkpoints}
SFT_SAVE_PATH=${SFT_SAVE_PATH:-${CHECKPOINT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}}
save_path=${1:-${SFT_SAVE_PATH}}
shift || true
mkdir -p "${save_path}"
# ---- end user-adjustable ----

torchrun --standalone --nnodes=1 --nproc_per_node="${nproc_per_node}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.messages_key=messages \
    data.max_length="${MAX_LENGTH}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.truncation=error \
    data.ignore_input_ids_mismatch=True \
    optim.lr="${LR}" \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size="${SP_SIZE}" \
    model.path="${STUDENT_MODEL}" \
    model.use_remove_padding=true \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="${LOGGER}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.max_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
    "$@"
