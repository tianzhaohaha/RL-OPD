#!/usr/bin/env bash
# Experiment D | ON-POLICY DISTILLATION (GKD-style, dense KL backprop).
#
# State distribution: student's own rollouts (on-policy).
# Signal: teacher per-token distribution, backpropagated directly as a
#         supervised KL loss (use_policy_gradient=False, loss_mode=k3).
#         See https://arxiv.org/abs/2306.13649 (GKD).
#
# This isolates "dense teacher signal on ON-POLICY states" vs:
#   - A (same dense teacher signal, OFF-policy states)
#   - C/GRPO (same on-policy states, sparse reward)
#
# Prereqs: bash 0_prepare_data.sh
#
# Usage:
#   bash run_D_opd_gkd.sh
#   STUDENT_MODEL=Qwen/Qwen3-4B TEACHER_MODEL=/path/to/teacher bash run_D_opd_gkd.sh

set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
# wandb: default to offline to avoid SSL/network retry loops on isolated nodes.
# Set WANDB_MODE=online (with a reachable network) to stream live.
export WANDB_MODE=${WANDB_MODE:-offline}

# ---- OPD resource split on 2 GPUs: student pool (1) + teacher pool (1) ----
# Teacher is a SEPARATE vLLM inference cluster, so it cannot share the student
# GPU. Student GPU runs FSDP training + vLLM rollout colocated, hence a low
# rollout mem-util to leave room for training; teacher GPU only does inference.
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
export TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export TEACHER_TP=${TEACHER_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
export TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.8}

# ---- pass@k validation: sample VAL_N responses so main_ppo logs best@k/maj@k ----
VAL_N=${VAL_N:-1}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0}
VAL_TOP_P=${VAL_TOP_P:-0.95}

# ---- shared comparison hyperparameters (keep identical across D/E and GRPO) ----
export STUDENT_MODEL=${STUDENT_MODEL:-/home/jcgu/qyliu/RL-OPD/OPD/model/Qwen3-4B}
export TEACHER_MODEL=${TEACHER_MODEL:-/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_grpo_gsm8k/qwen3_4b_grpo_gsm8k_2gpu_fsdp_20260612_2342/global_step_200/actor/huggingface_merged}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ACTOR_LR=${ACTOR_LR:-1e-6}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-50}
export TEST_FREQ=${TEST_FREQ:-20}
export SAVE_FREQ=${SAVE_FREQ:-100}

# ---- experiment-D specific: dense KL backprop (GKD) ----
export DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k3}
export USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-False}

export PROJECT_NAME=${PROJECT_NAME:-onoff_compare}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-D_opd_gkd_k3}

# use_task_rewards=False => pure distillation (no task reward mixed in), so the
# only difference from A is the state distribution (on-policy vs off-policy).
bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh \
    distillation.distillation_loss.use_task_rewards=False \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    trainer.val_before_train=True \
    "$@"
