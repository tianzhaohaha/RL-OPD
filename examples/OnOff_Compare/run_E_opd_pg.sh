#!/usr/bin/env bash
# Experiment E | ON-POLICY DISTILLATION (Thinking-Machines / policy-gradient form).
#
# State distribution: student's own rollouts (on-policy).
# Signal: per-token reverse-KL used as a dense REWARD inside a policy-gradient
#         update (use_policy_gradient=True, loss_mode=k1).
#         See https://thinkingmachines.ai/blog/on-policy-distillation/.
#
# Same on-policy states as D, but the dense teacher signal flows through the RL
# (policy-gradient) objective instead of direct supervised backprop. Comparing
# D vs E isolates "supervised dense loss" vs "dense reward via policy gradient".
#
# Prereqs: bash 0_prepare_data.sh
#
# Usage:
#   bash run_E_opd_pg.sh
#   STUDENT_MODEL=Qwen/Qwen3-8B TEACHER_MODEL=Qwen/Qwen3-32B bash run_E_opd_pg.sh

set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${REPO_ROOT}"
# wandb: default to offline to avoid SSL/network retry loops on isolated nodes.
# Set WANDB_MODE=online (with a reachable network) to stream live.
export WANDB_MODE=${WANDB_MODE:-offline}
# ---- shared comparison hyperparameters (identical to D) ----
export STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-4B-Base}
export TEACHER_MODEL=${TEACHER_MODEL:-/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_grpo_gsm8k/qwen3_4b_grpo_gsm8k_2gpu_fsdp_20260612_2342/global_step_200/actor/huggingface_merged}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ACTOR_LR=${ACTOR_LR:-1e-6}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-200}

# ---- experiment-E specific: KL-as-reward policy gradient (Thinking Machines) ----
export DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
export USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}

export PROJECT_NAME=${PROJECT_NAME:-onoff_compare}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-E_opd_pg_k1}

# use_task_rewards=False => pure distillation reward (no task reward mixed in).
bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh \
    distillation.distillation_loss.use_task_rewards=False \
    "$@"
