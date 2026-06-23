#!/usr/bin/env bash
# PPO ablation | FSDP | unclipped policy objective | no KL constraints.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_exp1_unclipped_no_kl_fsdp_$(date +%Y%m%d_%H%M)}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-2}
UNCLIPPED_RATIO=${UNCLIPPED_RATIO:-1000000.0}

export EXPERIMENT_NAME CUDA_VISIBLE_DEVICES NDEVICES_PER_NODE

exec "${SCRIPT_DIR}/run_qwen3_8b_fsdp.sh" \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.clip_ratio=${UNCLIPPED_RATIO} \
    actor_rollout_ref.actor.clip_ratio_low=${UNCLIPPED_RATIO} \
    actor_rollout_ref.actor.clip_ratio_high=${UNCLIPPED_RATIO} \
    actor_rollout_ref.actor.clip_ratio_c=${UNCLIPPED_RATIO} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    "$@"