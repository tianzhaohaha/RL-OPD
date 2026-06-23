#!/usr/bin/env bash
# PPO baseline | FSDP | clipped policy objective | no KL constraints.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_exp2_ppo_no_kl_fsdp_$(date +%Y%m%d_%H%M)}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-2}
PPO_CLIP_RATIO=${PPO_CLIP_RATIO:-0.2}

export EXPERIMENT_NAME CUDA_VISIBLE_DEVICES NDEVICES_PER_NODE

exec "${SCRIPT_DIR}/run_qwen3_8b_fsdp.sh" \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.clip_ratio=${PPO_CLIP_RATIO} \
    actor_rollout_ref.actor.clip_ratio_low=${PPO_CLIP_RATIO} \
    actor_rollout_ref.actor.clip_ratio_high=${PPO_CLIP_RATIO} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    "$@"