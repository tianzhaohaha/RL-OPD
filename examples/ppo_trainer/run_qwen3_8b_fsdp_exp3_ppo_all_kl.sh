#!/usr/bin/env bash
# PPO ablation | FSDP | clipped policy objective | reward KL + actor KL loss.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_exp3_ppo_all_kl_fsdp_$(date +%Y%m%d_%H%M)}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-2}
PPO_CLIP_RATIO=${PPO_CLIP_RATIO:-0.2}
REWARD_KL_COEF=${REWARD_KL_COEF:-0.001}
ACTOR_KL_COEF=${ACTOR_KL_COEF:-0.001}
REWARD_KL_TYPE=${REWARD_KL_TYPE:-kl}
ACTOR_KL_TYPE=${ACTOR_KL_TYPE:-low_var_kl}

export EXPERIMENT_NAME CUDA_VISIBLE_DEVICES NDEVICES_PER_NODE

exec "${SCRIPT_DIR}/run_qwen3_8b_fsdp.sh" \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.clip_ratio=${PPO_CLIP_RATIO} \
    actor_rollout_ref.actor.clip_ratio_low=${PPO_CLIP_RATIO} \
    actor_rollout_ref.actor.clip_ratio_high=${PPO_CLIP_RATIO} \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=${REWARD_KL_TYPE} \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=${REWARD_KL_COEF} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${ACTOR_KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type=${ACTOR_KL_TYPE} \
    "$@"