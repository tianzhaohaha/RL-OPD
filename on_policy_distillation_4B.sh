#!/bin/bash
#SBATCH --job-name=url
#SBATCH --output=logs/20251004/output_%j.log
#SBATCH --error=logs/20251004/error_%j.log
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --exclude=g[81-82]
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/verl${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN   # 顺手把注意力后端也固定成预编译的 flash-attn,避免再触发别的 JIT

set -x

# Configure logging when running outside SBATCH.
if [ -z "$SLURM_JOB_ID" ]; then
    # Create the log directory and file for local runs.
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    # Mirror output to both terminal and log file.
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

# ray stop --force
export RAY_memory_usage_threshold=0.99
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=5,6,7
export PYTHONUNBUFFERED=1
export PROJECT_NAME='OnPolicyDistillation' # TODO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export ADV_ESTIMATOR=token_reward_direct
# export ADV_ESTIMATOR=token_reward_direct_plus_grpo
# export ADV_ESTIMATOR=token_grpo
# export ADV_ESTIMATOR=grpo
export GRPO_OUTCOME_WEIGHT=1.0
# export ADV_ESTIMATOR=token_grpo
# Swanlab setting used to continue exp  
# export SWANLAB_RESUME=must
# export SWANLAB_RUN_ID="jri5qia6iy67v7su0zjsv"


# DeepMath-103K
export MAX_PROMPT_LENGTH=1024
export MAX_RESP_LENGTH=7168  # TODO: 31744 /15360 / 7168 / 3072 / 5120
export MAX_VAL_RESP_LENGTH=7168 # TODO: 15360 / 7168 / 3072
export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-48} # TODO: 1 / 8 / 16 / 32 / 64 (default 64)
export TEMPERATURE=${TEMPERATURE:-1.0} # TODO: 0.6 / 0.8 / 1.0 / 1.2 (default 1.0)
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0} # Teacher logits temperature (default 1.0, no scaling)
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0} # TODO: 1.0 / 1.1 / 1.2 (default 1.0, no penalty)
export N_RESPONSES=4 # TODO: 4 / 8 / 16 / 32 (default: 8)
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16} # 0 represents no top-k sampling
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-"only_stu"} # "only_stu" or "only_tch" or "intersection" or "union" or "union-intersection"
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-"student_p"} # "student_p" or "teacher_p" or "none"
# export LR=${LR:-1e-6}
# export LR_SCHEDULER=${LR_SCHEDULER:-constant}
export USE_KL=${USE_KL:-False} # TODO: True / False (default False)
export ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False} # TODO: True / False (default False)
export MODEL_DTYPE=${MODEL_DTYPE:-fp32} # actor/ref/critic fsdp_config.model_dtype: fp32 or bfloat16
export IS_PLOT=${IS_PLOT:-True} # TODO: True / False (default False)
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-"token-mean"} # TODO: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean" / "seq-mean-token-sum-norm" (default "token-mean")
export ENABLE_SPARSE_PROJECTION=${ENABLE_SPARSE_PROJECTION:-False} # True / False
export SPARSE_PROJECTION_INTERVAL=${SPARSE_PROJECTION_INTERVAL:-50}
export SPARSE_PROJECTION_ENERGY_RATIO=${SPARSE_PROJECTION_ENERGY_RATIO:-0.5}
export SPARSE_PROJECTION_ALPHA=${SPARSE_PROJECTION_ALPHA:-0.5}
export SPARSE_PROJECTION_MAX_DIM=${SPARSE_PROJECTION_MAX_DIM:-0} # 0 means no limit
export SPARSE_PROJECTION_TARGET_MATRICES=${SPARSE_PROJECTION_TARGET_MATRICES:-"all"} # all / attn / mlp / q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export SPARSE_PROJECTION_RESET_OPTIMIZER=${SPARSE_PROJECTION_RESET_OPTIMIZER:-False}

export ENABLE_ROW_DIRECTION_REGULARIZER=${ENABLE_ROW_DIRECTION_REGULARIZER:-False} # True / False
export ROW_DIRECTION_REGULARIZER_COEF=${ROW_DIRECTION_REGULARIZER_COEF:-1e-2}
export ROW_DIRECTION_REGULARIZER_MODE=${ROW_DIRECTION_REGULARIZER_MODE:-"orthogonal"} # orthogonal / hse
export ROW_DIRECTION_REGULARIZER_TARGET_MATRICES=${ROW_DIRECTION_REGULARIZER_TARGET_MATRICES:-"all"} # all / attn / mlp / q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export ROW_DIRECTION_REGULARIZER_ROW_SAMPLE_SIZE=${ROW_DIRECTION_REGULARIZER_ROW_SAMPLE_SIZE:-512} # 0 means all rows
export ROW_DIRECTION_REGULARIZER_S=${ROW_DIRECTION_REGULARIZER_S:-1.0} # Riesz exponent when mode=hse
export ROW_DIRECTION_REGULARIZER_EPS=${ROW_DIRECTION_REGULARIZER_EPS:-1e-5}

export ENABLE_HSE_MONITOR=${ENABLE_HSE_MONITOR:-True} # True / False
export HSE_MONITOR_INTERVAL=${HSE_MONITOR_INTERVAL:-50}
export HSE_MONITOR_TARGET_MATRICES=${HSE_MONITOR_TARGET_MATRICES:-"all"} # all / attn / mlp / q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj / all_params
export HSE_MONITOR_S=${HSE_MONITOR_S:-1.0}
export HSE_MONITOR_CHUNK_SIZE=${HSE_MONITOR_CHUNK_SIZE:-0} # 0 means full matrix, positive means exact row chunks
export HSE_MONITOR_EPS=${HSE_MONITOR_EPS:-1e-6}
export HSE_MONITOR_LOG_INITIAL=${HSE_MONITOR_LOG_INITIAL:-True}

# TODO: qwen3_1p7b_base / qwen3_1p7b / llama31_8b_base / llama31_8b_inst / qwen3_8b_base / qwen3_8b / qwen25_1p5b_base / qwen25_1p5b_inst / qwen25_7b_base / qwen25_7b_inst / qwen25_math_7b_base / qwen25_math_7b_inst / qwen25_math_1p5b_base / qwen25_math_1p5b_inst / distill_r1_1p5b / olmo2_1124_7b_base / olmo2_1124_7b_sft / olmo2_1124_7b_inst / llama32_3b_inst
# export EXPERIMENT_NAME=grpo_${TASK}_llama31_tulu3_8b_sft_8k-T_${TEMPERATURE}-n_${N_RESPONSES}-kl_${USE_KL}-mbs_${MINI_BATCH_SIZE}-${REWARD_TYPE}-$(date +%Y-%m-%d_%H-%M-%S)

# export TRAIN_DATASET=datasets/DAPO-Math-17k/data/dapo-math-17k-10percent.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/OpenThoughts3_opd.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/sampled_complement_30k.parquet
# export TRAIN_DATASET=datasets/DeepMath-103K/verl_format/train_filtered_sampled.parquet
export TRAIN_DATASET=datasets/dapo-math-17k.parquet
# export TRAIN_DATASET=datasets/Skywork-OR1-RL-Data/data/math-00000-of-00001.parquet
# export TRAIN_DATASET=datasets/Skywork-OR1-RL-Data/filtered/math-1p5b-filtered-diff-max8.parquet
# export TRAIN_DATASET=datasets/DAPO-Math-17k-Processed/DAPO-Math.parquet
# export TRAIN_DATASET=datasets/skywork/train_7b_math.parquet
# export TRAIN_DATASET=datasets/DAPO-Math-17k-Processed/DAPO-Math_part2.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/verl_format/train.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k
# export TRAIN_DATASET_NAME=POLARIS-4B-S1
# export TRAIN_DATASET_NAME=Skywork-OR1-RL-Data
# export TRAIN_DATASET_NAME=DAPO-Math-17k-1percent
# export TRAIN_DATASET_NAME=DeepMath-103K-filtered-sampled
# export TRAIN_DATASET_NAME=DAPO-Math-17k-10percent
# export TRAIN_DATASET_NAME=OpenThoughts3-1.2M-opd
# export TRAIN_DATASET_NAME=OpenThoughts3-1.2M-30k

export TEST_DATA_DIR=datasets/test_data
# TRAIN_DATASET=${TRAIN_FILE:-["$DATA_DIR/$TASK/train_${SAMPLE_SIZE}.parquet"]}
TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME25/test.parquet", "$TEST_DATA_DIR/AMC23/test.parquet", "$TEST_DATA_DIR/AIME24/test.parquet"]}
# TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME24/test.parquet"]}
# TEST_DATASET=${TEST_FILE:-["$DATA_DIR/AIME24/test.parquet","$DATA_DIR/AIME25/test.parquet","$DATA_DIR/AMC23/test.parquet","$DATA_DIR/MATH-500/test.parquet","$DATA_DIR/Minerva/test.parquet","$DATA_DIR/Olympiad-Bench/test.parquet"]}

# TODO:
# export ACTOR_MODEL_PATH=model/qwen3-1.7b-math-sft
# export ACTOR_MODEL_PATH=model/DS-1.5B-sft
# export ACTOR_MODEL_PATH=model/DS-1.5B-sft-skywork
# export ACTOR_MODEL_PATH=model/DS-1.5B-sft-ds-7b
# export ACTOR_MODEL_PATH=/workspace/model/Qwen3-1.7B-SFT-DAPO-4B-RL
# export ACTOR_MODEL_PATH=/workspace/model/Qwen3-1.7B-SFT-DAPO-4B
# export ACTOR_MODEL_PATH=model/Qwen2.5-Math-1.5B
# export ACTOR_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-1.5B
# export ACTOR_MODEL_PATH=model/JustRL-DeepSeek-1.5B-step_0400
# export ACTOR_MODEL_PATH=model/JustRL-DeepSeek-1.5B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-SFT
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base-SFT-OpenThought3-4B/checkpoint-1800
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base
export ACTOR_MODEL_PATH=model/Qwen3-1.7B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base-SFT-DeepMath-4B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-sft/checkpoint-6000
# export ACTOR_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-7B
# export ACTOR_MODEL_PATH=model/DS-1.5B-SFT
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
# export REWARD_MODEL_PATH=model/Qwen3-4B
# export REWARD_MODEL_PATH=model/Qwen3-4B-grpo
# export REWARD_MODEL_PATH=model/Qwen3-1.7B
# export REWARD_MODEL_PATH=model/OpenMath-Nemotron-1.5B
# export REWARD_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-7B
export REWARD_MODEL_PATH=model/Qwen3-4B-Non-Thinking-RL-Math
# export REWARD_MODEL_PATH=model/Skywork-OR1-Math-7B
# export REWARD_MODEL_PATH=model/Polaris-4B-Preview
# export REWARD_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-14B
# export REWARD_MODEL_PATH=model/JustRL-DeepSeek-1.5B
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

export PROJECT_PATH=checkpoint
export PARALLEL_SIZE=1
SPARSE_PROJECTION_TAG=sp_${ENABLE_SPARSE_PROJECTION}
if [ "$ENABLE_SPARSE_PROJECTION" = "True" ]; then
    SPARSE_PROJECTION_TARGET_TAG=${SPARSE_PROJECTION_TARGET_MATRICES//,/_}
    SPARSE_PROJECTION_TAG=sp_${ENABLE_SPARSE_PROJECTION}-spi_${SPARSE_PROJECTION_INTERVAL}-beta_${SPARSE_PROJECTION_ENERGY_RATIO}-alpha_${SPARSE_PROJECTION_ALPHA}-target_${SPARSE_PROJECTION_TARGET_TAG}
fi
ROW_DIRECTION_REGULARIZER_TAG=rd_${ENABLE_ROW_DIRECTION_REGULARIZER}
if [ "$ENABLE_ROW_DIRECTION_REGULARIZER" = "True" ]; then
    ROW_DIRECTION_REGULARIZER_TARGET_TAG=${ROW_DIRECTION_REGULARIZER_TARGET_MATRICES//,/_}
    ROW_DIRECTION_REGULARIZER_TAG=rd_${ENABLE_ROW_DIRECTION_REGULARIZER}-coef_${ROW_DIRECTION_REGULARIZER_COEF}-mode_${ROW_DIRECTION_REGULARIZER_MODE}-target_${ROW_DIRECTION_REGULARIZER_TARGET_TAG}-rows_${ROW_DIRECTION_REGULARIZER_ROW_SAMPLE_SIZE}
fi
export CKPT_PATH=${PROJECT_PATH}/${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-${SPARSE_PROJECTION_TAG}-${ROW_DIRECTION_REGULARIZER_TAG}-$(date +%Y-%m-%d_%H-%M-%S)
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN

# export VLLM_ATTENTION_BACKEND=XFORMERS
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export SWANLAB_LOG_DIR=${PROJECT_PATH}/swanlab_log
export HYDRA_FULL_ERROR=1


export EXPERIMENT_NAME=${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-${SPARSE_PROJECTION_TAG}-${ROW_DIRECTION_REGULARIZER_TAG}-$(date +%Y-%m-%d_%H-%M-%S)

KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

LR_ARGS=""
if [ "$LR_SCHEDULER" = "cosine" ]; then
    LR_ARGS="actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03"
fi

PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((1024 + MAX_RESP_LENGTH) > 32768) ? (1024 + MAX_RESP_LENGTH) : 32768))
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"


ray start --head --port=6444 --dashboard-port=8444 --temp-dir=/tmp/ray_d
sleep 5


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.grpo_outcome_weight=$GRPO_OUTCOME_WEIGHT \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$((${MINI_BATCH_SIZE}*${PARALLEL_SIZE})) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    $LR_ARGS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward=$ENABLE_FORMAT_REWARD \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=24 \
    custom_reward_function.path="verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=True \
    trainer.log_val_generations=2 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=validation_log/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=$IS_PLOT \
    trainer.sparse_projection.enable=$ENABLE_SPARSE_PROJECTION \
    trainer.sparse_projection.interval=$SPARSE_PROJECTION_INTERVAL \
    trainer.sparse_projection.energy_ratio=$SPARSE_PROJECTION_ENERGY_RATIO \
    trainer.sparse_projection.alpha=$SPARSE_PROJECTION_ALPHA \
    trainer.sparse_projection.max_projection_dim=$SPARSE_PROJECTION_MAX_DIM \
    "trainer.sparse_projection.target_matrices='$SPARSE_PROJECTION_TARGET_MATRICES'" \
    trainer.sparse_projection.reset_optimizer_state=$SPARSE_PROJECTION_RESET_OPTIMIZER \
    actor_rollout_ref.actor.row_direction_regularizer.enable=$ENABLE_ROW_DIRECTION_REGULARIZER \
    actor_rollout_ref.actor.row_direction_regularizer.coef=$ROW_DIRECTION_REGULARIZER_COEF \
    actor_rollout_ref.actor.row_direction_regularizer.mode=$ROW_DIRECTION_REGULARIZER_MODE \
    "actor_rollout_ref.actor.row_direction_regularizer.target_matrices='$ROW_DIRECTION_REGULARIZER_TARGET_MATRICES'" \
    actor_rollout_ref.actor.row_direction_regularizer.row_sample_size=$ROW_DIRECTION_REGULARIZER_ROW_SAMPLE_SIZE \
    actor_rollout_ref.actor.row_direction_regularizer.s=$ROW_DIRECTION_REGULARIZER_S \
    actor_rollout_ref.actor.row_direction_regularizer.eps=$ROW_DIRECTION_REGULARIZER_EPS \
    trainer.hse_monitor.enable=$ENABLE_HSE_MONITOR \
    trainer.hse_monitor.interval=$HSE_MONITOR_INTERVAL \
    "trainer.hse_monitor.target_matrices='$HSE_MONITOR_TARGET_MATRICES'" \
    trainer.hse_monitor.s=$HSE_MONITOR_S \
    trainer.hse_monitor.chunk_size=$HSE_MONITOR_CHUNK_SIZE \
    trainer.hse_monitor.eps=$HSE_MONITOR_EPS \
    trainer.hse_monitor.log_initial=$HSE_MONITOR_LOG_INITIAL

# Log the end time for local runs.
if [ -z "$SLURM_JOB_ID" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi
