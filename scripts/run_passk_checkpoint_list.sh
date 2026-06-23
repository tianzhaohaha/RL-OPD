#!/usr/bin/env bash
set -uo pipefail

# Put the checkpoints you want to evaluate here, one per line.
# Both .../global_step_xxx/actor and .../global_step_xxx are supported.
CHECKPOINTS=(
    "/home/jcgu/qyliu/RL-OPD/verl/checkpoints/verl_grpo_gsm8k/qwen3_4b_grpo_gsm8k_2gpu_fsdp_20260612_2342/global_step_500/actor"
)

for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "============================================================"
    echo "Evaluating checkpoint: ${checkpoint}"
    echo "============================================================"

    python verl/scripts/eval_checkpoint_passk.py \
        "${checkpoint}" \
        --k 1,2,4,8 \
        --data-file data/gsm8k/test.parquet \
        --backend vllm \
        --cuda-visible-devices 2 \
        --gpus 1 \
        --tp 1 \
        --temperature 1.0 \
        --top-p 1.0 \
        --max-model-len 8192 \
        --max-response-length 4096

    status=$?
    if [[ ${status} -ne 0 ]]; then
        echo "Failed checkpoint: ${checkpoint}"
        echo "Exit code: ${status}"
    else
        echo "Finished checkpoint: ${checkpoint}"
    fi
done