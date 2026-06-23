#!/usr/bin/env bash
# Step 0 | Prepare GSM8K + MATH datasets (RL/prompt format) shared by A / D / E.
#
# Produces:
#   /home/jcgu/qyliu/RL-OPD/data/gsm8k/{train,test}.parquet   (prompt + reward_model.ground_truth)
#   /home/jcgu/qyliu/RL-OPD/data/math/{train,test}.parquet
#
# These are the SAME files consumed by the GRPO and OPD trainers, so every
# experiment sees an identical prompt set / split.

set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${REPO_ROOT}"

# Make the in-repo `verl` package importable even when it isn't pip-installed.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DATA_ROOT="/home/jcgu/qyliu/RL-OPD/data"

# IMPORTANT: raw (input) and processed (output) directories MUST be different,
# otherwise the processed parquet overwrites the raw data and re-runs fail with
# KeyError: 'question' / 'problem'.
#
# Path to the RAW GSM8K dataset (parquet with the original question/answer
# columns). Override by exporting RAW_GSM8K_DIR. Leave empty to download from HF.
RAW_GSM8K_DIR="${RAW_GSM8K_DIR:-${DATA_ROOT}/gsm8k_raw}"
# Path to the RAW MATH-lighteval dataset. Leave empty to use the HF cache /
# download (DigitalLearningGmbH/MATH-lighteval).
RAW_MATH_DIR="${RAW_MATH_DIR:-}"

GSM8K_SRC_ARG=()
if [[ -n "${RAW_GSM8K_DIR}" && -d "${RAW_GSM8K_DIR}" ]]; then
    GSM8K_SRC_ARG=(--local_dataset_path "${RAW_GSM8K_DIR}")
fi

MATH_SRC_ARG=()
if [[ -n "${RAW_MATH_DIR}" && -d "${RAW_MATH_DIR}" ]]; then
    MATH_SRC_ARG=(--local_dataset_path "${RAW_MATH_DIR}")
fi

python3 examples/data_preprocess/gsm8k.py "${GSM8K_SRC_ARG[@]}" --local_save_dir "${DATA_ROOT}/gsm8k"
python3 examples/data_preprocess/math_dataset.py "${MATH_SRC_ARG[@]}" --local_save_dir "${DATA_ROOT}/math"

# SFT/messages-format GSM8K test split, used as the validation file for run_A_sft.sh.
python3 examples/data_preprocess/gsm8k_multiturn_sft.py "${GSM8K_SRC_ARG[@]}" --local_save_dir "${DATA_ROOT}/gsm8k_sft"

echo "Done. RL/prompt-format parquet files are under ${DATA_ROOT}/{gsm8k,math}."
echo "SFT-format GSM8K (for A's validation) is under ${DATA_ROOT}/gsm8k_sft."
