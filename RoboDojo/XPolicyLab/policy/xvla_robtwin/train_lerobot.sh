#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="${CONDA_ENV:-XVLA}"
MODEL_PATH="${MODEL_PATH:-${POLICY_DIR}/checkpoints/shared/X-VLA-RoboTwin2}"
DATASET_ROOT="${DATASET_ROOT:-${POLICY_DIR}/../data/lerobot_v30_ee}"
REPO_ID="${REPO_ID:-lerobot_v30_ee}"
OUTPUT_DIR="${OUTPUT_DIR:-${POLICY_DIR}/checkpoints/lerobot-v3-$(date +%Y%m%d-%H%M%S)}"
SPLIT_PATH="${SPLIT_PATH:-}"
TASKS_JSON="${TASKS_JSON:-[]}"
ALLOW_ALL_EPISODES="${ALLOW_ALL_EPISODES:-0}"
ALLOW_ALL_TASKS="${ALLOW_ALL_TASKS:-0}"
DOMAIN_ID="${DOMAIN_ID:-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
STEPS="${STEPS:-10000}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SEED="${SEED:-42}"
MIXED_PRECISION="${MIXED_PRECISION:-auto}"
GPU_IDS="${GPU_IDS:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Base model config not found: ${MODEL_PATH}/config.json" >&2
  exit 2
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "LeRobot v3 metadata not found: ${DATASET_ROOT}/meta/info.json" >&2
  exit 2
fi
if [[ "${DOMAIN_ID}" != "6" ]]; then
  echo "RoboDojo post-training from X-VLA-RoboTwin2 requires DOMAIN_ID=6." >&2
  exit 2
fi
if [[ -n "${SPLIT_PATH}" && ! -f "${SPLIT_PATH}" ]]; then
  echo "Split file not found: ${SPLIT_PATH}" >&2
  exit 2
fi
if [[ -z "${SPLIT_PATH}" && "${ALLOW_ALL_EPISODES}" != "1" ]]; then
  echo "SPLIT_PATH is required; set ALLOW_ALL_EPISODES=1 only for intentional all-data training." >&2
  exit 2
fi
if [[ "${TASKS_JSON}" == "[]" && "${ALLOW_ALL_TASKS}" != "1" ]]; then
  echo "TASKS_JSON must select exact tasks; set ALLOW_ALL_TASKS=1 only to train all tasks." >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${POLICY_DIR}"

if [[ "${MIXED_PRECISION}" == "auto" ]]; then
  MIXED_PRECISION=$(CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -c \
    'import torch; print("bf16" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else "fp16")')
fi
if [[ "${MIXED_PRECISION}" != "fp16" && "${MIXED_PRECISION}" != "bf16" && "${MIXED_PRECISION}" != "no" ]]; then
  echo "MIXED_PRECISION must be auto, fp16, bf16, or no; got ${MIXED_PRECISION}" >&2
  exit 2
fi

ARGS=(
  --model-path "${MODEL_PATH}"
  --dataset-root "${DATASET_ROOT}"
  --repo-id "${REPO_ID}"
  --output-dir "${OUTPUT_DIR}"
  --tasks-json "${TASKS_JSON}"
  --domain-id "${DOMAIN_ID}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}"
  --steps "${STEPS}"
  --learning-rate "${LR}"
  --num-workers "${NUM_WORKERS}"
  --save-interval "${SAVE_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --seed "${SEED}"
  --mixed-precision "${MIXED_PRECISION}"
)
if [[ -n "${SPLIT_PATH}" ]]; then
  ARGS+=(--split-path "${SPLIT_PATH}")
fi
if [[ "${ALLOW_ALL_EPISODES}" == "1" ]]; then
  ARGS+=(--allow-all-episodes)
fi
if [[ "${ALLOW_ALL_TASKS}" == "1" ]]; then
  ARGS+=(--allow-all-tasks)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

echo "[xvla_robtwin] base_model=${MODEL_PATH}"
echo "[xvla_robtwin] dataset=${DATASET_ROOT}"
echo "[xvla_robtwin] split=${SPLIT_PATH:-all episodes}"
echo "[xvla_robtwin] domain_id=${DOMAIN_ID}"
echo "[xvla_robtwin] mixed_precision=${MIXED_PRECISION}"
echo "[xvla_robtwin] output=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --dynamo_backend no \
  --mixed_precision "${MIXED_PRECISION}" \
  "${POLICY_DIR}/train_lerobot.py" \
  "${ARGS[@]}"
