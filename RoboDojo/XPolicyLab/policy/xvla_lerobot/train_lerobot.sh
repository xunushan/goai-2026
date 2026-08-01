#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-XVLA}"
MODEL_PATH="${MODEL_PATH:-${POLICY_DIR}/checkpoints/shared/X-VLA-RoboTwin2}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the LeRobot v3 dataset root}"
SPLIT_PATH="${SPLIT_PATH:-}"
EPISODES_JSON="${EPISODES_JSON:-}"
ALLOW_ALL_EPISODES="${ALLOW_ALL_EPISODES:-0}"
TASKS_JSON="${TASKS_JSON:-[]}"
OUTPUT_DIR="${OUTPUT_DIR:-${POLICY_DIR}/checkpoints/arx-ee-$(date +%Y%m%d-%H%M%S)}"
GPU_IDS="${GPU_IDS:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ITERS="${ITERS:-30000}"
FREEZE_STEPS="${FREEZE_STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LEARNING_COEF="${LEARNING_COEF:-0.1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
SEED="${SEED:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
MIXED_PRECISION="${MIXED_PRECISION:-auto}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Base model config not found: ${MODEL_PATH}/config.json" >&2
    exit 2
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "LeRobot v3 metadata not found: ${DATASET_ROOT}/meta/info.json" >&2
    exit 2
fi
selection_count=0
[[ -n "${SPLIT_PATH}" ]] && selection_count=$((selection_count + 1))
[[ -n "${EPISODES_JSON}" ]] && selection_count=$((selection_count + 1))
[[ "${ALLOW_ALL_EPISODES}" == "1" ]] && selection_count=$((selection_count + 1))
if [[ "${selection_count}" -ne 1 ]]; then
    echo "Set exactly one of SPLIT_PATH, EPISODES_JSON, or ALLOW_ALL_EPISODES=1." >&2
    exit 2
fi
if [[ -n "${SPLIT_PATH}" && ! -f "${SPLIT_PATH}" ]]; then
    echo "Episode split not found: ${SPLIT_PATH}" >&2
    exit 2
fi
if [[ "${NUM_PROCESSES}" != "1" ]]; then
    echo "NUM_PROCESSES must remain 1: the upstream iterable loader is not Accelerate-sharded." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
RUNTIME_META="${OUTPUT_DIR}/train_meta.json"
export XVLA_DATASET_ROOT="${DATASET_ROOT}"
export XVLA_SPLIT_PATH="${SPLIT_PATH}"
export XVLA_EPISODES_JSON="${EPISODES_JSON}"
export XVLA_ALLOW_ALL_EPISODES="${ALLOW_ALL_EPISODES}"
export XVLA_TASKS_JSON="${TASKS_JSON}"
export XVLA_VIDEO_BACKEND="${VIDEO_BACKEND}"
export XVLA_RUNTIME_META="${RUNTIME_META}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ "${MIXED_PRECISION}" == "auto" ]]; then
    MIXED_PRECISION=$(CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -c \
        'import torch; print("bf16" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else "fp16")')
fi
if [[ "${MIXED_PRECISION}" != "fp16" && "${MIXED_PRECISION}" != "bf16" ]]; then
    echo "MIXED_PRECISION must be auto, fp16, or bf16; got ${MIXED_PRECISION}" >&2
    exit 2
fi

python - <<'PY'
import json
import os
from pathlib import Path

meta = {
    "dataset_name": "RoboDojo_LerobotV3_ARX_EE",
    "dataset_root": str(Path(os.environ["XVLA_DATASET_ROOT"]).expanduser().resolve()),
    "repo_id": "lerobot_v30_ee",
    "domain_id": 6,
    "fps": 25,
    "query_duration": 1.0,
    "video_backend": os.environ["XVLA_VIDEO_BACKEND"],
    "observation_key": [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ],
    "tasks": json.loads(os.environ["XVLA_TASKS_JSON"]),
}
split_path = os.environ["XVLA_SPLIT_PATH"]
episodes_json = os.environ["XVLA_EPISODES_JSON"]
if split_path:
    meta["episode_split_path"] = str(Path(split_path).expanduser().resolve())
    meta["episode_split"] = "train"
elif episodes_json:
    candidate = Path(episodes_json).expanduser()
    meta["episodes"] = json.loads(candidate.read_text() if candidate.is_file() else episodes_json)
else:
    meta["allow_all_episodes"] = True
Path(os.environ["XVLA_RUNTIME_META"]).write_text(
    json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
PY
if [[ -n "${SPLIT_PATH}" ]]; then
    cp "${SPLIT_PATH}" "${OUTPUT_DIR}/dataset_split.json"
fi

echo "[xvla_lerobot] model=${MODEL_PATH}"
echo "[xvla_lerobot] dataset=${DATASET_ROOT} fps=25 qdur=1.0 anchors=30"
echo "[xvla_lerobot] meta=${RUNTIME_META} domain_id=6 action_mode=arx_ee6d"
echo "[xvla_lerobot] output=${OUTPUT_DIR}"
echo "[xvla_lerobot] mixed_precision=${MIXED_PRECISION} num_processes=1"

cd "${POLICY_DIR}"
exec env CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --mixed_precision "${MIXED_PRECISION}" \
    xvla/train.py \
    --models "${MODEL_PATH}" \
    --train_metas_path "${RUNTIME_META}" \
    --action_mode arx_ee6d \
    --learning_rate "${LEARNING_RATE}" \
    --learning_coef "${LEARNING_COEF}" \
    --iters "${ITERS}" \
    --freeze_steps "${FREEZE_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --save_interval "${SAVE_INTERVAL}"
