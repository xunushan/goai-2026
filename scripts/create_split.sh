#!/bin/bash
# 单独创建一次固定的 episode 训练/验证划分。
#
# 用法：
#   bash scripts/create_split.sh
#
# 可选环境变量：
#   DATASET_ROOT=/workspace/data/lerobot_v30_ee
#   SPLIT_PATH=/workspace/splits/lerobot_v30_ee_train90_seed42.json
#   TRAIN_RATIO=0.9
#   SEED=42
#   TASKS_JSON='["Stack the three blocks with different textures."]'
#
# TASKS_JSON 是精确任务文本的 JSON 列表；不设置则保留所有任务。
# 如果 SPLIT_PATH 已存在，Python 脚本会验证 ratio、seed 和 task 列表是否一致。
set -eo pipefail

source /root/miniconda/etc/profile.d/conda.sh
conda activate lerobot

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATASET_ROOT=${DATASET_ROOT:-/workspace/data/lerobot_v30_ee}
SEED=${SEED:-42}
TRAIN_RATIO=${TRAIN_RATIO:-0.9}

RATIO_TAG=$(python -c \
    'import sys; print(round(float(sys.argv[1]) * 100))' \
    "$TRAIN_RATIO")

SPLIT_PATH=${SPLIT_PATH:-/workspace/splits/lerobot_v30_ee_train${RATIO_TAG}_seed${SEED}.json}

mkdir -p "$(dirname "$SPLIT_PATH")"

SPLIT_CMD=(
    python "$SCRIPT_DIR/../utils/split_episodes.py"
    --dataset-root "$DATASET_ROOT" \
    --output "$SPLIT_PATH" \
    --train-ratio "$TRAIN_RATIO" \
    --seed "$SEED"
)

if [ -n "${TASKS_JSON:-}" ]; then
    mapfile -t TASKS < <(python -c \
        'import json,sys; tasks=json.loads(sys.argv[1]); assert isinstance(tasks,list), "TASKS_JSON must be a JSON list"; assert all(isinstance(v,str) and v for v in tasks), "each task must be a non-empty string"; print(*tasks,sep="\n")' \
        "$TASKS_JSON")
    if [ "${#TASKS[@]}" -gt 0 ]; then
        TASK_HASH=$(python -c \
            'import hashlib,json,sys; tasks=json.loads(sys.argv[1]); print("_tasks_"+hashlib.sha256(json.dumps(tasks,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:8])' \
            "$TASKS_JSON")
        SPLIT_PATH=/workspace/splits/lerobot_v30_ee_train${RATIO_TAG}${TASK_HASH}_seed${SEED}.json
        mkdir -p "$(dirname "$SPLIT_PATH")"
        SPLIT_CMD=(
            python "$SCRIPT_DIR/../utils/split_episodes.py"
            --dataset-root "$DATASET_ROOT" \
            --output "$SPLIT_PATH" \
            --train-ratio "$TRAIN_RATIO" \
            --seed "$SEED"
            --tasks "${TASKS[@]}"
        )
    fi
fi

"${SPLIT_CMD[@]}"

echo "固定划分创建完成：$SPLIT_PATH"
