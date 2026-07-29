#!/bin/bash
# 对一次训练产生的所有 LeRobot ACT checkpoint 依次执行离线 validation。
# 只读取 split JSON 中的 val episode，不更新模型参数。
#
# 用法：
#   bash act/evaluate_act.sh \
#       /workspace/outputs/act_xxx/checkpoints/last/pretrained_model
# 也可以直接传 checkpoints 目录：
#   bash act/evaluate_act.sh \
#       /workspace/outputs/act_xxx/checkpoints
#
# 可选环境变量：
#   DATASET_ROOT、SPLIT_PATH、BATCH_SIZE、NUM_WORKERS、
#   VAL_MAX_SAMPLES、VIDEO_BACKEND、OUTPUT_DIR
set -eo pipefail

source /root/miniconda/etc/profile.d/conda.sh
conda activate lerobot

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

CHECKPOINT_INPUT=${1:-${CHECKPOINT:-}}
if [ -z "$CHECKPOINT_INPUT" ]; then
    echo "错误：请传入 checkpoints 目录或任一 checkpoint 的 pretrained_model 目录。" >&2
    echo "示例：bash $0 /workspace/outputs/act_xxx/checkpoints" >&2
    exit 2
fi

DATASET_ROOT=${DATASET_ROOT:-/workspace/data/lerobot_v30_ee}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-6}
# val 集共 1877 × 64 = 120128 个 frame，默认均匀抽取 1/4。
# 可通过 VAL_MAX_SAMPLES 覆盖；设置为 0 表示评估完整 val 集。
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-30032}
VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}

if [ ! -d "$CHECKPOINT_INPUT" ]; then
    echo "错误：checkpoint 路径不存在：$CHECKPOINT_INPUT" >&2
    exit 2
fi
# 接受 checkpoints 根目录，也兼容原来的 .../last/pretrained_model 参数。
if [ "$(basename "$CHECKPOINT_INPUT")" = "checkpoints" ]; then
    CHECKPOINTS_DIR=$(cd "$CHECKPOINT_INPUT" && pwd -P)
elif [ "$(basename "$CHECKPOINT_INPUT")" = "pretrained_model" ]; then
    RESOLVED_CHECKPOINT=$(cd "$CHECKPOINT_INPUT" && pwd -P)
    STEP_DIR=$(dirname "$RESOLVED_CHECKPOINT")
    CHECKPOINTS_DIR=$(dirname "$STEP_DIR")
else
    echo "错误：参数必须是 checkpoints 目录或 pretrained_model 目录：$CHECKPOINT_INPUT" >&2
    exit 2
fi

RUN_DIR=$(dirname "$CHECKPOINTS_DIR")
if [ -z "${SPLIT_PATH:-}" ]; then
    if [ -f "$RUN_DIR/dataset_split.json" ]; then
        SPLIT_PATH="$RUN_DIR/dataset_split.json"
    else
        SPLIT_PATH=/workspace/splits/lerobot_v30_ee_train90_seed42.json
    fi
fi
if [ ! -f "$SPLIT_PATH" ]; then
    echo "错误：split 文件不存在：$SPLIT_PATH" >&2
    exit 2
fi
OUTPUT_DIR=${OUTPUT_DIR:-"$RUN_DIR/val_metrics"}
mkdir -p "$OUTPUT_DIR"

# 只匹配六位数字的 step 目录，自动排除 last 软链接，并从最大 step 开始。
CHECKPOINT_DIRS=()
while IFS= read -r CHECKPOINT_DIR; do
    CHECKPOINT_DIRS+=("$CHECKPOINT_DIR")
done < <(
    find "$CHECKPOINTS_DIR" -mindepth 1 -maxdepth 1 -type d \
        -name '[0-9][0-9][0-9][0-9][0-9][0-9]' |
        sort -r
)

if [ "${#CHECKPOINT_DIRS[@]}" -eq 0 ]; then
    echo "错误：没有在 $CHECKPOINTS_DIR 中找到六位数字 checkpoint 目录。" >&2
    exit 2
fi

echo "=================================================="
echo "ACT 所有 checkpoint 训练后离线验证"
echo "  Checkpoints: $CHECKPOINTS_DIR"
echo "  Count:       ${#CHECKPOINT_DIRS[@]}"
echo "  Dataset:    $DATASET_ROOT"
echo "  Split:      $SPLIT_PATH"
echo "  Batch:      $BATCH_SIZE"
echo "  Workers:    $NUM_WORKERS"
echo "  MaxSamples: $VAL_MAX_SAMPLES (0 表示完整 val 集)"
echo "  OutputDir:  $OUTPUT_DIR"
echo "=================================================="

for STEP_DIR in "${CHECKPOINT_DIRS[@]}"; do
    STEP=$(basename "$STEP_DIR")
    CHECKPOINT="$STEP_DIR/pretrained_model"
    OUTPUT="$OUTPUT_DIR/val_metrics_${STEP}.json"

    if [ ! -d "$CHECKPOINT" ]; then
        echo "错误：缺少 pretrained_model 目录：$CHECKPOINT" >&2
        exit 2
    fi

    echo
    echo "--------------------------------------------------"
    echo "开始验证 checkpoint $STEP"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Output:     $OUTPUT"
    echo "--------------------------------------------------"

    python "$SCRIPT_DIR/evaluate_val_loss.py" \
        --checkpoint "$CHECKPOINT" \
        --dataset-root "$DATASET_ROOT" \
        --repo-id lerobot_v30_ee \
        --split-path "$SPLIT_PATH" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --video-backend "$VIDEO_BACKEND" \
        --max-samples "$VAL_MAX_SAMPLES" \
        --output "$OUTPUT"
done

echo
echo "=================================================="
echo "全部 checkpoint 验证完成"
echo "指标目录：$OUTPUT_DIR"
echo "=================================================="
