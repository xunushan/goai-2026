#!/bin/bash
# ACT 训练脚本 (针对本地 lerobot_v30_ee 数据集)
# 数据集先按 TASKS_JSON 过滤，再按 task 分层，以 episode 为单位固定划分
# 90% train / 10% val。
# 本次训练使用 LeRobot 0.6 官方训练入口，只加载固定的 train episode。
# 训练期间不访问 val 集；训练结束后再单独计算一次 val_loss。
set -eo pipefail

source /root/miniconda/etc/profile.d/conda.sh
conda activate lerobot

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 训练超参数 (针对 T4 15GB 显存调优)
# 必须与 create_split.sh 使用相同的精确任务文本。
# 若要训练全部任务，请显式设置 TASKS_JSON='[]'。
TASKS_JSON=${TASKS_JSON:-'["Stack the three blocks with different textures."]'}
TASK_COUNT=$(python -c \
    'import json,sys; tasks=json.loads(sys.argv[1]); assert isinstance(tasks,list), "TASKS_JSON 必须是 JSON 列表"; print(len(tasks))' \
    "$TASKS_JSON")
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-5}
TARGET_EPOCHS=${TARGET_EPOCHS:-3}
LOG_FREQ=${LOG_FREQ:-100}
CHUNK_SIZE=${CHUNK_SIZE:-50}
N_ACTION_STEPS=${N_ACTION_STEPS:-10}
NUM_WORKERS=${NUM_WORKERS:-6}
SEED=${SEED:-42}
DATASET_ROOT=${DATASET_ROOT:-/workspace/data/lerobot_v30_ee}
TRAIN_RATIO=${TRAIN_RATIO:-0.9}
RATIO_TAG=$(python -c \
    'import sys; print(round(float(sys.argv[1]) * 100))' \
    "$TRAIN_RATIO")
TASK_HASH=$(python -c \
    'import hashlib,json,sys; tasks=json.loads(sys.argv[1]); assert isinstance(tasks,list), "TASKS_JSON 必须是 JSON 列表"; print("" if not tasks else "_tasks_"+hashlib.sha256(json.dumps(tasks,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:8])' \
    "$TASKS_JSON")
SPLIT_PATH=${SPLIT_PATH:-/workspace/splits/lerobot_v30_ee_train${RATIO_TAG}${TASK_HASH}_seed${SEED}.json}

OUTPUT_DIR=${OUTPUT_DIR:-/workspace/outputs/act_$(date +%Y%m%d_%H%M%S)}

# 划分必须提前通过 create_split.sh 创建；训练过程只读取，不重新划分。
if [ ! -f "$SPLIT_PATH" ]; then
    echo "错误：固定划分文件不存在：$SPLIT_PATH" >&2
    echo "请先执行：bash $SCRIPT_DIR/create_split.sh" >&2
    exit 2
fi

TRAIN_EPISODES=$(python -c \
    'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); assert data.get("train"), "split 中缺少非空 train"; print(json.dumps(data["train"], separators=(",", ":")))' \
    "$SPLIT_PATH")
TRAIN_EPISODE_COUNT=$(python -c \
    'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); print(len(data["train"]))' \
    "$SPLIT_PATH")
VAL_EPISODE_COUNT=$(python -c \
    'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); print(len(data["val"]))' \
    "$SPLIT_PATH")
TRAIN_FRAME_COUNT=$(python -c \
    'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); print(sum(int(row["train_frames"]) for row in data["summary"]["per_task"]))' \
    "$SPLIT_PATH")

if [ "$TRAIN_FRAME_COUNT" -le 0 ]; then
    echo "错误：split 中没有训练帧：$SPLIT_PATH" >&2
    exit 2
fi

# LeRobotDataset 的一个 sample 对应一帧。默认按实际训练帧数计算约
# TARGET_EPOCHS 个 epoch；显式传入 STEPS 时仍允许覆盖。
if [ -z "${STEPS:-}" ]; then
    STEPS=$(python -c \
        'import math,sys; print(math.ceil(int(sys.argv[1])*float(sys.argv[2])/int(sys.argv[3])))' \
        "$TRAIN_FRAME_COUNT" "$TARGET_EPOCHS" "$BATCH_SIZE")
fi

# 默认约每个 epoch 保存一次 checkpoint；显式传入 SAVE_FREQ 可覆盖。
if [ -z "${SAVE_FREQ:-}" ]; then
    SAVE_FREQ=$(python -c \
        'import sys; print(max(1, int(sys.argv[1])//int(sys.argv[2])))' \
        "$TRAIN_FRAME_COUNT" "$BATCH_SIZE")
fi

EFFECTIVE_EPOCHS=$(python -c \
    'import sys; print(f"{int(sys.argv[1])*int(sys.argv[2])/int(sys.argv[3]):.3f}")' \
    "$STEPS" "$BATCH_SIZE" "$TRAIN_FRAME_COUNT")

START_TIME=$(date +%s)
echo "=================================================="
echo "开始 ACT 训练"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  数据集:   $DATASET_ROOT"
echo "  划分文件: $SPLIT_PATH"
echo "  Task列表: $TASKS_JSON"
echo "  Episode:  train=$TRAIN_EPISODE_COUNT, val=$VAL_EPISODE_COUNT"
echo "  Train帧:  $TRAIN_FRAME_COUNT"
echo "  输出目录: $OUTPUT_DIR"
echo "  Batch:     $BATCH_SIZE"
echo "  LR:        $LR"
echo "  Steps:     $STEPS"
echo "  目标Epoch: $TARGET_EPOCHS"
echo "  实际Epoch: $EFFECTIVE_EPOCHS"
echo "  Chunk:     $CHUNK_SIZE"
echo "  ActSteps:  $N_ACTION_STEPS"
echo "  Workers:   $NUM_WORKERS"
echo "  Seed:      $SEED"
echo "  SaveFreq:  $SAVE_FREQ"
echo "  LogFreq:   $LOG_FREQ"
echo "  Validation: 训练期间关闭，训练完成后独立执行一次"
echo "=================================================="

# 使用 nosleep 防止 CloudStudio 休眠 (若存在)
NO_SLEEP=""
if [ -x "/workspace/nosleep" ]; then
    NO_SLEEP="/workspace/nosleep"
fi

$NO_SLEEP python -m lerobot.scripts.lerobot_train \
    --policy.type=act \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot_v30_ee \
    --dataset.root="$DATASET_ROOT" \
    --dataset.episodes="$TRAIN_EPISODES" \
    --dataset.video_backend=pyav \
    --output_dir="$OUTPUT_DIR" \
    --batch_size=$BATCH_SIZE \
    --policy.optimizer_lr=$LR \
    --policy.chunk_size=$CHUNK_SIZE \
    --policy.n_action_steps=$N_ACTION_STEPS \
    --steps=$STEPS \
    --log_freq=$LOG_FREQ \
    --eval_steps=0 \
    --save_freq=$SAVE_FREQ \
    --num_workers=$NUM_WORKERS \
    --seed=$SEED \
    --save_checkpoint=true

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))
SECS=$((ELAPSED % 60))

echo "=================================================="
echo "训练完成! 结果保存到: $OUTPUT_DIR"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  总耗时:   ${HOURS}h ${MINS}m ${SECS}s"
echo "=================================================="

# 将实际使用的数据划分随训练产物保存，便于复现实验。
cp "$SPLIT_PATH" "$OUTPUT_DIR/dataset_split.json"

# 将最新 checkpoint 转换为 PyTorch .ckpt 格式
echo "=================================================="
echo "开始转换 last checkpoint 为 .ckpt 格式"
echo "=================================================="
python "$SCRIPT_DIR/convert_to_ckpt.py" "$OUTPUT_DIR/checkpoints/last"

echo "=================================================="
echo "全部完成!"
echo "  .ckpt 文件: $OUTPUT_DIR/checkpoints/model.ckpt"
echo "  验证命令: bash $SCRIPT_DIR/evaluate_act.sh $OUTPUT_DIR/checkpoints/last/pretrained_model"
echo "=================================================="
