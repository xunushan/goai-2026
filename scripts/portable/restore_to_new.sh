#!/usr/bin/env bash
# restore_to_new.sh — 在【新机器】上把 pi05_portable/data/ 子树镜像还原到 /data/
# 用法（root）：bash <pi05_portable>/restore_to_new.sh
# 目标：所有绝对路径与源机一致（/data/openpi、/data/goai/envs、/data/checkpoints、/data/data）,
#      使打包的 venv/base/数据直接命中，无需 relocate。
# 若不想还原到 /data：改 TARGET 为你选的根，并按 README §3 relocate env 指针。
set -euo pipefail
SRC="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data}"
TARGET="${TARGET:-/data}"
[ "$(id -u)" = 0 ] || { echo "请用 root 运行（需要写 $TARGET）"; exit 1; }
[ -d "$SRC" ] || { echo "找不到包内 data/ 目录: $SRC"; exit 1; }
echo "== 还原 $SRC → $TARGET（需 root；镜像覆盖到 $TARGET 下同名路径）=="
rsync -a --info=progress2 "$SRC/" "$TARGET/"
echo
echo "== 还原完成。接下来："
echo "  1) 复用 env 验证:  GOAI_ROOT=$TARGET/goai DATASET_ROOT=$TARGET/data OPENPI_SRC=$TARGET/openpi bash $TARGET/pi05_env_scripts/verify_env.sh"
echo "  2) 训练用户软链:  mkdir -p ~/.cache/huggingface/lerobot && ln -sfn $TARGET/data/sim_lerobot_v30_joint-224x224 ~/.cache/huggingface/lerobot/RoboDojo_sim_arx-x5_v30"
echo "  3) Blackwell/jax 核验 + S1 冒烟（见 README §5/§7）"
