# =============================================================================
# openpi_env.sh — 进入 pi05_l060 环境跑 openpi 训练/脚本的 shell 环境导出
#
# 用法（必须 source，否则环境变量不生效）:
#   source scripts/env/openpi_env.sh [OPENPI_SRC路径]
#   例: source scripts/env/openpi_env.sh /data/openpi
# 之后直接调你的脚本：
#   python train.py ...        # 已指向 venv python + openpi 在 PYTHONPATH
#   或用显式解释器: "$PY" train.py
# =============================================================================
ENV_NAME="${ENV_NAME:-pi05_l060}"
export GOAI_ROOT="${GOAI_ROOT:-/cloud/cloud-ssd1/goai}"
export VENV_DIR="$GOAI_ROOT/envs/$ENV_NAME"
export PY="$VENV_DIR/bin/python"                       # venv 解释器（绝对路径）
SP="$VENV_DIR/lib/python3.12/site-packages"

# openpi 源码位置：第 1 参 > OPENPI_SRC > 探测 /data/openpi
OPENPI_SRC="${1:-$OPENPI_SRC}"
if [ -z "$OPENPI_SRC" ] && [ -d "/data/openpi/src/openpi" ]; then
  OPENPI_SRC="/data/openpi"
fi
export OPENPI_SRC
export PYTHONPATH="$OPENPI_SRC/src:$OPENPI_SRC/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

# torchcodec dlopen 需要 av.libs(哈希so) + fflib(soname软链)，必须先于 PIL/lerobot
export LD_LIBRARY_PATH="$VENV_DIR/fflib:$SP/av.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "venv        = $VENV_DIR"
echo "openpi src  = $OPENPI_SRC"
echo "python      = $PY"
echo "PYTHONPATH  = $PYTHONPATH"
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
