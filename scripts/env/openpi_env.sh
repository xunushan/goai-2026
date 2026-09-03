# =============================================================================
# openpi_env.sh — 进入 pi05_l060 环境跑 openpi 训练/脚本的 shell 环境导出
#
# 用法（必须 source，否则环境变量不生效）。openpi 仓库根必传，不预设路径：
#   source scripts/env/openpi_env.sh <openpi 仓库根>
#   例: source scripts/env/openpi_env.sh /path/to/openpi
#   或先 export OPENPI_SRC=<openpi 仓库根> 再 source（无参时读它）
# 之后直接调你的脚本：
#   python train.py ...        # 已指向 venv python + openpi 在 PYTHONPATH
#   或用显式解释器: "$PY" train.py
# =============================================================================
ENV_NAME="${ENV_NAME:-pi05_l060}"
export GOAI_ROOT="${GOAI_ROOT:-/cloud/cloud-ssd1/goai}"
export VENV_DIR="$GOAI_ROOT/envs/$ENV_NAME"
export PY="$VENV_DIR/bin/python"                       # venv 解释器（绝对路径）
SP="$VENV_DIR/lib/python3.12/site-packages"

# openpi 仓库根：第 1 参 > OPENPI_SRC，必传（不预设路径）。
# openpi 包在 <根>/src/openpi，client 在 <根>/packages/openpi-client/src，据此拼 PYTHONPATH。
OPENPI_SRC="${1:-$OPENPI_SRC}"
if [ -z "$OPENPI_SRC" ]; then
  echo "错误: 未传 openpi 仓库根。用法: source openpi_env.sh <openpi 仓库根>（或先 export OPENPI_SRC=<仓库根>）" >&2
  return 1 2>/dev/null || exit 1
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
