#!/bin/bash
# Create the complete LeRobot 0.6 ACT training/inference/policy-server env.
#
# Fresh server usage:
#   cd /data/RoboDojo
#   bash XPolicyLab/policy/act_lerobot/install.sh
#
# Optional environment override:
#   LEROBOT_ENV_NAME=lerobot bash XPolicyLab/policy/act_lerobot/install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
POLICY_ENV=${LEROBOT_ENV_NAME:-lerobot}

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=================================================="
echo "[1/6] 创建或复用 Conda 环境: ${POLICY_ENV} (Python 3.12)"
echo "=================================================="
if conda env list | awk '{print $1}' | grep -Fxq "${POLICY_ENV}"; then
    echo "环境已存在，将在原环境中校准依赖版本。"
else
    conda create -n "${POLICY_ENV}" python=3.12 -y
fi
conda activate "${POLICY_ENV}"

echo "=================================================="
echo "[2/6] 安装 PyTorch 2.7.1 + CUDA 12.8"
echo "=================================================="
python -m pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

echo "=================================================="
echo "[3/6] 安装 LeRobot 0.6.0（训练 + dataset）"
echo "=================================================="
python -m pip install --no-cache-dir \
    'lerobot[training,dataset]==0.6.0'

echo "=================================================="
echo "[4/6] 安装 FFmpeg 与 PyAV"
echo "=================================================="
conda install -n "${POLICY_ENV}" -c conda-forge ffmpeg -y
python -m pip install --no-cache-dir av

echo "=================================================="
echo "[5/6] 安装 XPolicyLab WebSocket 服务依赖"
echo "=================================================="
python -m pip install --no-cache-dir \
    pyyaml opencv-python websockets msgpack msgpack-numpy pydantic

echo "=================================================="
echo "[6/6] 验证环境（直接使用 git clone 后的源码）"
echo "=================================================="
PYTHONPATH="${XPL_ROOT}/..:${XPL_ROOT}:${PYTHONPATH:-}" python - <<'PY'
import av
import cv2
import lerobot
import msgpack
import msgpack_numpy
import pydantic
import torch
import websockets

from client_server.ws.model_server import PolicyServer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

version = getattr(lerobot, "__version__", "")
if not version.startswith("0.6."):
    raise RuntimeError(f"Expected LeRobot 0.6.x, got {version!r}")

print(f"Python/Torch environment OK")
print(f"LeRobot: {version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA runtime: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
print(f"PyAV: {av.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"WebSockets: {websockets.__version__}")
print("ACT + dataset + policy-server imports: OK")
PY

echo "=================================================="
echo "安装完成"
echo "  conda activate ${POLICY_ENV}"
echo "=================================================="
