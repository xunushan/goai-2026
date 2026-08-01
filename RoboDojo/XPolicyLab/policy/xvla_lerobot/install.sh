#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${XVLA_CONDA_ENV:-XVLA}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.1}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${XVLA_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
    if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
        conda create -n "${CONDA_ENV}" python=3.10 -y
    fi
fi

conda activate "${CONDA_ENV}"

cd "${POLICY_DIR}"
pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
pip install -r "${POLICY_DIR}/requirements.txt"
# Only lerobot.datasets.lerobot_dataset is used. Installing LeRobot's complete
# dependency set would pull rerun-sdk (NumPy>=2) and unrelated robot/UI stacks,
# conflicting with the released X-VLA NumPy 1.26 environment.
pip install --no-deps "lerobot==0.4.4"

python -c '
import importlib.metadata as metadata
import torch
import torchvision
import av
import accelerate
import datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from peft import LoraConfig
from transformers import AutoConfig
from xvla.models.modeling_xvla import XVLA
from xvla.models.processing_xvla import XVLAProcessor

versions = {
    name: metadata.version(name)
    for name in ("torch", "torchvision", "transformers", "peft",
                 "accelerate", "av", "datasets", "lerobot")
}
print("[xvla_lerobot] dependency versions:", versions)
print("[xvla_lerobot] torch CUDA runtime:", torch.version.cuda)
print("[xvla_lerobot] CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[xvla_lerobot] GPU:", torch.cuda.get_device_name(0))
'

echo "[xvla_lerobot] Installed shared X-VLA runtime in conda env ${CONDA_ENV}."
echo "[xvla_lerobot] Model directory: ${POLICY_DIR}/checkpoints/shared/X-VLA-RoboTwin2"
