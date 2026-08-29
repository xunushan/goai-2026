# XPolicyLab deploy: policy server env=XVLA; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
# 用法: bash install.sh
# 创建默认 conda 环境 XVLA 并安装 X-VLA 依赖（首次使用运行一次）。
#
# 可选环境变量:
#   XVLA_CONDA_ENV=...      环境名（默认 XVLA）
#   XVLA_SKIP_CONDA_CREATE=1  跳过创建环境（复用已有环境）
#   TORCH_INDEX_URL=...     PyTorch 下载源（默认 cu128，CUDA 12.8；其他版本 CUDA 请自行覆盖）
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XVLA_ROOT="${POLICY_DIR}/xvla"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${XVLA_CONDA_ENV:-XVLA}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# ---- conda 探测（兼容 conda 不在 PATH 的情况）----
if ! command -v conda >/dev/null 2>&1; then
  for c in /data/miniconda3 /opt/miniconda3 /opt/anaconda3 "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [[ -x "${c}/bin/conda" ]]; then
      export PATH="${c}/bin:${PATH}"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] 未找到 conda，请先安装 miniconda/anaconda" >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${XVLA_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.10 -y
  fi
fi

conda activate "${CONDA_ENV}"

cd "${XVLA_ROOT}"
pip install -r requirements.txt

# torch：环境中已有可用 torch 则跳过，避免重复下载
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "[INFO] 安装 PyTorch（${TORCH_INDEX_URL}）..."
  pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
else
  echo "[INFO] 已存在 torch，跳过安装"
fi

cd "${XPOLICYLAB_ROOT}"
pip install -e .

echo "[X_VLA] Done. conda activate ${CONDA_ENV}"
