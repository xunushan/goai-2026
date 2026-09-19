# XPolicyLab deploy: policy server env=XVLA; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XVLA_ROOT="${POLICY_DIR}/xvla"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${XVLA_CONDA_ENV:-XVLA}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${XVLA_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.10 -y
  fi
fi

conda activate "${CONDA_ENV}"

# 顺序和钉版都是必需的，做法对齐 X-VLA/scripts/install_env.sh：
#   - requirements.txt 未钉 torch，peft/timm 会把它从默认 PyPI 拉成最新 CUDA 打包版
#     （实测 torch 2.14.0+cu130 / torchvision 0.29.0）；装完再走 cu128 索引也不会降级，
#     因为 torch 已满足该要求，只留下 torchaudio 与 torch 大版本错配。故先装钉定版本。
#   - XPolicyLab 的 opencv-python-headless>=4.8 会解析到 5.x，而 5.x 声明 numpy>=2，
#     会把 requirements.txt 钉住的 1.26.3 顶成 2.2.6。最后把科学计算栈钉回来。
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

cd "${XVLA_ROOT}"
pip install -r requirements.txt

cd "${XPOLICYLAB_ROOT}"
pip install -e .
pip install "numpy==1.26.3" "opencv-python-headless<5"

echo "[X_VLA_OPT] Done. conda activate ${CONDA_ENV}"
