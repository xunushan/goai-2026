# xvla_2 policy-server 环境安装脚本（只在策略服务器端运行）。
#
# 用法：
#   bash install.sh [XVLA_REPO_PATH]
#     XVLA_REPO_PATH：本地已克隆的 X-VLA 仓库（默认 ${POLICY_DIR}/X-VLA）。
#     若未提供且默认路径不存在，脚本提示如何克隆（私有仓库需要 git 凭据/token，
#     见 README）。
#
# 可选环境变量：
#   XVLA_CONDA_ENV      conda 环境名（默认 XVLA，policy-server 上已有的环境）
#   XVLA_SKIP_CONDA_CREATE=1   跳过创建 conda 环境
#   XVLA_SKIP_TORCH=1          跳过 torch/torchvision 安装（环境已有）
# 完成后在 policy env 里 import 冒烟。
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${XVLA_CONDA_ENV:-XVLA}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${XVLA_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.10 -y
  fi
fi
conda activate "${CONDA_ENV}"

if [[ "${XVLA_SKIP_TORCH:-0}" != "1" ]]; then
  # 环境已有 torch/torchvision 时跳过（避免把已配好 CUDA 的 env 降级成 2.1）。
  # 全新环境才装与训练 environment.yml 一致的 pytorch 2.1 / torchvision 0.16 /
  # CUDA 12.1（xvla 包的 pyproject.toml 刻意不声明 torch，由 conda/pip 在此提供）。
  if python -c "import torch, torchvision" 2>/dev/null; then
    echo "[xvla_2] torch/torchvision already installed, skip"
  else
    pip install --no-input \
      "torch==2.1.2" "torchvision==0.16.2" \
      --index-url https://download.pytorch.org/whl/cu121
  fi
fi

# X-VLA 包：本地克隆路径优先（pip install -e，避免每次重新下载）。
XVLA_REPO="${1:-${POLICY_DIR}/X-VLA}"
if [[ ! -d "${XVLA_REPO}" ]]; then
  echo "[xvla_2][ERROR] X-VLA 本地仓库不存在：${XVLA_REPO}"
  echo "[xvla_2] 请先克隆后重跑，例如："
  echo "[xvla_2]   git clone https://<token>@github.com/xunushan/X-VLA.git ${XVLA_REPO}"
  echo "[xvla_2] （token 见 ~/Documents/token/github；脚本不写明文）"
  echo "[xvla_2] 或改用 pip 从私有仓库安装并设置好 git 凭据："
  echo "[xvla_2]   pip install 'git+https://github.com/xunushan/X-VLA.git'"
  exit 1
fi
pip install -e "${XVLA_REPO}"

# XPolicyLab ws 协议依赖 + 推理/日志解析依赖。
pip install websockets msgpack msgpack-numpy pydantic pyyaml opencv-python

# import 冒烟（模型真正加载在服务启动时，见 setup_eval_policy_server.sh）。
python - <<'PY'
from evaluation.robodojo import RoboDojoPolicyClient, parse_policy_log
from xvla_datasets.utils import ee16_to_xvla20, xvla20_to_ee16
import evaluation.robodojo.parse_log as _pl
print("[xvla_2] package import OK")
PY

echo "[xvla_2] Done. conda activate ${CONDA_ENV}"
