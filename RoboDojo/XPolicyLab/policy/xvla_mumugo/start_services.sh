#!/usr/bin/env bash
# xvla_mumugo —— X-VLA 策略服务「下载模型 + 启动服务」一键脚本
#
# 用法:
#   bash start_services.sh xvla-fw                       # 最简，默认端口 6000
#   bash start_services.sh xvla-fw --port 8080 --gpu 1 --host 0.0.0.0 --policy-env myenv
#
# 模型自动下载到本脚本所在目录 checkpoints/ 下；日志输出到本脚本所在目录 logs/ 下。
# 默认后台运行；默认监听 127.0.0.1（本地评测），对外暴露请用 --host 0.0.0.0。
set -euo pipefail
export PS1='$ '

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_NAME="$(basename "${SCRIPT_DIR}")"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---------- 解析参数 ----------
MODEL=""
PORT=""
GPU="0"
HOST=""
POLICY_ENV=""

usage() {
  echo "用法: bash start_services.sh <model> [--port N] [--gpu N] [--host IP] [--policy-env ENV]" >&2
  echo "  model: xvla-fw | xvla-sf" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)       PORT="$2"; shift 2 ;;
    --gpu)        GPU="$2"; shift 2 ;;
    --host)       HOST="$2"; shift 2 ;;
    --policy-env) POLICY_ENV="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    -*)
      echo "[ERROR] 未知参数: $1（仅支持 --port --gpu --host --policy-env）" >&2
      usage; exit 2 ;;
    *)
      if [[ -z "${MODEL}" ]]; then
        MODEL="$1"; shift
      else
        echo "[ERROR] 多余的参数: $1（model 只能指定一次）" >&2
        usage; exit 2
      fi ;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "[ERROR] 缺少必传参数 model" >&2
  usage; exit 2
fi

# ---------- 模型 → 权重目录 ----------
case "${MODEL}" in
  xvla-fw) HF_INCLUDE="T-formal-12000/ckpt-12000/*" ;;
  xvla-sf) HF_INCLUDE="A2/ckpt-2000/*" ;;
  *)
    echo "[ERROR] 未知模型 ${MODEL}，可选: xvla-fw | xvla-sf" >&2
    exit 2 ;;
esac

# ---------- 固定内部参数（不对外暴露） ----------
HF_REPO="tianSeconds/finetunning"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
CKPT_PATH="${CKPT_DIR}/${HF_INCLUDE%/\*}"     # checkpoints/T-formal-12000/ckpt-12000
LOG_DIR="${SCRIPT_DIR}/logs"

# ---------- 默认值 ----------
PORT="${PORT:-6000}"
HOST="${HOST:-127.0.0.1}"
POLICY_ENV="${POLICY_ENV:-${CONDA_DEFAULT_ENV:-XVLA}}"

# ---------- 环境定位 ----------
if [[ ! -f "${ROOT_DIR}/scripts/robodojo.sh" ]]; then
  echo "[ERROR] 未找到 ${ROOT_DIR}/scripts/robodojo.sh" >&2
  echo "       请确认本文件夹位于 RoboDojo/XPolicyLab/policy/${POLICY_NAME} 下" >&2
  exit 1
fi

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

if ! conda env list | awk '{print $1}' | grep -qx "${POLICY_ENV}"; then
  echo "[ERROR] conda 环境 ${POLICY_ENV} 不存在" >&2
  echo "       已安装过 X_VLA 环境请用 --policy-env <env> 指定；否则先运行: bash ${SCRIPT_DIR}/install.sh" >&2
  exit 1
fi
conda activate "${POLICY_ENV}"

# ---------- 依赖检查（缺什么补什么） ----------
if ! python -c "import websockets, msgpack, msgpack_numpy" >/dev/null 2>&1; then
  echo "[INFO] 在 ${POLICY_ENV} 中补装缺失依赖 (websockets/msgpack/msgpack-numpy) ..."
  pip install -q websockets msgpack msgpack-numpy
fi

# ---------- 模型下载（已存在则增量复用） ----------
# HF 下载走国内镜像、禁用 xet 后端（国内服务器直连 HF 常失败/超时）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if ! command -v hf >/dev/null 2>&1; then
  echo "[INFO] 安装 huggingface_hub (hf 命令) ..."
  pip install -q huggingface_hub
fi
echo "[INFO] 下载模型 ${MODEL} (${HF_INCLUDE}) -> ${CKPT_DIR}"
mkdir -p "${CKPT_DIR}"
hf download "${HF_REPO}" --include "${HF_INCLUDE}" --local-dir "${CKPT_DIR}"

for f in config.json model.safetensors preprocessor_config.json; do
  [[ -f "${CKPT_PATH}/${f}" ]] || {
    echo "[ERROR] ${CKPT_PATH} 缺少 ${f}，模型未下载完整" >&2
    exit 1
  }
done

# ---------- 启动服务 ----------
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/xvla_mumugo_${MODEL}_${TS}.log"

SERVER_ARGS=(
  server
  --policy-dir "XPolicyLab/policy/${POLICY_NAME}"
  --task stack_blocks
  --ckpt "${CKPT_PATH}"
  --policy-env "${POLICY_ENV}"
  --env-cfg arx_x5
  --action-type ee
  --seed 0
  --policy-gpu "${GPU}"
  --policy-port "${PORT}"
  --bind-host "${HOST}"
)

echo "[INFO] model=${MODEL}"
echo "[INFO] ckpt=${CKPT_PATH}"
echo "[INFO] endpoint=ws://${HOST}:${PORT}  gpu=${GPU}  env=${POLICY_ENV}"
echo "[INFO] 日志: ${LOG}"
echo "[INFO] 即将执行命令:"
echo "  cd ${ROOT_DIR} && bash scripts/robodojo.sh ${SERVER_ARGS[*]}"

cd "${ROOT_DIR}"
nohup bash "${ROOT_DIR}/scripts/robodojo.sh" "${SERVER_ARGS[@]}" > "${LOG}" 2>&1 &
SRV_PID=$!
echo "[INFO] 已后台启动 PID=${SRV_PID}"

# ---------- 等待服务就绪（模型加载需数分钟） ----------
echo "[INFO] 等待服务就绪，请耐心等待（模型加载约需数分钟）..."
READY=0
for i in $(seq 1 90); do   # 每 10s 探测一次，最多 15 分钟
  sleep 10
  if ! kill -0 "${SRV_PID}" 2>/dev/null; then
    echo "[ERROR] 服务进程已退出，启动失败。日志最后 30 行:" >&2
    tail -30 "${LOG}" >&2
    exit 1
  fi
  if python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', ${PORT})); s.close()" 2>/dev/null; then
    READY=1
    break
  fi
done

if [[ "${READY}" == "1" ]]; then
  echo "[INFO] 服务已就绪：ws://${HOST}:${PORT}"
  echo "[INFO] 可以开始仿真测试（仿真 client 的 host/port/模型名须与本服务一致）"
else
  echo "[ERROR] 等待 ${PORT} 端口就绪超时（15 分钟），请检查日志: ${LOG}" >&2
  tail -30 "${LOG}" >&2
  exit 1
fi
