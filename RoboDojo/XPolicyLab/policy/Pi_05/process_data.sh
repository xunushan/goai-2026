#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num_or_raw_task_dirs=${5:-}
raw_task_dirs=${6:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${OPENPI_DATA_MODE:-video}"

# lerobot 0.6.0 venv（/data/venvs/lerobot060）的 torchcodec 依赖 FFmpeg 6 库
# （libavutil.so.60 等），该目录不在默认 dlopen 路径，需显式加入动态链接路径，
# 否则 import torchcodec 报 libavutil.so.56 cannot open shared object file。
# 路径不存在时 prepend 空目录无害。
export LD_LIBRARY_PATH=/data/venvs/lerobot060/lib:${LD_LIBRARY_PATH:-}

# Python 3.11+ argparse：所有 positional 须在 option 之前，否则 nargs=? 位置参数不解析
py_args=(
  "${bench_name}"
  "${ckpt_name}"
  "${env_cfg_type}"
  "${action_type}"
)
if [[ -n "${expert_data_num_or_raw_task_dirs}" ]]; then
  py_args+=("${expert_data_num_or_raw_task_dirs}")
fi
if [[ -n "${raw_task_dirs}" ]]; then
  py_args+=("${raw_task_dirs}")
fi
py_args+=(--mode "${mode}")
if [[ -n "${OPENPI_REPO_ID:-}" ]]; then
  py_args+=(--repo_id "${OPENPI_REPO_ID}")
fi

cd "${POLICY_DIR}/openpi"
python scripts/process_data.py "${py_args[@]}"
