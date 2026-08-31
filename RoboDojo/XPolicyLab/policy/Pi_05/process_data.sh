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
