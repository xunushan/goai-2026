#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

resolve_checkpoint() {
    local raw=$1
    if [[ "${raw}" == /* && -e "${raw}" ]]; then
        realpath "${raw}"
    elif [[ -e "${raw}" ]]; then
        realpath "${raw}"
    elif [[ -e "${SCRIPT_DIR}/checkpoints/${raw}" ]]; then
        realpath "${SCRIPT_DIR}/checkpoints/${raw}"
    else
        printf '%s\n' "${raw}"
    fi
}

checkpoint_path="$(resolve_checkpoint "${ckpt_name}")"
if [[ ! -f "${checkpoint_path}/config.json" ]]; then
    echo "[SERVER][ERROR] X-VLA checkpoint config not found: ${checkpoint_path}/config.json" >&2
    exit 1
fi
if [[ "${action_type}" != "ee" ]]; then
    echo "[SERVER][ERROR] xvla_lerobot supports only action_type=ee" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

echo "[SERVER] policy=xvla_lerobot task=${task_name}"
echo "[SERVER] checkpoint=${checkpoint_path}"
echo "[SERVER] bind=${policy_server_host}:${policy_server_port} domain_id=6 control_hz=25"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${SCRIPT_DIR}/deploy.yml" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            checkpoint_path="${checkpoint_path}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name=xvla_lerobot \
            action_type=ee
