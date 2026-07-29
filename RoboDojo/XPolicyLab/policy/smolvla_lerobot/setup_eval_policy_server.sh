#!/bin/bash
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
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
yaml_file="${SCRIPT_DIR}/deploy.yml"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

resolve_checkpoint() {
    local raw=$1
    if [[ "${raw}" == /* ]]; then
        printf '%s\n' "${raw}"
    elif [[ -e "${raw}" ]]; then
        realpath "${raw}"
    elif [[ -e "${BENCH_ROOT}/${raw}" ]]; then
        realpath "${BENCH_ROOT}/${raw}"
    elif [[ -e "$(dirname "${BENCH_ROOT}")/${raw}" ]]; then
        realpath "$(dirname "${BENCH_ROOT}")/${raw}"
    else
        printf '%s\n' "/data/checkpoints/smolvla/${raw}"
    fi
}

checkpoint_path="$(resolve_checkpoint "${ckpt_name}")"
if [[ ! -e "${checkpoint_path}" ]]; then
    echo "[SERVER][ERROR] checkpoint does not exist: ${checkpoint_path}" >&2
    exit 1
fi

echo "[SERVER] policy=smolvla_lerobot task=${task_name}"
echo "[SERVER] checkpoint=${checkpoint_path}"
echo "[SERVER] bind=${policy_server_host}:${policy_server_port}"

exec env \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            checkpoint_path="${checkpoint_path}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="smolvla_lerobot" \
            action_type="${action_type}"
