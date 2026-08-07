#!/bin/bash
# xvla_2 policy-server 启动脚本（robodojo.sh server 调用）。
# 模型来源优先级：本地 checkpoint 目录（已提前下载）> HF repo id。
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
policy_server_host=${10:-"0.0.0.0"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
yaml_file="${SCRIPT_DIR}/deploy.yml"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

# 解析模型来源：绝对路径 / 相对已存在路径 / 服务器标准布局 /data/checkpoints /
# 都不是 → 视为 HF repo id（owner/repo/checkpoint，透传给 from_pretrained）。
resolve_model() {
    local raw=$1
    if [[ "${raw}" == /* ]]; then
        printf '%s\n' "${raw}"
    elif [[ -e "${raw}" ]]; then
        realpath "${raw}"
    elif [[ -e "${BENCH_ROOT}/${raw}" ]]; then
        realpath "${BENCH_ROOT}/${raw}"
    elif [[ -e "$(dirname "${BENCH_ROOT}")/${raw}" ]]; then
        realpath "$(dirname "${BENCH_ROOT}")/${raw}"
    elif [[ -e "/data/checkpoints/${raw}" ]]; then
        realpath "/data/checkpoints/${raw}"
    elif [[ "${raw}" == */* ]]; then
        printf '%s\n' "${raw}"   # HF repo id
    else
        printf '%s\n' "${SCRIPT_DIR}/checkpoints/${raw}"
    fi
}

model="$(resolve_model "${ckpt_name}")"
echo "[SERVER] policy=xvla_2 task=${task_name}"
echo "[SERVER] model=${model}"
echo "[SERVER] bind=${policy_server_host}:${policy_server_port}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
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
            model="${model}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="xvla_2" \
            action_type="${action_type}"
