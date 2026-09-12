#!/bin/bash
# codex_agent eval driver: policy server + simulator client on one machine.
#
# NOTE: `robodojo.sh eval` does not run this script -- it calls
# scripts/internal/run_policy_eval.sh with the same 10 positional arguments.
# This file exists for the XPolicyLab-side entry point and to be grepped by
# robodojo.sh, which inspects eval.sh to decide the argument list it will build.
# Keep the argument count at 10, and do NOT spell out the dataset-size keyword
# that some policies take first (grep it out of this file and you will see why
# it is missing): robodojo.sh greps this script for it to decide whether to
# prepend an extra argument, and codex_agent takes no dataset argument, so the
# 7-trailing-argument form of run_policy_eval.sh is the one that must be chosen.
set -euo pipefail
bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_conda_env=$9
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # Current Dir
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

# ckpt_name is inert here (Codex is the policy, there is no checkpoint) but the
# eval client records it in its result metadata, so pass it through unchanged.
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

echo "[MAIN] ckpt_name=${ckpt_name} is ignored by codex_agent (no checkpoint)"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start server, policy_server_port=${policy_server_port}"

bash "${SERVER_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}" &

SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 600

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"

bash "${CLIENT_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${eval_env_conda_env}" \
    "${additional_info}" \
    "${policy_server_port}" \
    "${policy_server_ip}"

echo "[MAIN] eval finished"
