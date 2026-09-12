#!/bin/bash
# codex_agent policy server.
#
# Same argument contract as every other XPolicyLab policy (robodojo.sh's
# run_policy_eval.sh calls this directly), but almost nothing is used: there is
# no checkpoint and no model weights. The one thing that must be forwarded is
# the Codex bridge endpoint, because the bridge runs on the operator's Mac and
# reaches this server over an `ssh -R` tunnel -- not on localhost:8765 of the
# GPU machine.
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
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] ckpt_name=${ckpt_name} is ignored (no checkpoint; Codex is the policy)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

overrides=(
    port="${policy_server_port}"
    host="${policy_server_host}"
    bench_name="${bench_name}"
    task_name="${task_name}"
    ckpt_name="${ckpt_name}"
    env_cfg_type="${env_cfg_type}"
    seed="${seed}"
    policy_name="${policy_name}"
    action_type="${action_type}"
    action_dim="${action_dim}"
)

# Where the Codex bridge lives, as seen *from this machine*. With the tunnel up
# (ssh -N -R 8765:localhost:8765 <gpu-host>) the default is already correct; set
# CODEX_BRIDGE_URL to point at a different host/port without editing deploy.yml.
if [[ -n "${CODEX_BRIDGE_URL:-}" ]]; then
    echo "[SERVER] bridge_url=${CODEX_BRIDGE_URL} (from CODEX_BRIDGE_URL)"
    overrides+=(bridge_url="${CODEX_BRIDGE_URL}")
else
    echo "[SERVER] bridge_url=<from deploy.yml> (CODEX_BRIDGE_URL not set)"
fi

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides "${overrides[@]}"
