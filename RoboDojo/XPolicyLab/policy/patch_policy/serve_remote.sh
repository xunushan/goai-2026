#!/usr/bin/env bash
set -euo pipefail

# Start only the patch_policy (flow_policy) policy server on a GPU machine.
#
# Usage:
#   bash serve_remote.sh <ckpt_path> [task] [gpu] [port] [host] [conda_env]
#
# The ckpt must be a single flow_policy checkpoint file (*.pt).

if (( $# < 1 || $# > 6 )); then
    echo "Usage: $0 <ckpt_path> [task] [gpu] [port] [host] [conda_env]" >&2
    exit 2
fi

ckpt_path=$1
task_name=${2:-stack_blocks}
gpu_id=${3:-0}
port=${4:-6000}
host=${5:-0.0.0.0}
conda_env=${6:-XVLA}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${ckpt_path}" ]]; then
    echo "[patch_policy SERVER][ERROR] checkpoint not found: ${ckpt_path}" >&2
    exit 1
fi

echo "[patch_policy SERVER] ckpt=${ckpt_path}"
echo "[patch_policy SERVER] task=${task_name}, endpoint=ws://${host}:${port}, gpu=${gpu_id}"

PATCH_POLICY_CKPT="${ckpt_path}" \
    exec bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
        RoboDojo \
        "${task_name}" \
        "$(basename "${ckpt_path}")" \
        arx_x5 \
        ee \
        0 \
        "${gpu_id}" \
        "${conda_env}" \
        "${port}" \
        "${host}"
