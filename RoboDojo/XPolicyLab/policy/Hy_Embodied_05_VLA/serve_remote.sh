#!/usr/bin/env bash
set -euo pipefail

# Start only the Hy-VLA policy server. This entry point is intended for a
# policy GPU machine that does not run Isaac Sim.
#
# Usage:
#   bash serve_remote.sh <checkpoint_dir> [task_name] [gpu_id] [port] [host] [policy_uv_root_or_uv]
#
# checkpoint_dir must contain config.json, model.safetensors and norm_stats.pkl.

if (( $# < 1 || $# > 6 )); then
    echo "Usage: $0 <checkpoint_dir> [task_name] [gpu_id] [port] [host] [policy_uv_root_or_uv]" >&2
    exit 2
fi

checkpoint_dir=$1
task_name=${2:-stack_blocks}
gpu_id=${3:-0}
port=${4:-6000}
host=${5:-0.0.0.0}
policy_uv_env=${6:-uv}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "[HY-VLA SERVER][ERROR] checkpoint directory not found: ${checkpoint_dir}" >&2
    exit 1
fi

for required_file in config.json model.safetensors norm_stats.pkl; do
    if [[ ! -f "${checkpoint_dir}/${required_file}" ]]; then
        echo "[HY-VLA SERVER][ERROR] missing ${required_file} under ${checkpoint_dir}" >&2
        exit 1
    fi
done

checkpoint_dir="$(cd "${checkpoint_dir}" && pwd)"

echo "[HY-VLA SERVER] checkpoint=${checkpoint_dir}"
echo "[HY-VLA SERVER] task=${task_name}, endpoint=ws://${host}:${port}, gpu=${gpu_id}"

HY_VLA_CKPT_PATH="${checkpoint_dir}" \
    exec bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
        RoboDojo \
        "${task_name}" \
        "${checkpoint_dir}" \
        arx_x5 \
        ee \
        0 \
        "${gpu_id}" \
        "${policy_uv_env}" \
        "${port}" \
        "${host}"
