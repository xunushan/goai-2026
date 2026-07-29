#!/usr/bin/env bash
set -euo pipefail

# Start only the X-VLA policy server on a standalone GPU machine.
#
# Usage:
#   bash serve_remote.sh <checkpoint_dir> <processor_dir> [task] [gpu] [port] [host] [conda_env]

if (( $# < 2 || $# > 7 )); then
    echo "Usage: $0 <checkpoint_dir> <processor_dir> [task] [gpu] [port] [host] [conda_env]" >&2
    exit 2
fi

checkpoint_dir=$1
processor_dir=$2
task_name=${3:-stack_blocks}
gpu_id=${4:-0}
port=${5:-6000}
host=${6:-0.0.0.0}
conda_env=${7:-XVLA}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for required_file in config.json model.safetensors; do
    if [[ ! -f "${checkpoint_dir}/${required_file}" ]]; then
        echo "[X-VLA SERVER][ERROR] missing ${required_file} under ${checkpoint_dir}" >&2
        exit 1
    fi
done

if [[ ! -f "${processor_dir}/preprocessor_config.json" ]]; then
    echo "[X-VLA SERVER][ERROR] missing preprocessor_config.json under ${processor_dir}" >&2
    exit 1
fi

checkpoint_dir="$(cd "${checkpoint_dir}" && pwd)"
processor_dir="$(cd "${processor_dir}" && pwd)"

echo "[X-VLA SERVER] checkpoint=${checkpoint_dir}"
echo "[X-VLA SERVER] processor=${processor_dir}"
echo "[X-VLA SERVER] task=${task_name}, endpoint=ws://${host}:${port}, gpu=${gpu_id}"

XVLA_MODEL_PATH="${checkpoint_dir}" \
XVLA_PROCESSOR_PATH="${processor_dir}" \
    exec bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
        RoboDojo \
        "${task_name}" \
        "${checkpoint_dir}" \
        arx_x5 \
        ee \
        0 \
        "${gpu_id}" \
        "${conda_env}" \
        "${port}" \
        "${host}"
