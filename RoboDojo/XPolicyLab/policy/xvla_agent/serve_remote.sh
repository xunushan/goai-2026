#!/usr/bin/env bash
set -euo pipefail

# Start only the X-VLA policy server on a standalone GPU machine.
#
# Usage:
#   bash serve_remote.sh <checkpoint_dir> [task] [gpu] [port] [host] [conda_env]
#
# The checkpoint directory must contain the model files (config.json,
# model.safetensors) and the co-located processor files
# (preprocessor_config.json, tokenizer files); the processor is loaded from the
# same directory as the model, so no separate processor path is needed.

if (( $# < 1 || $# > 6 )); then
    echo "Usage: $0 <checkpoint_dir> [task] [gpu] [port] [host] [conda_env]" >&2
    exit 2
fi

checkpoint_dir=$1
task_name=${2:-stack_blocks}
gpu_id=${3:-0}
port=${4:-6000}
host=${5:-0.0.0.0}
conda_env=${6:-XVLA}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for required_file in config.json model.safetensors preprocessor_config.json; do
    if [[ ! -f "${checkpoint_dir}/${required_file}" ]]; then
        echo "[X-VLA SERVER][ERROR] missing ${required_file} under ${checkpoint_dir}" >&2
        exit 1
    fi
done

checkpoint_dir="$(cd "${checkpoint_dir}" && pwd)"

echo "[X-VLA SERVER] checkpoint=${checkpoint_dir}"
echo "[X-VLA SERVER] task=${task_name}, endpoint=ws://${host}:${port}, gpu=${gpu_id}"

XVLA_MODEL_PATH="${checkpoint_dir}" \
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
