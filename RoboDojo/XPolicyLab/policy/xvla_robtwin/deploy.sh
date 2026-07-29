#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <gpu_id> <policy_conda_env> <model_path> [processor_path] [port] [device]" >&2
    exit 1
fi

gpu_id=$1
policy_conda_env=$2
model_path=$3
processor_path=${4:-${model_path}}
port=${5:-6000}
device=${6:-cuda}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${SCRIPT_DIR}/deploy.yml" \
        --overrides \
            port="${port}" \
            policy_name=xvla_robtwin \
            model_path="${model_path}" \
            processor_path="${processor_path}" \
            device="${device}"
