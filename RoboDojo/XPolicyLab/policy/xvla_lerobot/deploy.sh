#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <gpu_id> <policy_conda_env> <checkpoint_path> [port] [device]" >&2
    exit 1
fi

gpu_id=$1
policy_conda_env=$2
checkpoint_path=$3
port=${4:-6000}
device=${5:-cuda}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

if [[ ! -f "${checkpoint_path}/config.json" ]]; then
    echo "Checkpoint config not found: ${checkpoint_path}/config.json" >&2
    exit 2
fi
checkpoint_path="$(realpath "${checkpoint_path}")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

echo "[SERVER] policy=xvla_lerobot checkpoint=${checkpoint_path}"
echo "[SERVER] bind=localhost:${port} device=${device}"

exec env \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${SCRIPT_DIR}/deploy.yml" \
        --overrides \
            port="${port}" \
            policy_name=xvla_lerobot \
            checkpoint_path="${checkpoint_path}" \
            device="${device}"
