#!/usr/bin/env bash
set -euo pipefail

# Start only the codex_agent policy server on a GPU machine.
#
# Usage:
#   bash serve_remote.sh [task] [gpu] [port] [host] [conda_env]
#
# There is no checkpoint argument: the policy is a stateful Codex thread, and
# the thing this script cannot provide is the Codex bridge -- that runs on the
# operator's Mac. Keep the tunnel up before starting this:
#
#   tools/tunnel_mac.sh                       # on the Mac
#   export CODEX_BRIDGE_URL=http://localhost:8765   # only if not using deploy.yml
#
# Without a reachable bridge the server still starts and the client still runs,
# but every decision degrades to a hold and the episode scores zero, so verify
# with `curl -s localhost:8765/healthz` first.

if (( $# > 5 )); then
    echo "Usage: $0 [task] [gpu] [port] [host] [conda_env]" >&2
    exit 2
fi

task_name=${1:-plug_in_charger}
gpu_id=${2:-0}
port=${3:-6000}
host=${4:-0.0.0.0}
conda_env=${5:-XVLA}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[codex_agent SERVER] task=${task_name}, endpoint=ws://${host}:${port}, gpu=${gpu_id}"
echo "[codex_agent SERVER] bridge=${CODEX_BRIDGE_URL:-<deploy.yml bridge_url>}"

exec bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
    RoboDojo \
    "${task_name}" \
    none \
    arx_x5 \
    ee \
    0 \
    "${gpu_id}" \
    "${conda_env}" \
    "${port}" \
    "${host}"
