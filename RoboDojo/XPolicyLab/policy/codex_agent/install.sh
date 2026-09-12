#!/usr/bin/env bash
# codex_agent: validate, install nothing.
#
# The adapter has no torch model, no checkpoint and no CUDA work: it is stdlib
# plus numpy/scipy/PIL for pose maths and image encoding, and it reaches Codex
# over HTTP. The only heavier requirement is the websocket policy server, which
# already lives in the shared XVLA conda environment that XPolicyLab is
# installed into.
#
# So this script checks and exits. It deliberately runs no installer: the remote
# server's virtualenvs are not ours to mutate (see the project's remote-server
# rules on pip/conda installs).
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CODEX_AGENT_CONDA_ENV:-XVLA}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "[codex_agent] ERROR: conda env '${CONDA_ENV}' not found." >&2
    exit 1
fi

conda activate "${CONDA_ENV}"
cd "${POLICY_DIR}"

# setup_policy_server.py imports the model as `XPolicyLab.policy.<name>.model`
# with XPolicyLab's parent directory on sys.path, so that is how we check it.
export CODEX_AGENT_REPO_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"

python - <<'PY'
import importlib
import os
import sys

# Importable by the policy server: pose maths, image encoding, then the
# transport it will actually accept connections on.
mods = ["numpy", "scipy", "PIL", "yaml", "websockets"]
ok = True
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:12s} {getattr(mod, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  {name:12s} MISSING ({exc})")

# XPolicyLab must be importable under the name setup_policy_server.py uses.
sys.path.insert(0, os.environ["CODEX_AGENT_REPO_ROOT"])
try:
    importlib.import_module("XPolicyLab.policy.codex_agent.model")
    print("  codex_agent  importable as XPolicyLab.policy.codex_agent.model")
except Exception as exc:  # noqa: BLE001
    ok = False
    print(f"  codex_agent  NOT IMPORTABLE ({exc})")

if not ok:
    print("[codex_agent] ERROR: missing dependencies in conda env.")
    print("[codex_agent] Fix the environment by hand; this script installs nothing.")
    raise SystemExit(1)
print("[codex_agent] deps OK")
PY

# The bridge is not part of this environment -- it runs on the operator's Mac --
# but a run is wasted if the tunnel is not up by the time the first decision is
# taken, so report what the server would dial.
bridge_url="${CODEX_BRIDGE_URL:-}"
if [[ -n "${bridge_url}" ]]; then
    echo "[codex_agent] CODEX_BRIDGE_URL=${bridge_url}"
else
    echo "[codex_agent] CODEX_BRIDGE_URL unset; deploy.yml's bridge_url will be used."
    echo "[codex_agent]   From the Mac, keep a tunnel up: tools/tunnel_mac.sh"
fi

echo "[codex_agent] Done. use conda env ${CONDA_ENV}"
