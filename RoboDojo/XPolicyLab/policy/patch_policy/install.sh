# patch_policy: 复用服务器既有 conda 环境（issac-server 的 XVLA），不安装任何新包。
# 本脚本只校验依赖是否齐全，避免在服务器虚拟环境里执行安装命令（见远程服务器操作规范）。
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${PATCH_POLICY_CONDA_ENV:-XVLA}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "[patch_policy] ERROR: conda env '${CONDA_ENV}' not found." >&2
    exit 1
fi

conda activate "${CONDA_ENV}"
cd "${POLICY_DIR}"

python - <<'PY'
import importlib

mods = ["torch", "torchvision", "einops", "numpy", "scipy", "PIL", "yaml", "cv2"]
ok = True
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:12s} {getattr(mod, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  {name:12s} MISSING ({exc})")
if not ok:
    print("[patch_policy] ERROR: missing dependencies in conda env.")
    raise SystemExit(1)
import torch
print(f"  cuda available : {torch.cuda.is_available()}")
print("[patch_policy] deps OK")
PY

echo "[patch_policy] Done. use conda env ${CONDA_ENV}"
