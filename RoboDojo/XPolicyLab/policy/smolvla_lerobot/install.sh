#!/bin/bash
set -euo pipefail

# Reuse the existing environment; do not create or replace a Conda env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

python -m pip install --upgrade "lerobot[smolvla]>=0.6,<0.7"

python - <<'PY'
import lerobot
import transformers
from lerobot.policies.factory import get_policy_class

version = getattr(lerobot, "__version__", "unknown")
if not version.startswith("0.6."):
    raise RuntimeError(f"Expected LeRobot 0.6.x, got {version}")

print(f"[smolvla_lerobot] lerobot={version}")
print(f"[smolvla_lerobot] transformers={transformers.__version__}")
print(f"[smolvla_lerobot] policy={get_policy_class('smolvla')}")
PY
