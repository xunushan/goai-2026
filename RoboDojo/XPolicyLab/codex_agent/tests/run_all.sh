#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
"${PYTHON}" tests/test_app_server.py
"${PYTHON}" tests/test_experience.py
"${PYTHON}" tests/test_workspace.py
"${PYTHON}" tests/test_vla_review.py
