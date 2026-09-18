#!/bin/bash
# Launch exactly one codex_agent episode on the GPU machine.
#
#   bash tools/launch_eval.sh <task_name> [log_name]
#
# Runs ON THE SERVER, from a copy at /data/outputs/launch_eval.sh. It lives in the
# repo as well as on the server on purpose: the version that only existed on the
# server had `--task plug_in_charger` baked into it, so passing a different task
# name silently launched the old one -- the config in deploy.yml and the task that
# actually ran could disagree with nothing to catch it. Reading the task from $1
# and echoing it before the eval starts is the whole point.
#
# Three launch traps this script encodes:
#   * ~/.bashrc early-returns for non-interactive shells, so `conda` is not on
#     PATH inside a screen -- source conda.sh directly instead.
#   * eval_client/utils/setup_env_client.sh calls a bare `python`, and only the
#     policy server activates its own env, so the sim client needs RoboDojo
#     already active before robodojo.sh is invoked.
#   * a degraded bridge is not a soft failure: the adapter probes healthz once at
#     turn 0 and, if it fails twice, holds for the whole episode without ever
#     calling Codex. Retry before starting rather than discovering it 400
#     hold-steps later.
set -u

TASK="${1:-}"
if [ -z "$TASK" ]; then
    echo "usage: bash tools/launch_eval.sh <task_name>" >&2
    exit 64
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=/data/outputs/codex_agent_eval_${TASK}_${STAMP}.log
echo "$LOG" > /tmp/codex_eval_log_path
exec > "$LOG" 2>&1

echo "[launcher] start=$(date -Iseconds)  task=$TASK  log=$LOG"
. /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate RoboDojo
echo "[launcher] python=$(command -v python)  env=${CONDA_DEFAULT_ENV:-none}"

# The step budget the sim will actually enforce, against the one deploy.yml
# states. They disagree silently otherwise: the prompt quotes deploy.yml, the
# episode truncates on the task's step_lim, and the model plans against a number
# that is not real.
echo "[launcher] task step_lim: $(grep -m1 'step_lim' \
    /data/RoboDojo/task/RoboDojo/tasks/${TASK}.py 2>/dev/null || echo '(not found)')"

echo "[launcher] bridge healthz:"
ok=0
for i in 1 2 3 4 5; do
    body=$(curl -s --max-time 10 http://localhost:8765/healthz)
    if [ -n "$body" ]; then
        echo "[launcher] healthz try$i: $body"
        ok=1
        break
    fi
    echo "[launcher] healthz try$i: FAILED, retrying in 5s"
    sleep 5
done
if [ "$ok" != "1" ]; then
    echo "[launcher] ABORT: bridge unreachable; not starting the episode"
    exit 2
fi
echo

cd /data/RoboDojo || exit 1
echo "[launcher] cmd: robodojo.sh eval --policy-dir XPolicyLab/policy/agent_policy --task $TASK --ckpt none --policy-env XVLA --eval-num 1"
echo "[launcher] ===== eval output below ====="
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/agent_policy \
  --task "$TASK" \
  --ckpt none \
  --policy-env XVLA \
  --eval-num 1
rc=$?

echo "[launcher] exit=$rc"
echo "[launcher] end=$(date -Iseconds)"
exit $rc
