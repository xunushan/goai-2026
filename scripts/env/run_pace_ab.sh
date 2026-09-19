#!/usr/bin/env bash
# usage: run_pace_ab.sh <ckpt_base> <ckpt> <task> <seed> <eval_num> [views] [model_class]
#
# PACE 开/关 A/B：同一 ckpt、同一 task/seed/eval_num、同一 policy_seed，唯一变量是
# deploy.yml 的 pace.enabled。两臂各跑一次 run_model_eval.sh，结果 stage 到
# /tmp/eval_stage/<model>/ 下，互不覆盖正式评测的暂存目录。
#
#   臂 A  model_id=xvla_ab_paceoff  pace.enabled=false
#   臂 B  model_id=xvla_ab_paceon   pace.enabled=true
#
# 计时/调用量对比从各臂自己的策略服务日志（/data/outputs/<model>_<ckpt>_server_*.log）
# 里取 [x_vla][io] 的 server_actions 事件统计，不由本脚本汇总。
#
# deploy.yml 处理：开头整file备份，中途按臂改写并回读校验，结束时从备份整file复原。
# 脚本只跑 staging，不写 DB。
set -uo pipefail

CKPT_BASE="$1"
CKPT="$2"
TASK="$3"
SEED="$4"
EVAL_NUM_AB="$5"
VIEWS="${6:-1}"
MODEL_CLASS="${7:--}"

ROBO=/data/RoboDojo
DEPLOY="${ROBO}/XPolicyLab/policy/X_VLA_OPT/deploy.yml"
LOG_DIR=/data/outputs
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/pace_ab_${TASK}_s${SEED}_${TS}.log"
BAK="${DEPLOY}.bak_paceab_${TS}"

cp "${DEPLOY}" "${BAK}"
exec >> "${LOG}" 2>&1
echo "[pace_ab] start ${TS} log=${LOG}"
echo "[pace_ab] deploy.yml backup -> ${BAK}"

# ---------- 翻转 pace.enabled 并回读 ----------
set_pace () {
  local WANT="$1"
  python3 - "${WANT}" <<'PY' || exit 3
import re, sys
p = "/data/RoboDojo/XPolicyLab/policy/X_VLA_OPT/deploy.yml"
want = sys.argv[1]
assert want in ("true", "false"), want
s = open(p).read()
# pace: 块的第一个键必须是 enabled:，本文件如此；只改这一处，其余注释原样保留
pat = re.compile(r"^pace:\n(  enabled:\s*)(\S+)\s*$", flags=re.M)
m = pat.search(s)
assert m, "pace.enabled block not found"
s2 = pat.sub(lambda mm: f"pace:\n{mm.group(1)}{want}", s, count=1)
open(p, "w").write(s2)
chk = pat.search(open(p).read())
assert chk and chk.group(2) == want, f"readback mismatch: {chk.group(2) if chk else None}"
print(f"[pace_ab] deploy.yml pace.enabled -> {want}")
PY
}

run_arm () {
  local MODEL="$1" PACE="$2"
  set_pace "${PACE}"
  echo "[pace_ab] === arm ${MODEL} (pace.enabled=${PACE}) begin $(date +%Y%m%d_%H%M%S) ==="
  EVAL_NUM="${EVAL_NUM_AB}" bash "${LOG_DIR}/run_model_eval.sh" \
    "${MODEL}" "${CKPT_BASE}" "${CKPT}" "${VIEWS}" "${MODEL_CLASS}" "${TASK}:${SEED}"
  echo "[pace_ab] === arm ${MODEL} done rc=$? $(date +%Y%m%d_%H%M%S) ==="
}

run_arm xvla_ab_paceoff false
run_arm xvla_ab_paceon  true

# ---------- 复原 deploy.yml ----------
cp "${BAK}" "${DEPLOY}"
python3 - <<'PY'
import re
p = "/data/RoboDojo/XPolicyLab/policy/X_VLA_OPT/deploy.yml"
m = re.search(r"^pace:\n(  enabled:\s*)(\S+)\s*$", open(p).read(), flags=re.M)
print(f"[pace_ab] deploy.yml restored: pace.enabled={m.group(2) if m else 'MISSING'}")
PY
echo "[pace_ab] done $(date +%Y%m%d_%H%M%S)"
ls -la "/tmp/eval_stage/xvla_ab_paceoff" "/tmp/eval_stage/xvla_ab_paceon" 2>/dev/null
