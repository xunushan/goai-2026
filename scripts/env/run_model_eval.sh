#!/usr/bin/env bash
# usage: run_model_eval.sh <model> <ckpt_base> <ckpts_csv> <views 1|3> <model_class|-> <groups_csv>
#
#   model        模型名（= 服务器模型目录名；用于日志前缀 / 结果暂存目录 / DB model_id）
#   ckpt_base    该模型的 pretrained 目录（内含 ckpt-XXXX 子目录）
#   ckpts_csv    要评的 checkpoint，逗号分隔，如 ckpt-4000,ckpt-5000,ckpt-6000
#   views        摄像头路数 1 或 3
#   model_class  deploy.yml 的 policy_model_class；'-' 表示 null（基础 XVLA）
#   groups_csv   要评的组，逗号分隔 task:seed，如 stack_bowls:0
#
# 流程：快照 deploy.yml -> 按 views/model_class 改写并回读校验 -> 每个 ckpt 起一次
# 策略服务（screen）-> 逐个 (ckpt,task,seed) 跑 eval-num 8 的 smoke -> 结果 stage 到
# /tmp/eval_stage/<model>/<ckpt>_<task>_s<seed>.json。幂等：stage 文件已存在则跳过。
# 仅 staging，不写 DB（入库由智能体按 checkpoint-sim-eval 铁律执行）。
set -uo pipefail

MODEL="$1"
CKPT_BASE="$2"
CKPTS_CSV="$3"
VIEWS="$4"
MODEL_CLASS="$5"
GROUPS_CSV="$6"

CONDA=/home/ubuntu/miniconda3/etc/profile.d/conda.sh
ROBO=/data/RoboDojo
DEPLOY="${ROBO}/XPolicyLab/policy/X_VLA/deploy.yml"
LOG_DIR=/data/outputs
STAGE="/tmp/eval_stage/${MODEL}"
PORT=6000
EVAL_NUM=8

mkdir -p "${STAGE}" "${LOG_DIR}"
SELFLOG="${LOG_DIR}/${MODEL}_eval_run_$(date +%Y%m%d_%H%M%S).log"
exec > "${SELFLOG}" 2>&1
echo "[driver] selflog=${SELFLOG}"
echo "[driver] model=${MODEL} base=${CKPT_BASE} views=${VIEWS} model_class=${MODEL_CLASS} begin $(date +%Y%m%d_%H%M%S)"
echo "[driver] nargs=$# ckpts_csv=[${CKPTS_CSV}] groups_csv=[${GROUPS_CSV}]"
RC=0

# 组格式防呆：必须形如 task:seed。任何一条不符合就立即退出——避免把畸形 id
# 当成任务名反复喂给 smoke，那样只会得到一连串 "unknown task(s)" 并浪费一整轮启动。
#
# 变量名必须避开 GROUPS：GROUPS 是 bash 内建的特殊数组（保存当前用户的组 ID），
# 对它的赋值/read 会被静默忽略，`for G in "${GROUPS[@]}"` 于是去遍历 gid。
# 实测踩过：12 个组的 CSV 被换成了 ubuntu 的 12 个 gid（1000 4 20 24 25 27 29 30
# 44 46 119 120），smoke 全部报 unknown task(s)。
IFS=',' read -r -a GROUP_SPECS <<< "${GROUPS_CSV}"
for _g in "${GROUP_SPECS[@]}"; do
  if [[ ! "${_g}" =~ ^[A-Za-z_][A-Za-z0-9_]*:[0-9]+$ ]]; then
    echo "[driver] FATAL bad group spec [${_g}] (expect task:seed); raw groups_csv=[${GROUPS_CSV}]"
    exit 2
  fi
done

# ---------- deploy.yml: 快照 + 改写 + 回读校验 ----------
cp "${DEPLOY}" "${DEPLOY}.bak_${MODEL}_$(date +%Y%m%d_%H%M%S)"
MODEL="${MODEL}" VIEWS="${VIEWS}" MODEL_CLASS="${MODEL_CLASS}" python3 - <<'PY' || exit 3
import os, re
p = "/data/RoboDojo/XPolicyLab/policy/X_VLA/deploy.yml"
s = open(p).read()
model, views, mc = os.environ["MODEL"], os.environ["VIEWS"], os.environ["MODEL_CLASS"]

cams = "[cam_head]" if views == "1" else "[cam_head, cam_left_wrist, cam_right_wrist]"
s2 = re.sub(r"^camera_names:\s*\[.*\]\s*$", f"camera_names: {cams}", s, count=1, flags=re.M)
assert s2 != s or f"camera_names: {cams}" in s, "camera_names rewrite failed"

mc_value = "null" if mc == "-" else mc
s3 = re.sub(r"^policy_model_class:\s*.*$", f"policy_model_class: {mc_value}", s2, count=1, flags=re.M)
assert s3 != s2 or f"policy_model_class: {mc_value}" in s2, "policy_model_class rewrite failed"

open(p, "w").write(s3)
check = open(p).read()
line_cam = next(l for l in check.splitlines() if l.startswith("camera_names:"))
line_mc = next(l for l in check.splitlines() if l.startswith("policy_model_class:"))
assert line_cam.split(":", 1)[1].strip() == cams, f"readback camera_names mismatch: {line_cam}"
assert line_mc.split(":", 1)[1].strip() == mc_value, f"readback policy_model_class mismatch: {line_mc}"
print(f"[driver] deploy.yml -> {line_cam.strip()} | {line_mc.strip()}  (backup kept)")
PY

# ---------- boot / teardown ----------
boot_server () {
  local CKPT="$1" TASK="$2"
  SESS="${MODEL}_${CKPT}"
  SLOG="${LOG_DIR}/${MODEL}_${CKPT}_server_$(date +%Y%m%d_%H%M%S).log"
  screen -ls 2>/dev/null | grep -q "\.${SESS}\b" && screen -S "${SESS}" -X quit >/dev/null 2>&1
  sleep 2
  screen -dmS "${SESS}" bash -c "source ${CONDA} && conda activate RoboDojo && cd ${ROBO} && \
echo '[${MODEL} eval] ckpt=${CKPT} views=${VIEWS} model_class=${MODEL_CLASS} server_started $(date +%Y%m%d_%H%M%S)' > '${SLOG}' 2>&1 && \
exec bash scripts/robodojo.sh server --policy-dir XPolicyLab/policy/X_VLA --task ${TASK} \
  --ckpt ${CKPT_BASE}/${CKPT} --policy-env XVLA --env-cfg arx_x5 --action-type ee \
  --seed 0 --policy-gpu 0 --policy-port ${PORT} --bind-host 127.0.0.1 >> '${SLOG}' 2>&1"
  for i in $(seq 1 75); do
    if grep -q "model_chunk_size" "${SLOG}" 2>/dev/null && ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
      echo "[driver] ${CKPT} SERVER_READY ~$((i*4))s log=${SLOG}"; return 0
    fi
    if grep -qE "Traceback \(most recent call last\)|CUDA out of memory|No module named|Error:" "${SLOG}" 2>/dev/null; then
      echo "[driver] ${CKPT} SERVER_ERROR"; tail -30 "${SLOG}"; return 1
    fi
    sleep 4
  done
  echo "[driver] ${CKPT} SERVER_TIMEOUT"; tail -35 "${SLOG}"; return 1
}

check_server_config () {
  local SLOG="$1"
  local cam_line mc_line want_cam want_mc
  # 服务端用 Python list repr 打印，比对时按 repr 拼期望值。
  cam_line=$(grep -o "camera_names=\[[^]]*\]" "${SLOG}" | tail -1)
  mc_line=$(grep -o "policy_model_class=[A-Za-z_]*" "${SLOG}" | tail -1)
  echo "[driver] server config: ${cam_line} | ${mc_line}"
  if [ "${VIEWS}" = "1" ]; then
    want_cam="camera_names=['cam_head']"
  else
    want_cam="camera_names=['cam_head', 'cam_left_wrist', 'cam_right_wrist']"
  fi
  if [ "${MODEL_CLASS}" = "-" ]; then
    want_mc="policy_model_class=XVLA"
  else
    want_mc="policy_model_class=WristActionResidualXVLA"
  fi
  if [ "${cam_line}" != "${want_cam}" ] || [ "${mc_line}" != "${want_mc}" ]; then
    echo "[driver] CONFIG_MISMATCH want '${want_cam}' / '${want_mc}'"; return 1
  fi
  echo "[driver] SERVER_CONFIG_OK"; return 0
}

run_group () {
  local CKPT="$1" TASK="$2" SEED="$3"
  local RID="${MODEL}_${CKPT}_${TASK}_s${SEED}"
  local TS LOG WRAP INNER
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG="${LOG_DIR}/${RID}_${TS}.log"
  {
    echo "[driver] model=${MODEL} ckpt=${CKPT} task=${TASK} seed=${SEED} eval_num=${EVAL_NUM} client_start ${TS}"
    source "${CONDA}" && conda activate RoboDojo
    cd "${ROBO}" || exit 1
    bash scripts/robodojo.sh smoke \
      --policy-dir XPolicyLab/policy/X_VLA \
      --ckpt "${CKPT}" \
      --policy-host 127.0.0.1 \
      --policy-port "${PORT}" \
      --env-cfg arx_x5 \
      --action-type ee \
      --env-gpu 0 \
      --seed "${SEED}" \
      --eval-num "${EVAL_NUM}" \
      --run-id "${RID}" \
      --only "${TASK}"
    echo "CLIENT_EXIT=$?"
  } >> "${LOG}" 2>&1
  WRAP="${ROBO}/smoke_results/${RID}.json"
  INNER=""
  [ -f "${WRAP}" ] && INNER=$(python3 -c "import json;d=json.load(open('${WRAP}'));rs=d.get('results') or [];print(rs[0]['result_path'] if rs else '')" 2>/dev/null)
  if [ -n "${INNER}" ] && [ -f "${INNER}" ]; then
    cp "${INNER}" "${STAGE}/${CKPT}_${TASK}_s${SEED}.json"
    python3 - "${STAGE}/${CKPT}_${TASK}_s${SEED}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
items = list(d["details"].values()) if isinstance(d["details"], dict) else d["details"]
succ = sum(1 for v in items if v.get("success"))
print(f"[driver] RESULT {succ}/{len(items)} score={d.get('score', 0):.2f} -> {sys.argv[1]}")
PY
    echo "[driver] STAGED ${CKPT} ${TASK} s${SEED}"; return 0
  fi
  echo "[driver] RESULT_MISSING ${CKPT} ${TASK} s${SEED} wrap=${WRAP} inner=${INNER}"
  tail -20 "${LOG}"
  return 1
}

IFS=',' read -r -a CKPTS <<< "${CKPTS_CSV}"
echo "[driver] parsed ckpts=${#CKPTS[@]} [${CKPTS[*]}] groups=${#GROUP_SPECS[@]} [${GROUP_SPECS[*]}]"

for CKPT in "${CKPTS[@]}"; do
  PENDING=()
  for G in "${GROUP_SPECS[@]}"; do
    T="${G%%:*}"; S="${G##*:}"
    [ -f "${STAGE}/${CKPT}_${T}_s${S}.json" ] && { echo "[driver] skip staged ${CKPT} ${T} s${S}"; continue; }
    PENDING+=("${T}:${S}")
  done
  [ ${#PENDING[@]} -eq 0 ] && { echo "[driver] ${CKPT} all groups staged, skip"; continue; }

  FIRST_TASK="${PENDING[0]%%:*}"
  if boot_server "${CKPT}" "${FIRST_TASK}"; then
    check_server_config "${SLOG}" || RC=1
    for G in "${PENDING[@]}"; do
      run_group "${CKPT}" "${G%%:*}" "${G##*:}" || RC=1
    done
  else
    RC=1
  fi
  screen -S "${MODEL}_${CKPT}" -X quit >/dev/null 2>&1 || true
  pkill -9 -f "eval_client/main.py" >/dev/null 2>&1 || true
  sleep 3
done

echo "[driver] ${MODEL} DONE rc=${RC} $(date +%Y%m%d_%H%M%S)"
ls -la "${STAGE}"
exit ${RC}
