#!/usr/bin/env bash
# 串行链：等 x1_mainvis_6k 驱动退出后，依次评测 x1_mainvis_sd01_6k / A1_sfbase / A2_sf。
#
# 四个模型共用同一口径：EVAL_NUM=6、3 路图像、pace off、policy_model_class null、
# policy_seed 42、action_type ee、port 6000，组集合固定为
# plug_in_charger:0 + stack_bowls:0/1 + fill_pen_holder:0/1。
#
# 每个模型前后各查一次 /data 余量；低于 MIN_FREE_GB 就停下并记录，不继续下一个
# （数据盘写满在本项目会伪装成显存 OOM）。每个模型的 rc 单独记入链日志。
set -uo pipefail

EVAL_NUM="${EVAL_NUM:-6}"
MIN_FREE_GB="${MIN_FREE_GB:-4}"
WAIT_FOR="${WAIT_FOR:-x1mainvis6k_driver}"
# 变量名必须叫 GROUPS_CSV：GROUPS 是 bash 内建特殊数组（当前用户组 ID），
# 对它的赋值会被静默忽略 —— 实测踩过，${GROUPS} 展开成了 gid「1000」。
# run_model_eval.sh 里同样的坑已有注释记录。
GROUPS_CSV="plug_in_charger:0,stack_bowls:0,stack_bowls:1,fill_pen_holder:0,fill_pen_holder:1"
DRIVER=/data/outputs/run_model_eval.sh

LOG=/data/outputs/eval_chain_x1a_$(date +%Y%m%d_%H%M%S).log
echo "[chain] log=${LOG} eval_num=${EVAL_NUM} min_free_gb=${MIN_FREE_GB}" > "${LOG}"
echo "[chain] wait_for=${WAIT_FOR} groups=${GROUPS_CSV}" >> "${LOG}"

free_gb () { df -B1 /data | tail -1 | awk '{printf "%.2f", $4/1024/1024/1024}'; }

while screen -ls 2>/dev/null | grep -q "\.${WAIT_FOR}\b"; do sleep 60; done
echo "[chain] ${WAIT_FOR} gone at $(date +%F_%T), free=$(free_gb)GB" >> "${LOG}"

run_model () {
  local MODEL="$1" BASE="$2" CKPTS="$3"
  local f
  f=$(free_gb)
  echo "[chain] --- ${MODEL} start $(date +%F_%T) free=${f}GB ---" >> "${LOG}"
  awk -v f="${f}" -v m="${MIN_FREE_GB}" 'BEGIN{exit !(f<m)}' && {
    echo "[chain] ABORT ${MODEL}: free=${f}GB < ${MIN_FREE_GB}GB" >> "${LOG}"
    return 9
  }
  # 单个模型内的组级失败（rc=1）不中断链：staged 结果已落盘，缺的组可由
  # 幂等驱动补跑（stage 文件存在即 skip）。只有磁盘不足（rc=9）才中断整条链。
  EVAL_NUM="${EVAL_NUM}" bash "${DRIVER}" "${MODEL}" "${BASE}" "${CKPTS}" 3 - "${GROUPS_CSV}" >> "${LOG}" 2>&1
  local rc=$?
  echo "[chain] --- ${MODEL} rc=${rc} $(date +%F_%T) free=$(free_gb)GB ---" >> "${LOG}"
  return 0
}

for spec in \
  "x1_mainvis_sd01_6k /data/x1_mainvis_sd01_6k/pretrained    ckpt-4000,ckpt-5000,ckpt-6000" \
  "A1_sfbase           /workspace/sf_sim/A1/pretrained       ckpt-2000,ckpt-3000,ckpt-4000" \
  "A2_sf               /workspace/sf_sim/A2/pretrained       ckpt-2000,ckpt-3000,ckpt-4000" ; do
  # shellcheck disable=SC2086
  run_model ${spec} || { echo "[chain] chain halted after $(echo ${spec} | awk '{print $1}')" >> "${LOG}"; break; }
done

echo "[chain] ALL DONE $(date +%F_%T) free=$(free_gb)GB" >> "${LOG}"
