#!/usr/bin/env bash
# download_eval_results.sh
# 从 issac-server 和 policy-server 下载仿真评测结果到本地。
#
# 每个 episode 的输出包含以下文件：
#   _result.json                          评测结果：success_rate、score、每个 layout 的详情
#   episode_0000000_cam_head_*.mp4        头部相机视频（_success 或 _fail 后缀）
#   episode_0000000_cam_left_wrist_*.mp4A 左腕相机视频
#   episode_0000000_cam_right_wrist_*.mp4 右腕相机视频
#   ik_failures.jsonl                     cuRobo IK 求解失败记录（每行一条 JSON）
#
# 用法：
#   bash scripts/download_eval_results.sh <task_name> <policy_name> <ckpt_name> <timestamp> [local_dir]
#
# 参数：
#   task_name    任务名，如 stack_blocks
#   policy_name  策略名，如 act_lerobot 或 xvla_robtwin
#   ckpt_name    checkpoint 名，如 act-30k 或 X-VLA-RoboTwin2
#   timestamp    结果目录的时间戳，如 2026-07-28_19-15-40
#   local_dir    本地保存目录（可选，默认为 eval_results/<task_name>_<policy_name>）
#
# 示例：
#   bash scripts/download_eval_results.sh stack_blocks xvla_robtwin X-VLA-RoboTwin2 2026-07-28_19-15-40
#   bash scripts/download_eval_results.sh stack_blocks act_lerobot act-30k 2026-07-28_09-46-34 eval_results/stack_blocks_act30k
#
# 前提：
#   - 本地已配置 SSH 别名 issac-server 和 policy-server
#   - issac-server 上的结果路径为 /data/RoboDojo/eval_result/RoboDojo/<task>/...
#   - policy-server 上的服务日志路径为 /data/RoboDojo/outputs/

set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <task_name> <policy_name> <ckpt_name> <timestamp> [local_dir]" >&2
    exit 1
fi

TASK_NAME="$1"
POLICY_NAME="$2"
CKPT_NAME="$3"
TIMESTAMP="$4"
LOCAL_DIR="${5:-eval_results/${TASK_NAME}_${POLICY_NAME}}"

# Isaac Sim 评测结果的远程路径组件
ENV_CFG="arx_x5"
ACTION_TYPE="ee"
LAYOUT_ID="0"
RESULT_SUBPATH="${POLICY_NAME}/${ENV_CFG}/${LAYOUT_ID}_ckpt_name=${CKPT_NAME},action_type=${ACTION_TYPE}/${TIMESTAMP}"
REMOTE_BASE="/data/RoboDojo/eval_result/RoboDojo/${TASK_NAME}/${RESULT_SUBPATH}"

mkdir -p "${LOCAL_DIR}"

echo "[download] task=${TASK_NAME} policy=${POLICY_NAME} ckpt=${CKPT_NAME} ts=${TIMESTAMP}"
echo "[download] remote: issac-server:${REMOTE_BASE}"
echo "[download] local:  ${LOCAL_DIR}"

# 1. 评测结果 JSON
echo "[download] _result.json"
scp "issac-server:${REMOTE_BASE}/_result.json" "${LOCAL_DIR}/"

# 2. IK 失败日志
echo "[download] ik_failures.jsonl"
scp "issac-server:${REMOTE_BASE}/ik_failures.jsonl" "${LOCAL_DIR}/" 2>/dev/null || echo "[download] ik_failures.jsonl not found, skipping"

# 3. 三路相机视频（success 或 fail 后缀）
echo "[download] videos"
for CAM in cam_head cam_left_wrist cam_right_wrist; do
    for SUFFIX in success fail; do
        REMOTE_FILE="${REMOTE_BASE}/episode_0000000_${CAM}_${SUFFIX}.mp4"
        if ssh issac-server "test -f ${REMOTE_FILE}" 2>/dev/null; then
            scp "issac-server:${REMOTE_FILE}" "${LOCAL_DIR}/"
        fi
    done
done

# 4. policy-server 服务日志
echo "[download] policy-server log"
scp "policy-server:/data/RoboDojo/outputs/${POLICY_NAME}_server.log" "${LOCAL_DIR}/" 2>/dev/null \
    || echo "[download] ${POLICY_NAME}_server.log not found, skipping"

# 5. issac-server 评测客户端日志
echo "[download] issac-server eval log"
scp "issac-server:/data/RoboDojo/outputs/xvla_eval_${TASK_NAME}.log" "${LOCAL_DIR}/" 2>/dev/null \
    || scp "issac-server:/data/RoboDojo/outputs/act_lerobot_server_*.log" "${LOCAL_DIR}/" 2>/dev/null \
    || echo "[download] eval client log not found, skipping"

echo "[download] done. Files in ${LOCAL_DIR}/:"
ls -lh "${LOCAL_DIR}/"
