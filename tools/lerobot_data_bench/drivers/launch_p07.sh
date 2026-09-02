#!/bin/bash
# p07 launch: v4.4 cells then v6.0 cells sequentially; log to /data/outputs with timestamp.
cd /cloud/cloud-ssd1/lerobot_bench/p07
TS=$(date +%Y%m%d_%H%M%S)
LOG=/data/outputs/p07_sim_${TS}.log
echo "$LOG" > /cloud/cloud-ssd1/lerobot_bench/p07/lastlog.path
{
  echo "=== P07 start $(date +%F_%T) ==="
  echo "=== V44 start $(date +%T) ==="
  bash run_v44.sh
  echo "=== V60 start $(date +%T) ==="
  bash run_v60.sh
  echo "=== P07 ALL DONE $(date +%F_%T) ==="
} > "$LOG" 2>&1
