#!/bin/bash
# Launch item1 (cache) + item2 (224 matrix) sequentially in screen; log to /data/outputs with timestamp.
cd /cloud/cloud-ssd1/lerobot_bench
TS=$(date +%Y%m%d_%H%M%S)
LOG=/data/outputs/p06_matrix_${TS}.log
echo "$LOG" > /cloud/cloud-ssd1/lerobot_bench/bench060.logpath
{
  echo "=== ITEM1 cache matrix start $(date +%F_%T) ==="
  ./run_06cache.sh
  echo "=== ITEM2 224 matrix start $(date +%F_%T) ==="
  ./run_224p3.sh
  echo "=== ALL DONE $(date +%F_%T) ==="
} > "$LOG" 2>&1
