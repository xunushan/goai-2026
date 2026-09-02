#!/bin/bash
# Phase 1: decoder cache 16/64/256 on LeRobot 0.4.4 + torchcodec nw=0 (prod path).
# Unified conditions: shuffle=True, openpi collate, warmup 10, 100 batches, 3 reps, seeded.
set -u
export LD_LIBRARY_PATH=/cloud/cloud-ssd1/lerobot_bench/fflib:/data/venvs/pi05_openpi/lib/python3.11/site-packages/av.libs:${LD_LIBRARY_PATH:-}
PY=/data/venvs/pi05_openpi/bin/python
cd /cloud/cloud-ssd1/lerobot_bench
OUT=/cloud/cloud-ssd1/lerobot_bench/out
mkdir -p "$OUT"
ROOT=/cloud/cloud-ssd1/lerobot_data
REPO=real_lerobot_v30_joint
common="--dataset-root $ROOT --repo-id $REPO --batch-size 32 --warmup-batches 10 --num-batches 100 --seed 0 --backend lerobot --video-backend torchcodec --num-workers 0"

echo "ENV: $($PY -c 'import sys,lerobot,torch,torchcodec;print(sys.version.split()[0],"lerobot",lerobot.__version__,"torch",torch.__version__,"torchcodec",torchcodec.__version__)')"

run() {
  local tag=$1; shift
  for r in 1 2 3; do
    echo "=== $tag rep$r start $(date +%H:%M:%S)"
    timeout 1500 $PY bench_v2.py $common "$@" --out "$OUT/${tag}_rep${r}.json"
    echo "rc=$?  $tag rep$r done $(date +%H:%M:%S)"
  done
}

run phase1_cache16  --decoder-cache 16
run phase1_cache64  --decoder-cache 64
run phase1_cache256 --decoder-cache 256
echo "ALL DONE $(date +%H:%M:%S)"
