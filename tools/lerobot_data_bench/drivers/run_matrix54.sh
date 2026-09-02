#!/bin/bash
# Matrix 5.4: return_uint8 isolation on lerobot 0.6.0 + torchcodec nw=0 (native uint8).
# float vs uint8, unified conditions (shuffle=True, openpi collate, warmup 10, 100 batches, 3 reps).
set -u
export LD_LIBRARY_PATH=/cloud/cloud-ssd1/lerobot_bench/fflib:/cloud/cloud-ssd1/lerobot_bench/venv_l060/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}
PY=/cloud/cloud-ssd1/lerobot_bench/venv_l060/bin/python
cd /cloud/cloud-ssd1/lerobot_bench
OUT=/cloud/cloud-ssd1/lerobot_bench/out
mkdir -p "$OUT"
ROOT=/cloud/cloud-ssd1/lerobot_data
REPO=real_lerobot_v30_joint
common="--dataset-root $ROOT --repo-id $REPO --batch-size 32 --warmup-batches 10 --num-batches 100 --seed 0"

echo "ENV: $($PY -c 'import sys,lerobot,torch,torchcodec;print(sys.version.split()[0],"lerobot",lerobot.__version__,"torch",torch.__version__,"torchcodec",torchcodec.__version__)')"

run() {
  local tag=$1; shift
  for r in 1 2 3; do
    echo "=== $tag rep$r start $(date +%H:%M:%S)"
    timeout 1500 $PY bench_v2.py $common "$@" --out "$OUT/${tag}_rep${r}.json"
    echo "rc=$?  $tag rep$r done $(date +%H:%M:%S)"
  done
}

run mat54_l060_tc_float --backend lerobot --video-backend torchcodec --num-workers 0
run mat54_l060_tc_uint8 --backend lerobot --video-backend torchcodec --num-workers 0 --return-uint8
echo "ALL DONE $(date +%H:%M:%S)"
