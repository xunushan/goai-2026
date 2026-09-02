#!/bin/bash
# Item1: LEROBOT_VIDEO_DECODER_CACHE_SIZE 在 lerobot 0.6.0 下的试验（全量 640 数据集）。
# 0.6.0 默认 cache size=100；'-1'=unbounded。env var 需在进程启动前设置（模块级 cache 在 import 时实例化）。
set -u
export LD_LIBRARY_PATH=/cloud/cloud-ssd1/lerobot_bench/fflib:/cloud/cloud-ssd1/lerobot_bench/venv_l060/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}
PY=/cloud/cloud-ssd1/lerobot_bench/venv_l060/bin/python
cd /cloud/cloud-ssd1/lerobot_bench
OUT=/cloud/cloud-ssd1/lerobot_bench/out/p06cache
mkdir -p "$OUT"
common="--dataset-root /cloud/cloud-ssd1/lerobot_data --repo-id real_lerobot_v30_joint --batch-size 32 --warmup-batches 10 --num-batches 100 --seed 0 --backend lerobot --video-backend torchcodec --num-workers 0"

echo "ENV: $($PY -c 'import sys,lerobot,torch,torchcodec;print(sys.version.split()[0],"lerobot",lerobot.__version__,"torch",torch.__version__,"torchcodec",torchcodec.__version__)')"

run() {  # $1=tag $2=env var value (empty = unset/default)
  local tag=$1 val=$2
  if [ -n "$val" ]; then export LEROBOT_VIDEO_DECODER_CACHE_SIZE="$val"; else unset LEROBOT_VIDEO_DECODER_CACHE_SIZE; fi
  echo "--- cache_size='${val:-default100}' ($tag) $(date +%T)"
  for r in 1 2 3; do
    echo "=== $tag rep$r start $(date +%H:%M:%S)"
    timeout 1500 $PY bench_v2.py $common --out "$OUT/${tag}_rep${r}.json"
    echo "rc=$? $tag rep$r done $(date +%H:%M:%S)"
  done
}
unset LEROBOT_VIDEO_DECODER_CACHE_SIZE
run cache16 16
run cache64 64
run cache_default ""
run cache256 256
run cache_unbounded -1
echo "CACHE MATRIX DONE $(date +%F_%T)"
