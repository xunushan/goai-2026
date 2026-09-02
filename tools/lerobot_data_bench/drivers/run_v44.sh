#!/bin/bash
# p07 lerobot 0.4.4 (pi05_openpi): sim h264 640 vs 224, torchcodec nw=0, float only.
set -u
export LD_LIBRARY_PATH=/cloud/cloud-ssd1/lerobot_bench/fflib:/data/venvs/pi05_openpi/lib/python3.11/site-packages/av.libs:${LD_LIBRARY_PATH:-}
PY=/data/venvs/pi05_openpi/bin/python
cd /cloud/cloud-ssd1/lerobot_bench/p07
OUT=/cloud/cloud-ssd1/lerobot_bench/out/p07
mkdir -p "$OUT"
ROOT=/cloud/cloud-ssd1/lerobot_data
common="--backend lerobot --video-backend torchcodec --num-workers 0 --batch-size 32 --warmup-batches 10 --num-batches 100 --seed 0"

echo "ENV: $($PY -c 'import sys,lerobot,torch,torchcodec;print(sys.version.split()[0],"lerobot",lerobot.__version__,"torch",torch.__version__,"torchcodec",torchcodec.__version__)')"

run() {  # $1=tag $2=repo
  local tag=$1 repo=$2
  for r in 1 2 3; do
    echo "=== $tag rep$r start $(date +%H:%M:%S)"
    timeout 2400 $PY bench_v2.py $common --dataset-root "$ROOT" --repo-id "$repo" --out "$OUT/${tag}_rep${r}.json"
    echo "rc=$?  $tag rep$r done $(date +%H:%M:%S)"
  done
}
run v44_joint640 sim_lerobot_v30_joint
run v44_joint224 sim_lerobot_v30_joint-224x224
echo "V44 DONE $(date +%F_%T)"
