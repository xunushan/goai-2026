#!/bin/bash
# p07 lerobot 0.6.0 (lerobot060 conda): sim h264 640 vs 224 x float/uint8, torchcodec nw=0.
set -u
export LD_LIBRARY_PATH=/data/venvs/lerobot060/lib:${LD_LIBRARY_PATH:-}
PY=/data/venvs/lerobot060/bin/python
cd /cloud/cloud-ssd1/lerobot_bench/p07
OUT=/cloud/cloud-ssd1/lerobot_bench/out/p07
mkdir -p "$OUT"
ROOT=/cloud/cloud-ssd1/lerobot_data
common="--backend lerobot --video-backend torchcodec --num-workers 0 --batch-size 32 --warmup-batches 10 --num-batches 100 --seed 0"

echo "ENV: $($PY -c 'import sys,lerobot,torch,torchcodec;print(sys.version.split()[0],"lerobot",lerobot.__version__,"torch",torch.__version__,"torchcodec",torchcodec.__version__)')"

run() {  # $1=tag $2=repo $3=extra
  local tag=$1 repo=$2 extra=$3
  for r in 1 2 3; do
    echo "=== $tag rep$r start $(date +%H:%M:%S)"
    timeout 2400 $PY bench_v2.py $common --dataset-root "$ROOT" --repo-id "$repo" $extra --out "$OUT/${tag}_rep${r}.json"
    echo "rc=$?  $tag rep$r done $(date +%H:%M:%S)"
  done
}
run v60_joint640_float sim_lerobot_v30_joint ""
run v60_joint640_uint8 sim_lerobot_v30_joint "--return-uint8"
run v60_joint224_float sim_lerobot_v30_joint-224x224 ""
run v60_joint224_uint8 sim_lerobot_v30_joint-224x224 "--return-uint8"
echo "V60 DONE $(date +%F_%T)"
