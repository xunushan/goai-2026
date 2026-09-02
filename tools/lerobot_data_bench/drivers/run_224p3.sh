#!/bin/bash
# Item2: lerobot 0.6.0 下评估 224x224（gop=2） vs 640x480 原版，eps 46-62 子集，float vs uint8。
# eps 46-62 是 ds224 三个 224 视频覆盖的完整范围（41-45 在 left_wrist file-000，ds224 无此文件）。
# 640 控制 = 原数据集 root + episodes=41-62；224 = ds224 子集目录（4 个视频已换成 224 重编码）。
set -u
export LD_LIBRARY_PATH=/cloud/cloud-ssd1/lerobot_bench/fflib:/cloud/cloud-ssd1/lerobot_bench/venv_l060/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}
PY=/cloud/cloud-ssd1/lerobot_bench/venv_l060/bin/python
cd /cloud/cloud-ssd1/lerobot_bench
OUT=/cloud/cloud-ssd1/lerobot_bench/out/p06_224
mkdir -p "$OUT"
EPISODES="46-62"
common="--batch-size 32 --warmup-batches 10 --num-batches 100 --seed 0 --backend lerobot --video-backend torchcodec --num-workers 0 --episodes $EPISODES"

echo "ENV: $($PY -c 'import sys,lerobot,torch,torchcodec;print(sys.version.split()[0],"lerobot",lerobot.__version__,"torch",torch.__version__,"torchcodec",torchcodec.__version__)')"

run() {  # $1=tag $2=dataset-root $3=repo-id $4=extra
  local tag=$1 root=$2 repo=$3 extra=$4
  for r in 1 2 3; do
    echo "=== $tag rep$r start $(date +%H:%M:%S)"
    timeout 1500 $PY bench_v2.py $common --dataset-root "$root" --repo-id "$repo" $extra --out "$OUT/${tag}_rep${r}.json"
    echo "rc=$? $tag rep$r done $(date +%H:%M:%S)"
  done
}
run ctl640_float  /cloud/cloud-ssd1/lerobot_data real_lerobot_v30_joint ""
run ctl640_uint8  /cloud/cloud-ssd1/lerobot_data real_lerobot_v30_joint "--return-uint8"
run p3_224_float  /cloud/cloud-ssd1/lerobot_bench/ds224 real_lerobot_v30_joint_224p3 ""
run p3_224_uint8  /cloud/cloud-ssd1/lerobot_bench/ds224 real_lerobot_v30_joint_224p3 "--return-uint8"
echo "224 MATRIX DONE $(date +%F_%T)"
