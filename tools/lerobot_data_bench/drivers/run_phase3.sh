#!/bin/bash
# Phase 3: encode variants A (gop=2) + B (gop=10) at 224x224 CRF=25 (libsvtav1),
# then decode-throughput benchmark + PSNR/SSIM quality vs source-downscaled ref.
set -u
export LD_LIBRARY_PATH=/cloud/cloud-ssd1/lerobot_bench/fflib:/data/venvs/pi05_openpi/lib/python3.11/site-packages/av.libs:${LD_LIBRARY_PATH:-}
PY=/data/venvs/pi05_openpi/bin/python
cd /cloud/cloud-ssd1/lerobot_bench
DATA=/cloud/cloud-ssd1/lerobot_data/real_lerobot_v30_joint/videos
OUT=/cloud/cloud-ssd1/lerobot_bench/p3
mkdir -p "$OUT"

SRC_HIGH=$DATA/observation.images.cam_high/chunk-000/file-002.mp4
SRC_LEFT=$DATA/observation.images.cam_left_wrist/chunk-000/file-001.mp4
SRC_RIGHT=$DATA/observation.images.cam_right_wrist/chunk-000/file-001.mp4

echo "== $(hostname) nproc=$(nproc) mem=$(free -g | awk '/Mem:/{print $2"GB"}') $(date +%F_%T)"

encode_variant() {  # $1=tag $2=gop  (8 cores -> sequential encodes)
  local tag=$1 gop=$2
  local d=$OUT/$tag
  mkdir -p "$d"
  echo "--- encode $tag gop=$gop start $(date +%T)"
  $PY phase3_reescale.py --src "$SRC_HIGH"  --dst "$d/cam_high.mp4"        --gop "$gop" --crf 25 --res 224 --preset 8
  $PY phase3_reescale.py --src "$SRC_LEFT"  --dst "$d/cam_left_wrist.mp4"  --gop "$gop" --crf 25 --res 224 --preset 8
  $PY phase3_reescale.py --src "$SRC_RIGHT" --dst "$d/cam_right_wrist.mp4" --gop "$gop" --crf 25 --res 224 --preset 8
  echo "--- encode $tag done $(date +%T)"
}

encode_variant A_gop2 2
encode_variant B_gop10 10

echo "== sizes (MB) =="
du -sb "$OUT"/* | awk '{printf "%.1f MB  %s\n", $1/1e6, $2}'
echo "-- original --"
du -sb "$SRC_HIGH" "$SRC_LEFT" "$SRC_RIGHT" | awk '{printf "%.1f MB  %s\n", $1/1e6, $2}'

echo "== bench control (original 640x480) =="
$PY phase3_bench.py --files "$SRC_HIGH" "$SRC_LEFT" "$SRC_RIGHT" --samples 500 --warmup 30 --seed 0 | tee /tmp/p3_bench_ctl.txt
echo "== bench A gop2 224x224 =="
$PY phase3_bench.py --files "$OUT/A_gop2"/cam_high.mp4 "$OUT/A_gop2"/cam_left_wrist.mp4 "$OUT/A_gop2"/cam_right_wrist.mp4 --samples 500 --warmup 30 --seed 0 | tee /tmp/p3_bench_A.txt
echo "== bench B gop10 224x224 =="
$PY phase3_bench.py --files "$OUT/B_gop10"/cam_high.mp4 "$OUT/B_gop10"/cam_left_wrist.mp4 "$OUT/B_gop10"/cam_right_wrist.mp4 --samples 500 --warmup 30 --seed 0 | tee /tmp/p3_bench_B.txt

echo "== quality cam_high (ref vs A vs B) =="
$PY phase3_quality.py --src "$SRC_HIGH" --variants "$OUT/A_gop2/cam_high.mp4" "$OUT/B_gop10/cam_high.mp4" --n 50 --seed 0 --viz-out "$OUT/viz" | tee /tmp/p3_quality.txt

echo "ALL PHASE3 DONE $(date +%F_%T)"
