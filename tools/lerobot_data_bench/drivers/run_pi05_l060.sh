#!/usr/bin/env bash
# Run lerobot 0.6.0-on-data-disk (py3.12) dataloader benchmark vs 0.4.4 (pi05)
# baseline, on sim_lerobot_v30_joint-224x224 (torchcodec, nw=0).
# Correctness A/B dumps + bench matrix. Run with: bash run_pi05_l060.sh
set -euo pipefail

# ---- paths ----
V060=/cloud/cloud-ssd1/goai_envs/pi05_py312_l060        # data-disk py3.12 + lerobot 0.6.0
V044=/data/venvs/pi05_openpi                            # py3.11 + lerobot 0.4.4
FFLIB=/cloud/cloud-ssd1/lerobot_bench/fflib             # FFmpeg7 soname symlinks (data disk)
DATASET_ROOT=/cloud/cloud-ssd1/lerobot_data
REPO=sim_lerobot_v30_joint-224x224
WORK=/cloud/cloud-ssd1/lerobot_bench/pi05_l060
OUT=$WORK/out
BENCH=$WORK/bench/bench_v2.py
CORRECT=$WORK/correctness
mkdir -p "$OUT" "$CORRECT"

LD060=$V060/lib/python3.12/site-packages/av.libs:$FFLIB
LD044=$V044/lib/python3.11/site-packages/av.libs:$FFLIB

TS=$(date +%Y%m%d_%H%M%S)
LOG=/data/outputs/pi05_l060_${TS}.log
exec > >(tee -a "$LOG") 2>&1
echo "== run_pi05_l060 started $(date) | log $LOG =="
echo "0.6.0 env: $V060 (lerobot $($V060/bin/python -c 'import lerobot;print(lerobot.__version__)'))"
echo "0.4.4 env: $V044 (lerobot $($V044/bin/python -c 'import lerobot;print(lerobot.__version__)'))"

# ---- 1. cross-version correctness dumps ----
echo; echo "=== [correctness] 0.4.4 dump ==="
LD_LIBRARY_PATH=$LD044 PYTHONPATH= "$V044/bin/python" "$CORRECT/check_cross_version.py" \
  --dataset-root "$DATASET_ROOT" --repo-id "$REPO" --out "$CORRECT/a_044.npz"
echo "=== [correctness] 0.6.0 dump ==="
LD_LIBRARY_PATH=$LD060 PYTHONPATH= "$V060/bin/python" "$CORRECT/check_cross_version.py" \
  --dataset-root "$DATASET_ROOT" --repo-id "$REPO" --out "$CORRECT/b_060.npz"
echo "=== [correctness] compare ==="
"$V060/bin/python" "$CORRECT/check_cross_version_compare.py" "$CORRECT/a_044.npz" "$CORRECT/b_060.npz"

# ---- 2. bench matrix (3 reps each) ----
run_bench() { # $1 envpython $2 ld $3 extra-args $4 name
  local py=$1 ld=$2 extra=$3 name=$4
  for k in 1 2 3; do
    echo; echo "=== [bench] $name rep$k ==="
    LD_LIBRARY_PATH=$ld PYTHONPATH= "$py" "$BENCH" \
      --dataset-root "$DATASET_ROOT" --repo-id "$REPO" \
      --video-backend torchcodec --num-workers 0 \
      $extra --out "$OUT/${name}_rep${k}.json"
  done
}
run_bench "$V060/bin/python" "$LD060" "--return-uint8"            b060_s224_uint8
run_bench "$V060/bin/python" "$LD060" ""                          b060_s224_float
run_bench "$V044/bin/python" "$LD044" ""                          b044_s224

echo; echo "== run_pi05_l060 done $(date) =="
