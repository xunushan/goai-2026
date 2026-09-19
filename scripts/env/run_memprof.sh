#!/usr/bin/env bash
# True-peak memory profile on train-4090. Direct train.py invocation (no run_smoke)
# so --save-interval can be disabled (tyro first-wins made run_smoke's flag stick).
set -e
export LD_LIBRARY_PATH=/data/goai/envs/pi05_l060/fflib:/data/goai/envs/pi05_l060/lib/python3.12/site-packages/av.libs
export PYTHONPATH=/data/openpi/src:/data/openpi/packages/openpi-client/src
export OPENPI_PI05_BASE=/data/checkpoints/pi05_base/params
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # let nvidia-smi see TRUE live usage
PY=/data/goai/envs/pi05_l060/bin/python
CKPT=/cloud/cloud-ssd1/pi05_smoke/ckpts_memprof
LOGD=/cloud/cloud-ssd1/pi05_smoke/logs
mkdir -p $CKPT $LOGD
cd /data/openpi
: > $LOGD/_memprof_summary.txt
for spec in \
  "p1 mp_b1_noema --batch-size 1 --ema-decay None" \
  "p1 mp_b2_noema --batch-size 2 --ema-decay None" \
  "p1 mp_b3_noema --batch-size 3 --ema-decay None" \
  "p1 mp_b1_ema  --batch-size 1" \
  "p0 mp_b1_noema --batch-size 1 --ema-decay None" \
  "p0 mp_b1_ema  --batch-size 1"
do
  read -r scheme tag extra <<< "$spec"
  rm -f $LOGD/$tag.mem
  nohup $PY scripts/train.py goai_pi05_$scheme \
    --exp-name "$tag" --checkpoint-base-dir $CKPT \
    --num-train-steps 5 --save-interval 100000000 --keep-period 100000000 \
    --no-wandb-enabled --overwrite $extra \
    > $LOGD/$tag.log 2>&1 &
  PID=$!
  ( while kill -0 $PID 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 1; done ) > $LOGD/$tag.mem &
  wait $PID
  RC=$?
  PEAK=$(sort -n $LOGD/$tag.mem 2>/dev/null | tail -1)
  CRASH=$(tail -3 $LOGD/$tag.log | grep -cE 'Traceback|RESOURCE_EXHAUSTED|Error' || true)
  echo "$tag rc=$RC peak_MiB=${PEAK:-NA} crash=$CRASH  <= $extra" >> $LOGD/_memprof_summary.txt
  rm -rf $CKPT/goai_pi05_$scheme/$tag   # drop any ckpt dir to save disk
done
echo MEMPROF_ALL_DONE >> $LOGD/_memprof_summary.txt
