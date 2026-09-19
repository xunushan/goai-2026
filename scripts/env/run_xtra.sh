#!/usr/bin/env bash
# xtra: 3 follow-ups on train-4090 (code 2e55cb6), pool env (no PREALLOCATE=false).
# 1) P1+EMA @ MEM_FRACTION=0.85  -> bracket true peak (should run if true <= 20.4G)
# 2) P0+EMA remat-off bs2 @0.90   -> confirm bs1 is the batch ceiling on 24G (expect OOM)
# 3) P1+EMA @0.90 with a real checkpoint save at step 8 (EMA trainable-only format, no save OOM)
set +e
export LD_LIBRARY_PATH=/data/goai/envs/pi05_l060/fflib:/data/goai/envs/pi05_l060/lib/python3.12/site-packages/av.libs
export PYTHONPATH=/data/openpi/src:/data/openpi/packages/openpi-client/src
export OPENPI_PI05_BASE=/data/checkpoints/pi05_base/params
export CUDA_VISIBLE_DEVICES=0
PY=/data/goai/envs/pi05_l060/bin/python
CKPT=/cloud/cloud-ssd1/pi05_smoke/ckpts_xtra
LOGD=/cloud/cloud-ssd1/pi05_smoke/logs_xtra
SUMMARY=$LOGD/_xtra_summary.txt
mkdir -p $CKPT $LOGD
cd /data/openpi
: > $SUMMARY
echo "code=$(git rev-parse --short HEAD)" >> $SUMMARY

run_one () {
  scheme=$1; tag=$2; fraction=$3; steps=$4; extra=$5; keep=$6
  rm -f $LOGD/$tag.mem $LOGD/$tag.log
  echo "===== $tag : goai_pi05_$scheme frac=$fraction steps=$steps $extra =====" >> $SUMMARY
  XLA_PYTHON_CLIENT_MEM_FRACTION=$fraction nohup $PY scripts/train.py goai_pi05_$scheme \
    --exp-name "$tag" --checkpoint-base-dir $CKPT \
    --num-train-steps $steps --log-interval 1 \
    --no-wandb-enabled --overwrite $extra \
    > $LOGD/$tag.log 2>&1 &
  PID=$!
  ( while kill -0 $PID 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 1; done ) > $LOGD/$tag.mem &
  wait $PID
  RC=$?
  PEAK=$(sort -n $LOGD/$tag.mem 2>/dev/null | tail -1)
  echo "rc=$RC peak_used_MiB=${PEAK:-NA}" >> $SUMMARY
  grep -E "EMA decay:|Gradient checkpointing:|Parameter audit:|EMA parameter audit:|Step 1 data timing|Step [0-9]+ data timing" $LOGD/$tag.log | head -2 >> $SUMMARY
  grep -E "Step 1 data timing|Step [0-9]+ data timing" $LOGD/$tag.log | tail -1 >> $SUMMARY
  if [ "$RC" -ne 0 ]; then
    grep -E "RESOURCE_EXHAUSTED|OutOfMemory|Error" $LOGD/$tag.log | tail -1 >> $SUMMARY
    echo "RESULT: CRASHED" >> $SUMMARY
  else
    echo "RESULT: OK" >> $SUMMARY
  fi
  if [ "$keep" != "1" ]; then rm -rf $CKPT/goai_pi05_$scheme/$tag; fi
}

run_one p1 br_p1_ema85 0.85 6 "" 0
run_one p0 ro_p0_ema_b2 0.90 6 "--model.no-gradient-checkpointing --batch-size 2" 0
run_one p1 sav_p1_ema 0.90 8 "--save-interval 8 --keep-period 8 --batch-size 1" 1
echo "XTRA_ALL_DONE" >> $SUMMARY
