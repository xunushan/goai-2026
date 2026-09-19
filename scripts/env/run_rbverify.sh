#!/usr/bin/env bash
# rbverify: re-baseline + Q1 remat-off verification on train-4090 (code 2e55cb6).
# Fixed timing + EMA=trainable-only. PREALLOCATE=false -> nvidia-smi shows TRUE live peak.
# Direct train.py invocation (like run_memprof2.sh) so --save-interval disable sticks.
set +e
export LD_LIBRARY_PATH=/data/goai/envs/pi05_l060/fflib:/data/goai/envs/pi05_l060/lib/python3.12/site-packages/av.libs
export PYTHONPATH=/data/openpi/src:/data/openpi/packages/openpi-client/src
export OPENPI_PI05_BASE=/data/checkpoints/pi05_base/params
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/data/goai/envs/pi05_l060/bin/python
CKPT=/cloud/cloud-ssd1/pi05_smoke/ckpts_rb
LOGD=/cloud/cloud-ssd1/pi05_smoke/logs_rb
SUMMARY=$LOGD/_rbverify_summary.txt
STEPS="${1:-16}"
mkdir -p $CKPT $LOGD
cd /data/openpi
: > $SUMMARY
echo "steps=$STEPS  code=$(git rev-parse --short HEAD)" >> $SUMMARY

run_one () {
  scheme=$1; tag=$2; extra=$3
  rm -f $LOGD/$tag.mem $LOGD/$tag.log
  echo "===== $tag : goai_pi05_$scheme $extra =====" >> $SUMMARY
  nohup $PY scripts/train.py goai_pi05_$scheme \
    --exp-name "$tag" --checkpoint-base-dir $CKPT \
    --num-train-steps $STEPS --log-interval 1 \
    --save-interval 100000000 --keep-period 100000000 \
    --no-wandb-enabled --overwrite $extra \
    > $LOGD/$tag.log 2>&1 &
  PID=$!
  ( while kill -0 $PID 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 1; done ) > $LOGD/$tag.mem &
  wait $PID
  RC=$?
  PEAK=$(sort -n $LOGD/$tag.mem 2>/dev/null | tail -1)
  echo "rc=$RC peak_used_MiB=${PEAK:-NA}" >> $SUMMARY
  grep -E "Training batches:|EMA decay:|Gradient checkpointing:|Parameter audit:|EMA parameter audit:" $LOGD/$tag.log | tail -5 >> $SUMMARY
  FIRST=$(grep -E "Step [0-9]+ data timing" $LOGD/$tag.log | head -1)
  LAST=$(grep -E "Step [0-9]+ data timing" $LOGD/$tag.log | tail -1)
  echo "first_timing: ${FIRST:-NONE}" >> $SUMMARY
  echo "last_timing:  ${LAST:-NONE}" >> $SUMMARY
  FIRSTLOSS=$(grep -E "^Step [0-9]+: loss=" $LOGD/$tag.log | head -1)
  LASTLOSS=$(grep -E "^Step [0-9]+: loss=" $LOGD/$tag.log | tail -1)
  echo "first_loss: ${FIRSTLOSS:-NONE}" >> $SUMMARY
  echo "last_loss:  ${LASTLOSS:-NONE}" >> $SUMMARY
  if [ "$RC" -ne 0 ]; then
    grep -E "Traceback|RESOURCE_EXHAUSTED|OutOfMemory|Error" $LOGD/$tag.log | tail -2 >> $SUMMARY
    echo "RESULT: CRASHED rc=$RC" >> $SUMMARY
  else
    echo "RESULT: OK" >> $SUMMARY
  fi
  rm -rf $CKPT/goai_pi05_$scheme/$tag
}

run_one p1 rb_p1_ema ""
run_one p1 rb_p1_noema "--ema-decay None"
run_one p0 rb_p0_ema ""
run_one p0 rb_p0_noema "--ema-decay None"
run_one p1 ro_p1_ema "--model.no-gradient-checkpointing"
run_one p0 ro_p0_ema "--model.no-gradient-checkpointing"
echo "RBVERIFY_ALL_DONE" >> $SUMMARY
