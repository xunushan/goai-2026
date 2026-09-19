#!/usr/bin/env bash
# memverify: true-peak memory of the configs that fit under pool 0.90 (code 2e55cb6).
# PREALLOCATE=false -> nvidia-smi reflects true live allocation. Sampler alive for the whole run.
set +e
export LD_LIBRARY_PATH=/data/goai/envs/pi05_l060/fflib:/data/goai/envs/pi05_l060/lib/python3.12/site-packages/av.libs
export PYTHONPATH=/data/openpi/src:/data/openpi/packages/openpi-client/src
export OPENPI_PI05_BASE=/data/checkpoints/pi05_base/params
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/data/goai/envs/pi05_l060/bin/python
CKPT=/cloud/cloud-ssd1/pi05_smoke/ckpts_memv
LOGD=/cloud/cloud-ssd1/pi05_smoke/logs_memv
SUMMARY=$LOGD/_memverify_summary.txt
STEPS="${1:-6}"
mkdir -p $CKPT $LOGD
cd /data/openpi
: > $SUMMARY
echo "steps=$STEPS code=$(git rev-parse --short HEAD) mem_env=PREALLOCATE=false" >> $SUMMARY

run_one () {
  scheme=$1; tag=$2; extra=$3
  rm -f $LOGD/$tag.mem $LOGD/$tag.log
  echo "===== $tag : goai_pi05_$scheme $extra =====" >> $SUMMARY
  nohup $PY scripts/train.py goai_pi05_$scheme \
    --exp-name "$tag" --checkpoint-base-dir $CKPT \
    --num-train-steps $STEPS \
    --save-interval 100000000 --keep-period 100000000 \
    --no-wandb-enabled --overwrite $extra \
    > $LOGD/$tag.log 2>&1 &
  PID=$!
  ( while kill -0 $PID 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 1; done ) > $LOGD/$tag.mem &
  wait $PID
  RC=$?
  PEAK=$(sort -n $LOGD/$tag.mem 2>/dev/null | tail -1)
  echo "rc=$RC peak_true_MiB=${PEAK:-NA}" >> $SUMMARY
  grep -E "EMA decay:|Gradient checkpointing:|Parameter audit:|EMA parameter audit:" $LOGD/$tag.log | tail -4 >> $SUMMARY
  if [ "$RC" -ne 0 ]; then
    grep -E "RESOURCE_EXHAUSTED|OutOfMemory" $LOGD/$tag.log | tail -1 >> $SUMMARY
    echo "RESULT: CRASHED" >> $SUMMARY
  else
    echo "RESULT: OK" >> $SUMMARY
  fi
  rm -rf $CKPT/goai_pi05_$scheme/$tag
}

run_one p1 mp_p1_noema "--ema-decay None"
run_one p1 mp_p1_ema ""
run_one p0 mp_p0_noema "--ema-decay None"
run_one p0 mp_p0_ema ""
run_one p0 mp_ro_p0_ema "--model.no-gradient-checkpointing"
echo "MEMVERIFY_ALL_DONE" >> $SUMMARY
