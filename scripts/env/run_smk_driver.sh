#!/usr/bin/env bash
# Sequential pi05 smoke driver for S1/S2/S3/S5 on train-4090 (24GB).
# S4 (=P1 ema.999 remat, the goai_pi05_p1 default) already measured OOM -> not rerun.
# usage: bash run_smk_driver.sh <steps>
set +e
STEPS="${1:-12}"
RS=/data/pi05_env_scripts/run_smoke.sh
: > /cloud/cloud-ssd1/pi05_smoke/logs/_driver_summary.txt
for spec in \
  "p0 smk_s1_p0_noema_remat --ema-decay None" \
  "p0 smk_s2_p0_ema_remat" \
  "p1 smk_s3_p1_noema_remat --ema-decay None" \
  "p1 smk_s5_p1_ema_noremat --no-model.gradient-checkpointing"
do
  read -r scheme tag extra <<< "$spec"
  echo "===== $tag ($scheme) =====" >> /cloud/cloud-ssd1/pi05_smoke/logs/_driver_summary.txt
  bash $RS "$scheme" "$tag" "$STEPS" $extra --overwrite --save-interval 100000000 --keep-period 100000000
  # wait for this run's train.py to finish before next
  while pgrep -f "scripts/train.py goai_pi05_$scheme --exp-name $tag" >/dev/null; do sleep 5; done
  LOG=/cloud/cloud-ssd1/pi05_smoke/logs/$tag.log
  MEM=/cloud/cloud-ssd1/pi05_smoke/logs/$tag.mem
  PEAK=$(sort -n "$MEM" 2>/dev/null | tail -1)
  LAST=$(grep -E 'Step [0-9]+ .*data timing|RESOURCE_EXHAUSTED|OOM|Error|error' "$LOG" 2>/dev/null | tail -3)
  echo "peak_gpu_mem_MiB=$PEAK" >> /cloud/cloud-ssd1/pi05_smoke/logs/_driver_summary.txt
  echo "last_timing_lines: $LAST" >> /cloud/cloud-ssd1/pi05_smoke/logs/_driver_summary.txt
done
echo "ALL_DONE" >> /cloud/cloud-ssd1/pi05_smoke/logs/_driver_summary.txt
