#!/usr/bin/env bash
# Driver #2: corrected S5 (remat off) + batch sweep + loader workers bench, on train-4090.
# All on P1 scheme except where noted. Save-interval huge -> no ckpt writes.
set +e
STEPS="${1:-6}"
RS=/data/pi05_env_scripts/run_smoke.sh
: > /cloud/cloud-ssd1/pi05_smoke/logs/_driver2_summary.txt
for spec in \
  "p1 smk_s5_p1_ema_noremat --model.no-gradient-checkpointing" \
  "p1 smk_b2_p1_noema_remat_bs2 --batch-size 2 --ema-decay None" \
  "p1 smk_w2_p1_noema_remat --num-workers 2 --ema-decay None" \
  "p1 smk_w4_p1_noema_remat --num-workers 4 --ema-decay None"
do
  read -r scheme tag extra <<< "$spec"
  echo "===== $tag =====" >> /cloud/cloud-ssd1/pi05_smoke/logs/_driver2_summary.txt
  bash $RS "$scheme" "$tag" "$STEPS" $extra --overwrite --save-interval 100000000 --keep-period 100000000
  while pgrep -f "scripts/train.py goai_pi05_$scheme --exp-name $tag" >/dev/null; do sleep 5; done
  LOG=/cloud/cloud-ssd1/pi05_smoke/logs/$tag.log
  MEM=/cloud/cloud-ssd1/pi05_smoke/logs/$tag.mem
  PEAK=$(sort -n "$MEM" 2>/dev/null | tail -1)
  echo "peak_gpu_mem_MiB=$PEAK" >> /cloud/cloud-ssd1/pi05_smoke/logs/_driver2_summary.txt
  # first real timing line + last real timing line (steady)
  FIRST=$(grep -E 'Step [0-9]+ data timing' "$LOG" 2>/dev/null | head -1)
  LAST=$(grep -E 'Step [0-9]+ data timing' "$LOG" 2>/dev/null | tail -1)
  echo "first_timing: $FIRST" >> /cloud/cloud-ssd1/pi05_smoke/logs/_driver2_summary.txt
  echo "last_timing:  $LAST" >> /cloud/cloud-ssd1/pi05_smoke/logs/_driver2_summary.txt
  # error marker if crashed
  tail -1 "$LOG" | grep -qE 'Traceback|Error|RESOURCE_EXHAUSTED' && echo "RESULT: CRASHED" >> /cloud/cloud-ssd1/pi05_smoke/logs/_driver2_summary.txt || true
done
echo "ALL_DONE2" >> /cloud/cloud-ssd1/pi05_smoke/logs/_driver2_summary.txt
