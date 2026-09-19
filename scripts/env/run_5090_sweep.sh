#!/usr/bin/env bash
# 5090 (RTX5090 32G, Blackwell sm_120) pi05 P0/P1 冒烟：目标配置 = remat OFF + EMA ON，eff batch 32。
#
# 判定口径（照抄 goai_2026/docs/pi0.5_P0P1冒烟测试结论_24G.md §5.1，不自创）：
#   - 通过/失败一律以 pool 为准：XLA_PYTHON_CLIENT_MEM_FRACTION=0.90（32G → 池 ≈28.8GiB）
#   - 真峰用 PREALLOCATE=false 采 nvidia-smi 高水位；该模式在 bs>=2 有碎片假 OOM，只用于量真峰，不用于判定
#   - 「最大安全 batch」= 无 OOM/NaN + pool 通过 + 真峰留 ≥10% 显存余量 + checkpoint 保存不 OOM
#
# usage:
#   run_5090_sweep.sh scan                                  # 固定 fail-stop 边界扫描矩阵（remat off + EMA on）
#   run_5090_sweep.sh run <scheme> <tag> <b> <a> <steps> <fraction> [extra train.py args...]
#
# 常用环境变量：
#   OUT=<数据盘目录>          必填。日志/ckpt/summary 的根（如 /cloud/cloud-ssd1/pi05_smoke）
#   GOAI_ROOT=/data/goai      venv 根（默认与迁移包还原后的 /data 布局一致）
#   OPENPI_SRC=/data/openpi   训练 fork 代码
#   OPENPI_PI05_BASE=/data/checkpoints/pi05_base/params
#   SAVE_INTERVAL / KEEP_PERIOD   默认 100000000（即禁保存）。注意 tyro 重复参数**先到先得**，
#                                 想开保存必须用这两个变量，不能在 extra 里再追加 --save-interval
#   PREALLOCATE=false         量真峰用；判定仍走 fraction
#   SCAN_EXTRA=<train.py args>  scan 模式每档附加的 train.py 参数。
#                              默认 "--model.no-gradient-checkpointing"（关 remat）；
#                              跑 remat ON 的对照扫描传 SCAN_EXTRA= （空串）
#
# 例：
#   OUT=/cloud/cloud-ssd1/pi05_smoke bash run_5090_sweep.sh scan
#   OUT=... SAVE_INTERVAL=8 KEEP_PERIOD=8 bash run_5090_sweep.sh run p1 sav_p1_b1 1 32 8 0.90 --model.no-gradient-checkpointing
set +e

GOAI_ROOT="${GOAI_ROOT:-/data/goai}"
OPENPI_SRC="${OPENPI_SRC:-/data/openpi}"
BASE_PARAMS="${OPENPI_PI05_BASE:-/data/checkpoints/pi05_base/params}"
OUT="${OUT:?请设置 OUT=<数据盘目录>（日志/ckpt 落这里，勿写系统盘）}"
MODE="${1:?usage: $0 <scan|run> [args...]}"

ENVDIR="$GOAI_ROOT/envs/pi05_l060"
export LD_LIBRARY_PATH="$ENVDIR/fflib:$ENVDIR/lib/python3.12/site-packages/av.libs"
export PYTHONPATH="$OPENPI_SRC/src:$OPENPI_SRC/packages/openpi-client/src"
export OPENPI_PI05_BASE="$BASE_PARAMS"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${PREALLOCATE:-true}"
PY="$ENVDIR/bin/python"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000000}"
KEEP_PERIOD="${KEEP_PERIOD:-100000000}"

[ -x "$PY" ] || { echo "FATAL: 找不到解释器 $PY"; exit 1; }
[ -d "$OPENPI_SRC/scripts" ] || { echo "FATAL: 找不到 $OPENPI_SRC/scripts"; exit 1; }

LOGD="$OUT/logs_5090"
CKPT="$OUT/ckpts_5090"
SUMMARY="$LOGD/_5090_summary.txt"
mkdir -p "$CKPT" "$LOGD"
cd "$OPENPI_SRC" || exit 1

# run_one <scheme> <tag> <b> <a> <steps> <fraction> [extra...]
run_one () {
  local scheme=$1 tag=$2 b=$3 a=$4 steps=$5 frac=$6; shift 6
  local extra="$*"
  rm -f "$LOGD/$tag.mem" "$LOGD/$tag.log"
  {
    echo "===== $tag : goai_pi05_$scheme  physical=$b accumulation=$a effective=$((b*a)) steps=$steps frac=$frac prealloc=$XLA_PYTHON_CLIENT_PREALLOCATE"
    echo "      extra: ${extra:-<none>}"
  } >> "$SUMMARY"

  XLA_PYTHON_CLIENT_MEM_FRACTION="$frac" nohup "$PY" scripts/train.py "goai_pi05_$scheme" \
    --exp-name "$tag" --checkpoint-base-dir "$CKPT" \
    --batch-size "$b" --gradient-accumulation-steps "$a" \
    --num-train-steps "$steps" --log-interval 1 \
    --save-interval "$SAVE_INTERVAL" --keep-period "$KEEP_PERIOD" \
    --ema-decay 0.999 \
    --no-wandb-enabled --overwrite $extra \
    > "$LOGD/$tag.log" 2>&1 &
  local pid=$!
  ( while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 1; done ) > "$LOGD/$tag.mem" &
  local samp=$!

  local t0=$SECONDS
  wait "$pid"; local rc=$?
  local wall=$((SECONDS - t0))
  kill "$samp" 2>/dev/null; wait "$samp" 2>/dev/null

  local peak
  peak=$(sort -n "$LOGD/$tag.mem" 2>/dev/null | tail -1)
  echo "rc=$rc wall=${wall}s peak_used_MiB=${peak:-NA}" >> "$SUMMARY"
  grep -E "Training batches:|EMA decay:|Gradient checkpointing:|Parameter audit:|EMA parameter audit:" \
    "$LOGD/$tag.log" 2>/dev/null | tail -5 >> "$SUMMARY"
  # 稳态耗时：排除首步（含 JAX 编译）与 checkpoint 保存步
  grep -E "Step [0-9]+ data timing" "$LOGD/$tag.log" 2>/dev/null | tail -1 >> "$SUMMARY"
  grep -E "^Step [0-9]+: loss=" "$LOGD/$tag.log" 2>/dev/null | tail -1 >> "$SUMMARY"

  if [ "$rc" -ne 0 ]; then
    grep -E "RESOURCE_EXHAUSTED|OutOfMemory|Out of memory" "$LOGD/$tag.log" 2>/dev/null | tail -1 >> "$SUMMARY"
    echo "RESULT: CRASHED" >> "$SUMMARY"
    return 1
  fi
  echo "RESULT: OK" >> "$SUMMARY"
  return 0
}

case "$MODE" in
  scan)
    : > "$SUMMARY"
    {
      echo "5090-SCAN (5090/32G) ema=0.999 eff=32 tagpfx=${TAGPFX:-scan}"
      echo "scan_extra=${SCAN_EXTRA---model.no-gradient-checkpointing}"
      echo "code=$(git rev-parse --short HEAD 2>/dev/null) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
      echo "started=$(date -Is)"
    } >> "$SUMMARY"
    # fail-stop：某档 OOM 即跳过该 scheme 的更大档（不预设任何档可行）
    for scheme in p0 p1; do
      for spec in "1 32" "2 16" "4 8" "8 4" "16 2" "32 1"; do
        set -- $spec; b=$1; a=$2
        tag="${TAGPFX:-scan}_${scheme}_b${b}a${a}"
        if run_one "$scheme" "$tag" "$b" "$a" 3 0.90 "${SCAN_EXTRA---model.no-gradient-checkpointing}"; then
          echo ">>> ${scheme} b${b}/a${a}: PASS（继续更大档）" >> "$SUMMARY"
        else
          echo ">>> ${scheme} b${b}/a${a}: FAIL — 该 scheme 更大的 physical batch 不再尝试" >> "$SUMMARY"
          break
        fi
      done
    done
    echo "SCAN_ALL_DONE $(date -Is)" >> "$SUMMARY"
    ;;
  run)
    shift
    [ "$#" -ge 6 ] || { echo "usage: $0 run <scheme> <tag> <b> <a> <steps> <fraction> [extra...]"; exit 2; }
    {
      echo "5080-RUN (5090/32G)"
      echo "code=$(git rev-parse --short HEAD 2>/dev/null) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
      echo "started=$(date -Is)"
    } >> "$SUMMARY"
    run_one "$@"
    echo "RUN_DONE $(date -Is)" >> "$SUMMARY"
    ;;
  *)
    echo "unknown mode: $MODE（用 scan 或 run）"; exit 2
    ;;
esac
