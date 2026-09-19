#!/usr/bin/env bash
# 5090 冒烟第二阶段：连续跑完剩余全部项，进程之间无空隙（GPU 一直有活）。
#   [1] remat ON 边界扫描（fail-stop，对照 remat OFF 已完成的矩阵）
#   [2] 边界档 50 步稳态计时（OFF/ON × P0/P1）
#   [3] 边界档 MEM_FRACTION=0.85 余量验证（证明留 >10% 显存余量）
#   [4] 边界档 checkpoint 保存验证（§5.2 欠账）
set +e
S=/data/pi05_env_scripts/run_5090_sweep.sh
export OUT=/cloud/cloud-ssd1/pi05_smoke_5090
export SAVE_INTERVAL=100000000 KEEP_PERIOD=100000000
LOG=$OUT/logs_5090
SM=$LOG/_5090_summary.txt
DRV=$LOG/_phase2_driver.log
mkdir -p "$LOG"
: > "$DRV"
say() { echo "[$(date -Is)] $*" | tee -a "$DRV"; }

# 保存 remat OFF 阶段的扫描结果（scan 模式会截断 summary）
cp "$SM" "$LOG/_5090_summary_rematOFF.txt" 2>/dev/null

# 从 summary 里取某 scheme 的最后一个 PASS 档（fail-stop → 即边界）
boundary() {
  awk -v s="$1" '
    /^>>> / {
      if ($2 == s && $4 ~ /^PASS/) { gsub(/^b|a|:/, "", $3); split($3, p, "/"); b = p[1] " " p[2] }
    }
    END { if (b != "") print b; else print "1 32" }' "$SM"
}

say "=== [1] remat ON 边界扫描 ==="
TAGPFX=scanON SCAN_EXTRA= bash "$S" scan
P0ON=$(boundary p0); P1ON=$(boundary p1)
say "rematON 边界: p0='$P0ON' p1='$P1ON'"

# remat OFF 阶段已确认的边界（来自第一轮扫描）
P0OFF="2 16"
P1OFF="1 32"

say "=== [2] 边界档 50 步稳态计时 ==="
run_timing() {  # scheme tag b a extra
  say "timing $2 (b$3/a$4 $5)"
  bash "$S" run "$1" "$2" "$3" "$4" 50 0.90 $5
}
run_timing p0 t50_p0off_b2a16 $P0OFF "--model.no-gradient-checkpointing"
run_timing p1 t50_p1off_b1a32 $P1OFF "--model.no-gradient-checkpointing"
run_timing p0 t50_p0on_b${P0ON% *}a${P0ON#* } $P0ON ""
run_timing p1 t50_p1on_b${P1ON% *}a${P1ON#* } $P1ON ""

say "=== [3] 边界档 0.85 余量验证 ==="
run_margin() {  # scheme tag b a extra
  say "margin0.85 $2"
  bash "$S" run "$1" "$2" "$3" "$4" 3 0.85 $5
}
run_margin p0 m85_p0off_b2a16 $P0OFF "--model.no-gradient-checkpointing"
run_margin p1 m85_p1off_b1a32 $P1OFF "--model.no-gradient-checkpointing"
run_margin p0 m85_p0on_b${P0ON% *}a${P0ON#* } $P0ON ""
run_margin p1 m85_p1on_b${P1ON% *}a${P1ON#* } $P1ON ""

say "=== [4] checkpoint 保存验证（边界档，第 2 步存盘）==="
SAVE_INTERVAL=2 KEEP_PERIOD=2 bash "$S" run p0 ckpt_p0off_b2a16 $P0OFF 3 0.90 --model.no-gradient-checkpointing
SAVE_INTERVAL=2 KEEP_PERIOD=2 bash "$S" run p0 ckpt_p0on_b${P0ON% *}a${P0ON#* } $P0ON 3 0.90

say "PHASE2_ALL_DONE"
