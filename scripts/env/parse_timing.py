#!/usr/bin/env python3
"""从 pi05 sweep 的 train 日志提取稳态单步耗时统计。

日志行格式（scripts/train.py）：
    Step <n> data timing: load=<ls> total=<ts> ratio=<r>% throughput=<sps> samples/s
其中 total == optimizer_step_seconds（一次 optimizer update 的 wall time，含全部 micro-batch）。

用法：
    python3 parse_timing.py <log> [--skip-steps 8,16] [--skip-first 1] [--json]

默认排除第 1 步（含 JAX 冷编译）以及 --skip-steps 指定的 checkpoint 保存步。
"""
import argparse
import json
import re
import statistics as st
import sys

LINE = re.compile(
    r"Step\s+(\d+)\s+data timing:\s+load=([\d.]+)s\s+total=([\d.]+)s\s+ratio=([\d.]+)%\s+throughput=([\d.]+)"
)
LOSS = re.compile(r"^Step\s+(\d+):\s+(.*)$")


def parse(path: str):
    rows = []
    losses = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = LINE.search(line)
            if m:
                rows.append(
                    {
                        "step": int(m.group(1)),
                        "load": float(m.group(2)),
                        "total": float(m.group(3)),
                        "ratio": float(m.group(4)),
                        "throughput": float(m.group(5)),
                    }
                )
                continue
            m = LOSS.match(line)
            if m:
                losses[int(m.group(1))] = m.group(2).strip()
    return rows, losses


def pct(values, q):
    """线性插值分位数（不依赖 numpy）。"""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--skip-first", type=int, default=1, help="跳过前 N 步（默认 1，含冷编译）")
    ap.add_argument("--skip-steps", default="", help="额外跳过的步号，逗号分隔（checkpoint 保存步）")
    ap.add_argument("--eff-batch", type=int, default=32, help="等效 batch，用于换算吞吐/epoch")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    rows, losses = parse(a.log)
    if not rows:
        print(f"没有可用的 data timing 行：{a.log}（日志是否跑完/是否被 OOM 打断？）", file=sys.stderr)
        return 1

    skip = {int(x) for x in a.skip_steps.split(",") if x.strip()}
    kept = [r for r in rows if r["step"] > a.skip_first and r["step"] not in skip]
    if not kept:
        print(f"过滤后无剩余样本（seen={len(rows)}, skip_first={a.skip_first}, skip={sorted(skip)}）", file=sys.stderr)
        return 1

    totals = [r["total"] for r in kept]
    loads = [r["load"] for r in kept]
    ratios = [r["ratio"] for r in kept]
    mean = st.fmean(totals)

    out = {
        "log": a.log,
        "steps_seen": len(rows),
        "steps_used": len(kept),
        "skipped": sorted({r["step"] for r in rows} - {r["step"] for r in kept}),
        "total_mean_s": round(mean, 3),
        "total_p50_s": round(pct(totals, 0.50), 3),
        "total_p95_s": round(pct(totals, 0.95), 3),
        "total_min_s": round(min(totals), 3),
        "total_max_s": round(max(totals), 3),
        "load_mean_s": round(st.fmean(loads), 3),
        "load_ratio_mean_pct": round(st.fmean(ratios), 1),
        "throughput_samples_per_s": round(a.eff_batch / mean, 2),
        "per_epoch_min": round(mean * 3926 / 60, 1),
        "has_nan_loss": any(("nan" in v.lower() or "inf" in v.lower()) for v in losses.values()),
    }

    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print(f"日志: {a.log}")
    print(f"  计入 {out['steps_used']} / 共 {out['steps_seen']} 步；跳过 {out['skipped']}")
    print(
        f"  total(optimizer step)  mean={out['total_mean_s']}s  P50={out['total_p50_s']}s  "
        f"P95={out['total_p95_s']}s  min={out['total_min_s']}s  max={out['total_max_s']}s"
    )
    print(f"  load  mean={out['load_mean_s']}s  占 total {out['load_ratio_mean_pct']}%")
    print(f"  吞吐  {out['throughput_samples_per_s']} samples/s（等效 batch {a.eff_batch}）")
    print(f"  推算  {out['per_epoch_min']} min/epoch（3926 optimizer steps）")
    print(f"  loss 出现 NaN/Inf: {out['has_nan_loss']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
