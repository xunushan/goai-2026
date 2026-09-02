#!/usr/bin/env python
"""Aggregate p07 sim h264 loader bench JSONs -> markdown table (median over 3 reps)."""
import argparse, glob, json, statistics, os


def load(globpat):
    rows = {}
    for fp in sorted(glob.glob(globpat)):
        tag = os.path.basename(fp).rsplit("_rep", 1)[0]
        d = json.load(open(fp))
        rows.setdefault(tag, []).append(d)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="/cloud/cloud-ssd1/lerobot_bench/out/p07/*.json")
    p.add_argument("--fmt", default="table")
    a = p.parse_args()
    rows = load(a.json)
    order = ["v44_joint640", "v44_joint224",
             "v60_joint640_float", "v60_joint640_uint8",
             "v60_joint224_float", "v60_joint224_uint8"]
    print(f"# p07 sim h264 loader bench (median of {len(rows[next(iter(rows))]) if rows else 0} reps/cell)")
    print("| config | env/lerobot | uint8 | samples/s | p50 ms/batch | p95 ms | peak RSS GB | reps |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tag in order:
        if tag not in rows:
            continue
        reps = rows[tag]
        sps = sorted(r["batch_stats"]["samples_per_s"] for r in reps)
        p50 = sorted(r["batch_stats"]["p50_s"] * 1e3 for r in reps)
        p95 = sorted(r["batch_stats"]["p95_s"] * 1e3 for r in reps)
        rss = sorted(r["batch_stats"]["peak_total_rss_gb"] for r in reps)
        med = lambda xs: statistics.median(xs)
        args = reps[0]["args"]
        env = "0.4.4" if "v44" in tag else "0.6.0"
        uint8 = "yes" if (args.get("return_uint8") or "uint8" in tag) else "no"
        res = " / ".join(f"{x:.1f}" for x in sps)
        print(f"| {tag} | {env} | {uint8} | **{med(sps):.1f}** ({res}) | {med(p50):.0f} | {med(p95):.0f} | {med(rss):.2f} | {len(reps)} |")


if __name__ == "__main__":
    main()
