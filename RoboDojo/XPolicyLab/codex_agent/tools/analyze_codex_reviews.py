#!/usr/bin/env python3
"""盘点 bridge 落盘的每一次决策：原因、延迟、失败类型。

数据源是 ``workspace/output/<episode_id>/rollout.jsonl``（``bridge/record.py``
每行一次请求）。归属判据照抄 bridge 的路由，不是事后猜的：

* **放行路径**（``bridge.py:416``）：``vla_review`` 非空、不是
  ``verify_previous``、且 chunk 内夹爪变化没超过 ``gripper_change_threshold``
  —— 直接吃下整个 VLA chunk，不问 Codex。这种行 ``ok=True`` 且
  ``decision=None``（``turn_record`` 的默认值）、延迟≈0。
* **其余行都问了 Codex**，原因二选一：
  - ``verify_previous``：上一次决策把 ``verify_next`` 置了 true，要求看新图核验
    抓取/释放的结果；
  - ``gripper_change``：chunk 内夹爪 max-min 超过阈值，即这一段里含抓取/释放。
* **失败行同样 ``decision=None``**，只能靠 ``ok=False`` 与放行区分。失败不写
  continuation，所以它后面那行按 ``gripper_change`` 归类。

``policy_timeout`` 是 bridge 自己的墙钟到期（``app_server.py:449``：新线程首轮
``--timeout-first-turn-s``，热轮次 ``--timeout-s``），不是接口报错。每次超时在
策略侧都会退化成 hold 一步。

用法::

    python3 tools/analyze_codex_reviews.py                 # 全部 episode
    python3 tools/analyze_codex_reviews.py 28795d9e        # 指定 episode
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "workspace" / "output"

PASS = "放行_不问Codex"
GRIPPER = "gripper_change"
VERIFY = "verify_previous"


def analyze(path: Path) -> dict:
    """读一个 episode 的 rollout.jsonl，返回逐行归属与汇总。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries: list[dict] = []
    reasons: collections.Counter = collections.Counter()
    failures: collections.Counter = collections.Counter()
    latencies: list[float] = []
    prev_verify = False

    for row in rows:
        decision = row.get("decision")
        ok = bool(row.get("ok"))
        latency_s = (row.get("latency_ms") or 0) / 1000
        if decision is None and ok:
            reason = PASS
            prev_verify = False
        else:
            reason = VERIFY if prev_verify else GRIPPER
            # 只认成功的决策：失败行没有 continuation 可继承
            prev_verify = bool(decision.get("verify_next")) if isinstance(decision, dict) else False
            if ok:
                latencies.append(latency_s)
            else:
                failures[str(row.get("error_kind") or "unknown")] += 1
        reasons[reason] += 1
        entries.append(
            {
                "request_id": row.get("request_id"),
                "turn_index": row.get("turn_index"),
                "reason": reason,
                "mode": (decision or {}).get("mode") if isinstance(decision, dict) else None,
                "latency_s": latency_s,
                "ok": ok,
                "error_kind": row.get("error_kind"),
                "error": row.get("error"),
            }
        )
    return {"episode": path.parent.name, "rows": entries, "reasons": reasons,
            "failures": failures, "latencies": latencies}


def report(result: dict) -> str:
    lines = [f"===== episode {result['episode']}  |  请求 {len(result['rows'])} 次 ====="]
    for entry in result["rows"]:
        lines.append(
            f"  {str(entry['request_id']):<22} turn={entry['turn_index']:<3} "
            f"{entry['reason']:<16} {str(entry['mode']):<5} "
            f"lat={entry['latency_s']:>6.1f}s ok={entry['ok']} "
            f"{entry['error_kind'] or ''} {(entry['error'] or '')[:60]}"
        )
    lines.append(f"  -- 归属: {dict(result['reasons'])}")
    lines.append(f"  -- 失败: {dict(result['failures']) or '无'}")
    latencies = sorted(result["latencies"])
    if latencies:
        lines.append(
            f"  -- Codex 成功调用 {len(latencies)} 次，延迟 min={latencies[0]:.1f}s "
            f"中位={latencies[len(latencies) // 2]:.1f}s max={latencies[-1]:.1f}s"
        )
    total = sum(latencies) + sum(
        e["latency_s"] for e in result["rows"] if not e["ok"]
    )
    lines.append(f"  -- 墙钟总计 {total:.0f}s（含失败等待）")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episodes", nargs="*", help="episode id；不给则分析 root 下全部")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"含各 episode 子目录的根（默认 {DEFAULT_ROOT}）")
    args = parser.parse_args()

    root: Path = args.root.expanduser().resolve()
    if args.episodes:
        paths = [root / episode / "rollout.jsonl" for episode in args.episodes]
    else:
        paths = sorted(root.glob("*/rollout.jsonl"), key=lambda p: p.stat().st_mtime)

    missing = [p for p in paths if not p.is_file()]
    for path in missing:
        print(f"[skip] 没有 rollout.jsonl: {path}")
    found = [p for p in paths if p.is_file()]
    if not found:
        print(f"[error] {root} 下没有可分析的 rollout.jsonl")
        return 1

    totals: collections.Counter = collections.Counter()
    for path in found:
        result = analyze(path)
        print(report(result))
        print()
        totals.update(result["reasons"])
        totals.update({f"失败/{k}": v for k, v in result["failures"].items()})

    if len(found) > 1:
        print(f"=== 合计 {len(found)} 个 episode ===")
        print(f"  {dict(totals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
