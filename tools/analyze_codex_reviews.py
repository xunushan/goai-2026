#!/usr/bin/env python3
"""盘点 bridge 落盘的每一次决策：原因、延迟、失败类型。

数据源是 ``workspace/output/<episode_id>/rollout.jsonl``（``bridge/record.py``
每行一次请求）。归属判据照抄 bridge 的路由，不是事后猜的：

* **放行路径**（``bridge.py:448``）：``vla_review`` 非空、不是
  ``verify_previous``、且 chunk 内夹爪变化没超过 ``gripper_change_threshold``
  —— 直接吃下整个 VLA chunk，不问 Codex。这种行 ``ok=True`` 且
  ``decision=None``（``turn_record`` 的默认值）、延迟≈0。
* **其余行都问了 Codex**，原因二选一：
  - ``verify_previous``：上一次决策把 ``verify_next`` 置了 true，要求看新图核验
    抓取/释放的结果；
  - ``gripper_change``：chunk 内夹爪 max-min 超过阈值，即这一段里含抓取/释放。
* **失败行同样 ``decision=None``**，只能靠 ``ok=False`` 与放行区分。失败不写
  continuation，所以它后面那行按 ``gripper_change`` 归类。

**2026-09-20 起有一处口径变化**（``e974d55`` gate follow-up review on executed
gripper events）：bridge 现在只在「模型要求 ``verify_next`` **且** 实际下发的
prefix 里真有夹爪变化」时才把 ``verify_previous`` 传回去（``bridge.py:345``
``_executed_prefix_has_gripper_change``，判据是被执行的那段 chunk，不是整个
proposal）。所以此后：

* 本工具给出的 ``verify_previous`` 是**模型要求核验**的次数，不等于实际按核验
  路由的次数——被 gate 压掉的那些（模型要求了、但下发的 prefix 只是移 EEF 或
  提前停住）实际按 ``gripper_change`` 走，报告里会单独提示这一批的条数；
* ``gripper_change`` 与放行数、以及 Codex 调用总数不受影响（``verify_next``
  为假时本就没有核验可言，那部分分类是精确的）；
* rollout 记录里没有 chunk（``decision`` 只有 ``mode/note/phase/verify_next/
  vla_steps``，``observation_state`` 只有两臂位姿与夹爪），所以这是个**不可
  事后复原的**信息——要精确就得让 bridge 把下发的 continuation 也写进记录。

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

DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "RoboDojo/XPolicyLab/codex_agent/workspace/output"
)

PASS = "放行_不问Codex"
GRIPPER = "gripper_change"
VERIFY = "verify_previous"

# e974d55 "gate follow-up review on executed gripper events" 的提交时刻（+0800 10:34:32）。
# 此前的 rollout 里 verify_previous 就是实际路由；此后是「模型要求核验」，其中一部分
# 被 gate 改判成 gripper_change，记录里无法复原，只报条数。见模块 docstring。
GATE_UTC = "2026-09-20T02:34:32"


def analyze(path: Path) -> dict:
    """读一个 episode 的 rollout.jsonl，返回逐行归属与汇总。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries: list[dict] = []
    reasons: collections.Counter = collections.Counter()
    failures: collections.Counter = collections.Counter()
    latencies: list[float] = []
    prev_verify = False
    verify_after_gate = 0

    for row in rows:
        decision = row.get("decision")
        ok = bool(row.get("ok"))
        latency_s = (row.get("latency_ms") or 0) / 1000
        if decision is None and ok:
            reason = PASS
            prev_verify = False
        else:
            reason = VERIFY if prev_verify else GRIPPER
            if reason == VERIFY and str(row.get("recorded_at") or "")[:19] >= GATE_UTC:
                verify_after_gate += 1
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
            "failures": failures, "latencies": latencies,
            "verify_after_gate": verify_after_gate}


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
    if result["verify_after_gate"]:
        lines.append(
            f"  -- ⚠️ 其中 verify_previous {result['verify_after_gate']} 行发生在 e974d55 "
            f"（{GATE_UTC}Z）之后：那是模型**要求**核验的次数，被 gate 压掉的那部分实际走 "
            f"gripper_change，记录里无法区分（见 docstring）"
        )
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
