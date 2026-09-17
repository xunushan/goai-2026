#!/usr/bin/env python3
"""定位策略 rollout 中「哪个 action chunk 里会出现关键帧（抓取/释放）」。

方法直接复用 [tools/xyz_gripper_segment.py](xyz_gripper_segment.py) 的
`segment_episode`（即 docs/xyz_gripper_event_segmentation.md §4~§9 的
夹爪锚点 → XYZ 反查最终操作邻域 → grasp/place 区间），只是把输入换成
**rollout 的命令轨迹**，再在事件区间上切 chunk 网格做命中判定。

为什么用 action 而不是 state：策略服务日志只在 chunk 边界上报 state，
`state_*` 每 30 帧才有一个采样点，逐帧的只有 `action_*`。仿真里
action_type=ee，action 的 16 维就是绝对末端位姿（xyz+四元数+夹爪），
与训练集 `observation.state` 同一坐标系同一布局，故 §5/§6 的 XYZ 反查
可以直接跑在 action 上。注意语义差异：这是**命令**轨迹，不是实际执行轨迹。

两种命中口径（doc §14 用「事件标签」决定 chunk 长度，故以口径 A 为准）：

  A. interval —— chunk 与事件的 grasp/place 区间相交（该 chunk 内有帧被标成
     grasp/place，属于细粒度操作段）。**默认口径**。
  B. moment   —— chunk 含事件发生的**那一帧** t0：抓取 t0 = hold_start
     (close_end)，释放 t0 = hold_end (open_start)。见 keyframe-weight skill §1.3。

**实测（2026-09-18，stack_bowls 官方 ckpt-100000，env 1/5/3）**：口径 A 高饱和——
事件区间 span = 24~73 帧（中位约 32），因为区间含三段：最终定位（lead 7~19 帧，
标定良好）+ **夹爪行程本身 19~30 帧** + k_confirm 4 帧。故单个事件天然跨 ≥1 个
chunk，env 1 有 57% 的帧落在区间内、26/27 个 chunk 命中，几乎不具判别力。
口径 B 与事件一一对应（env 1: 14/27、env 5: 7/27、env 3: 6/21），更适合回答
「哪个 chunk 里有关键帧」。两者都输出，清单里 hit_interval / hit_moment 两列并列。

用法:
    python tools/policy_chunk_keyframe_locate.py \
        --csv outputs/simu_analysis/stack_bowls_policy_log.csv \
        --envs 1,5,3 --out outputs/simu_analysis/chunk_keyframes
    # 只输出清单不出图:
    python tools/policy_chunk_keyframe_locate.py --csv ... --envs 1 --no-plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.extract_policy_log_csv import STATE_NAMES  # noqa: E402
from tools.xyz_gripper_segment import (  # noqa: E402
    XYZ_SLICE, load_config, segment_episode, task_params,
)

DEFAULT_CONFIG = ROOT / "configs" / "policy_chunk_keyframe_config.json"

# dataviz 调色板（同 tools/episode_state_insight.py，保持项目内图风格一致）
INK = "#0b0b0b"; MUT = "#898781"; GRID = "#e1e0d9"; SURF = "#fcfcfb"
C_XYZ = ("#2a78d6", "#eb6834", "#1baf7a")
C_LEFT = "#2a78d6"
C_RIGHT = "#eb6834"
C_GRASP = "#1baf7a"      # 抓取区间底色
C_PLACE = "#eb6834"      # 释放区间底色

EVENT_MOMENT_KEY = {"grasp": "close_end", "place": "open_start"}


def build_action_array(grp: pd.DataFrame) -> np.ndarray:
    """该 env 的逐帧 action -> (T,16)，列序同 STATE_NAMES。"""
    g = grp.sort_values("frame_index")
    return g[[f"action_{n}" for n in STATE_NAMES]].to_numpy(dtype=float)


def chunk_windows(T: int, size: int) -> list[tuple[int, int]]:
    """chunk 网格：第 i 个 chunk = [i*size, min((i+1)*size, T) - 1]（闭区间）。"""
    return [(i * size, min((i + 1) * size, T) - 1) for i in range((T + size - 1) // size)]


def hit_chunks(events: list[dict], windows: list[tuple[int, int]]) -> list[dict]:
    """判定每个 chunk 的命中情况（两种口径）。"""
    out = []
    for ci, (a, b) in enumerate(windows):
        iv, mo = [], []
        for ev in events:
            lo, hi = ev["approach_start"], ev["confirm_end"]
            t0 = ev["t0"]
            if lo <= b and a <= hi:                       # 口径 A: 区间相交
                iv.append({**ev, "overlap": [max(a, lo), min(b, hi)],
                           "t0_in": bool(a <= t0 <= b), "t0": t0})
            if a <= t0 <= b:                             # 口径 B: 含事件帧
                mo.append(ev)
        out.append({"chunk": ci, "start": a, "end": b,
                    "interval_hits": iv, "moment_hits": mo,
                    "hit": bool(iv)})
    return out


def plot_chunk(grp: pd.DataFrame, arr: np.ndarray, rec: dict, meta: dict,
               chunk_size: int, out_path: Path) -> None:
    """单个命中 chunk 的小图：夹爪 + 左右臂 xyz，标出事件区间与事件帧 t0。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = rec["start"], rec["end"]
    f = np.arange(a, b + 1)
    plt.rcParams.update({
        "axes.edgecolor": MUT, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "grid.color": GRID,
        "font.family": "sans-serif", "figure.dpi": 110,
    })

    events = rec["interval_hits"]
    ev_desc = "  ".join(
        f"{e['arm'][0].upper()}:{e['event']}"
        + (f"@t0={e['t0']}" if e["t0_in"] else f"(t0={e['t0']} outside)")
        for e in events)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    fig.suptitle(
        f"{meta['task']}   |   env {meta['env']} / {meta['uuid']}"
        f"   |   chunk {rec['chunk']}  (frames {a}-{b} of {meta['T']})\n"
        f"hit events: {ev_desc}    [interval-overlap criterion]",
        fontsize=10, y=0.985)

    # 事件区间底色（所有面板共用，便于对齐阅读）
    for ax in axes:
        for e in events:
            o = e["overlap"]
            ax.axvspan(o[0] - 0.5, o[1] + 0.5,
                       color=C_GRASP if e["event"] == "grasp" else C_PLACE,
                       alpha=0.13, lw=0)
            if e["t0_in"]:
                ax.axvline(e["t0"], color=INK, lw=1.0, ls="--", alpha=0.55)
        # chunk 边界（窗口端点）用点线标出，方便对齐 30 帧网格
        for edge in (0, chunk_size):
            gx = a + edge - 0.5
            if a - 1 <= gx <= b + 1:
                ax.axvline(gx, color=MUT, lw=0.8, ls=":", alpha=0.9)

    # 面板 1: 夹爪
    ax = axes[0]
    ax.plot(f, arr[a:b + 1, 7], color=C_LEFT, ls="-", lw=1.4, marker="o", ms=2.6, label="Left")
    ax.plot(f, arr[a:b + 1, 15], color=C_RIGHT, ls="--", lw=1.4, marker="o", ms=2.6, label="Right")
    ax.set_ylabel("gripper\n(0 closed ~ 1 open)")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8, ncol=2, loc="lower right")

    # 面板 2/3: 左右臂 xyz（用 action 的绝对末端位姿）
    for ax, side, name in ((axes[1], "left", "Left arm"), (axes[2], "right", "Right arm")):
        sl = XYZ_SLICE[side]
        for d, c in enumerate(C_XYZ):
            ax.plot(f, arr[a:b + 1, sl][:, d], color=c, ls="-", lw=1.3,
                    marker="o", ms=2.6, label=f"{side[0].upper()} {'xyz'[d]}")
        ax.set_ylabel(f"{name}\nxyz (m)"); ax.legend(fontsize=8, ncol=3, loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frame index")
    axes[-1].set_xlim(a - 1, b + 1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="extract_policy_log_csv.py 输出的 CSV")
    ap.add_argument("--envs", required=True, help="要处理的 env_idx，逗号分隔，如 1,5,3")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--chunk", type=int, default=None,
                    help="chunk 长度，默认取配置 chunk_size")
    ap.add_argument("--task-index", type=int, default=2, help="任务编号（stack_bowls=2）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--no-plot", action="store_true", help="只出清单不出图")
    args = ap.parse_args()

    ch_cfg = json.loads(Path(args.config).read_text())
    chunk_size = args.chunk or int(ch_cfg["chunk_size"])
    seg_cfg = load_config()                      # 夹爪锚点 detect 参数沿用训练集配置
    params = {**task_params(seg_cfg, args.task_index), **ch_cfg["segment"]}

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    envs = [int(x) for x in args.envs.split(",")]

    manifest, n_png = [], 0
    for e in envs:
        grp = df[df["env_idx"] == e]
        if grp.empty:
            print(f"  ! env {e} 不存在，跳过"); continue
        uuid = str(grp["episode_uuid"].iloc[0]) if "episode_uuid" in grp else ""
        arr = build_action_array(grp)
        T = len(arr)
        res = segment_episode(e, args.task_index, arr, np.arange(T), seg_cfg, params)
        # 事件补充 t0（keyframe-weight skill §1.3 的关键帧定义）
        events = [{**ev, "t0": int(ev[EVENT_MOMENT_KEY[ev["event"]]])} for ev in res["events"]]

        windows = chunk_windows(T, chunk_size)
        recs = hit_chunks(events, windows)
        meta = {"task": str(grp["task"].iloc[0]), "env": e, "uuid": uuid, "T": T}
        n_hit = sum(r["hit"] for r in recs)
        n_mo = sum(bool(r["moment_hits"]) for r in recs)
        print(f"\n=== env {e} {uuid}  T={T}  事件={len(events)}  "
              f"chunk={len(windows)}  命中(区间口径)={n_hit}  (事件帧口径)={n_mo} ===")
        for r in recs:
            if not (r["hit"] or r["moment_hits"]):
                continue
            desc = "  ".join(
                f"{ev['arm'][0].upper()}:{ev['event']}[{ev['approach_start']}-{ev['confirm_end']}]"
                + (f" <t0={ev['t0']}>" if ev["t0_in"] else "")
                for ev in r["interval_hits"]) or "(仅事件帧)"
            mark = "HIT " if r["hit"] else "mom "
            print(f"  {mark} chunk {r['chunk']:3d}  frames [{r['start']:4d},{r['end']:4d}]  {desc}")
            manifest.append({"env": e, "uuid": uuid, "task": meta["task"], "T": T,
                             "chunk": r["chunk"], "start": r["start"], "end": r["end"],
                             "hit_interval": r["hit"],
                             "hit_moment": bool(r["moment_hits"]),
                             "events": [{"arm": ev["arm"], "event": ev["event"],
                                         "start": ev["approach_start"], "end": ev["confirm_end"],
                                         "t0": ev["t0"]} for ev in r["interval_hits"]]})
            if r["hit"] and not args.no_plot:
                p = out_dir / f"ep{e}_{uuid}_c{r['chunk']:03d}_f{r['start']}-{r['end']}.png"
                plot_chunk(grp, arr, r, meta, chunk_size, p)
                n_png += 1

    mf = out_dir / "chunk_keyframe_manifest.csv"
    if manifest:
        flat = [{k: v for k, v in m.items() if k != "events"} | {
            "n_events": len(m["events"]),
            "events": "|".join(f"{ev['arm']}:{ev['event']}:{ev['start']}-{ev['end']}@t0={ev['t0']}"
                               for ev in m["events"])} for m in manifest]
        pd.DataFrame(flat).to_csv(mf, index=False)
        print(f"\n清单 -> {mf}  ({len(manifest)} 行)")
    print(f"图   -> {out_dir}  ({n_png} 张)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
