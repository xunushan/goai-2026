#!/usr/bin/env python3
"""定位策略 rollout 中「哪个 action chunk 里会出现关键帧（抓取/释放）」。

方法沿用 [tools/xyz_gripper_segment.py](xyz_gripper_segment.py) 的
`segment_episode`（docs/xyz_gripper_event_segmentation.md §4~§9 的
夹爪锚点 → XYZ 反查 → grasp/place 区间），在事件区间上切 chunk 网格做命中判定。

为什么用 action 而不是 state：策略服务日志只在 chunk 边界上报 state，
`state_*` 每 30 帧才有一个采样点，逐帧的只有 `action_*`。仿真里
action_type=ee，action 的 16 维就是绝对末端位姿（xyz+四元数+夹爪），
与训练集 `observation.state` 同一坐标系同一布局。注意语义差异：这是
**命令**轨迹，不是实际执行轨迹。

三种命中口径：

  C. gripper  —— **默认**。chunk 内该臂夹爪的位移 max(g)-min(g) >= delta_min。
     直接对应「chunk 里要做动作 → 夹爪必然对齐并变化」这条判据，不依赖
     锚点检测的准确性。
  A. interval —— chunk 与 grasp/place 区间 [approach_start, confirm_end] 相交。
  B. moment   —— chunk 含事件帧 t0：抓取 t0 = close_end(hold_start)，
     释放 t0 = open_start(hold_end)。见 keyframe-weight skill §1.3。

**为什么默认改成口径 C（2026-09-18，用户验收后修订）**：口径 A 在 rollout 上
高饱和且含大量假阳性。区间 span 实测 24~73 帧，由三段构成——最终定位
（lead 7~19 帧）+ 夹爪行程 19~30 帧 + k_confirm 4 帧，天然跨 ≥1 个 chunk；
env 1 有 57% 的帧落在区间内、26/27 个 chunk 命中。更关键的是，区间的
**approach 段与 confirm 段夹爪完全不动**，落在这些段的 chunk 被判为命中属于
假阳性（用户实测指出 7 处，其 ΔGrip 仅 0.000~0.071）。改判据后这 7 处全部排除。

**锚点不可靠的实测证据**：在 rollout 命令轨迹上，检测器给出的 close_end 与真实
闭合终点可差 ±30 帧以上（env 1 c13/c21/c23/c25 夹爪实测位移 0.16~0.29 却落在
[close_start, close_end] 之外；env 2 c7 反之，窗口内 30 帧只动 0.069）。故口径 C
刻意不依赖锚点，事件区间只作为**归属标注**输出。

delta_min=0.10 的取值依据：夹爪实测行程 L/R 均约 [0.03, 1.00]，即满行程 ≈0.97，
0.10 相当于满行程的 ~10%。实测 111 个 chunk 的 max(dGrip) 排序在 0.081→0.094→
0.102→0.126→0.134 处有个自然稀疏带，0.10 落在 0.094/0.102 之间。边界样本
（ΔGrip 0.069~0.102，均只含 ≤5 帧行程的首/尾）会在清单里标 `edge`，供人工复核。

用法:
    python tools/policy_chunk_keyframe_locate.py \
        --csv outputs/simu_analysis/stack_bowls_policy_log.csv \
        --envs all --out outputs/simu_analysis/chunk_keyframes
    python tools/policy_chunk_keyframe_locate.py --csv ... --envs 1,5 --no-plot
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
DEFAULT_RESULT = ROOT / "outputs" / "simu_analysis" / "stack_bowls" / "_result.json"

# dataviz 调色板（同 tools/episode_state_insight.py，保持项目内图风格一致）
INK = "#0b0b0b"; MUT = "#898781"; GRID = "#e1e0d9"; SURF = "#fcfcfb"
C_XYZ = ("#2a78d6", "#eb6834", "#1baf7a")
C_LEFT = "#2a78d6"
C_RIGHT = "#eb6834"
C_GRASP = "#1baf7a"      # 抓取区间底色
C_PLACE = "#eb6834"      # 释放区间底色
C_ACTIVE = "#9467bd"     # 夹爪实际在动的帧

GRIP_IDX = {"left": 7, "right": 15}
EVENT_MOMENT_KEY = {"grasp": "close_end", "place": "open_start"}
EVENT_ACT_KEY = {"grasp": ("close_start", "close_end"), "place": ("open_start", "open_end")}


def build_action_array(grp: pd.DataFrame) -> np.ndarray:
    """该 env 的逐帧 action -> (T,16)，列序同 STATE_NAMES。"""
    g = grp.sort_values("frame_index")
    return g[[f"action_{n}" for n in STATE_NAMES]].to_numpy(dtype=float)


def active_mask(g: np.ndarray, rate_min: float) -> np.ndarray:
    """逐帧「夹爪在动」掩码（长度 = len(g)）：|g[t+1]-g[t]| >= rate_min。"""
    m = np.zeros(len(g), dtype=bool)
    if len(g) > 1:
        m[:-1] = np.abs(np.diff(g)) >= rate_min
    return m


def runs(mask: np.ndarray, offset: int) -> list[tuple[int, int]]:
    """把布尔掩码转成若干连续 [start, end] 闭区间（绝对帧号）。"""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((offset + s, offset + i)); s = None
    if s is not None:
        out.append((offset + s, offset + len(mask) - 1))
    return out


def chunk_windows(T: int, size: int) -> list[tuple[int, int]]:
    """chunk 网格：第 i 个 chunk = [i*size, min((i+1)*size, T) - 1]（闭区间）。"""
    return [(i * size, min((i + 1) * size, T) - 1) for i in range((T + size - 1) // size)]


def analyse_chunks(arr: np.ndarray, events: list[dict], size: int,
                   delta_min: float, rate_min: float) -> list[dict]:
    """逐 chunk 计算三种口径的命中情况。"""
    T = len(arr)
    out = []
    for ci, (a, b) in enumerate(chunk_windows(T, size)):
        # 口径 C: 夹爪实际位移（逐臂）
        d_grip, act, arm_best = {}, {}, None
        for side, gi in GRIP_IDX.items():
            g = arr[a:b + 1, gi]
            d_grip[side] = float(g.max() - g.min())
            act[side] = runs(active_mask(g, rate_min), a)
        arm_best = max(d_grip, key=lambda s: d_grip[s])
        hit_c = d_grip[arm_best] >= delta_min

        # 口径 A/B: 事件归属（仅作标注，不参与默认判定）
        iv, mo = [], []
        for ev in events:
            lo, hi = ev["approach_start"], ev["confirm_end"]
            t0 = ev["t0"]
            if lo <= b and a <= hi:
                ak, ek = EVENT_ACT_KEY[ev["event"]]
                ov = min(b, ev[ek]) - max(a, ev[ak]) + 1
                iv.append({**ev, "overlap": [max(a, lo), min(b, hi)],
                           "t0_in": bool(a <= t0 <= b), "act_overlap": max(0, ov)})
            if a <= t0 <= b:
                mo.append(ev)

        out.append({
            "chunk": ci, "start": a, "end": b, "T": T,
            "d_grip": d_grip, "arm": arm_best, "active": act,
            "hit": hit_c, "hit_interval": bool(iv), "hit_moment": bool(mo),
            "interval_hits": iv, "moment_hits": mo,
        })
    return out


def plot_chunk(arr: np.ndarray, rec: dict, meta: dict, size: int,
               delta_min: float, out_path: Path) -> None:
    """单个命中 chunk 的小图：夹爪 + 左右臂 xyz，标出动作帧与事件区间。"""
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

    dL, dR = rec["d_grip"]["left"], rec["d_grip"]["right"]
    ev_desc = "  ".join(
        f"{e['arm'][0].upper()}:{e['event']}"
        + (f"@t0={e['t0']}" if e["t0_in"] else f"(t0={e['t0']} out)")
        for e in rec["interval_hits"]) or "(no event attributed)"
    status = "PASS" if meta.get("success") else "FAIL"
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    fig.suptitle(
        f"{meta['task']}   |   env {meta['env']} / {meta['uuid']}  [{status}]"
        f"   |   chunk {rec['chunk']}  (frames {a}-{b} of {rec['T']})\n"
        f"dGrip  L={dL:.3f}  R={dR:.3f}   -> hit arm = {rec['arm']}  "
        f"(threshold {delta_min:.2f})    events: {ev_desc}",
        fontsize=10, y=0.985)

    for ax in axes:
        # 事件区间底色（含 approach/confirm 段，说明它们不参与判定）
        for e in rec["interval_hits"]:
            o = e["overlap"]
            ax.axvspan(o[0] - 0.5, o[1] + 0.5,
                       color=C_GRASP if e["event"] == "grasp" else C_PLACE,
                       alpha=0.10, lw=0)
            if e["t0_in"]:
                ax.axvline(e["t0"], color=INK, lw=1.0, ls="--", alpha=0.55)
        # 夹爪真正在动的帧（判定依据）
        for side, color in (("left", C_LEFT), ("right", C_RIGHT)):
            for s, e_ in rec["active"][side]:
                ax.axvspan(s - 0.5, e_ + 0.5, color=C_ACTIVE, alpha=0.22, lw=0)
        for edge in (0, size):
            gx = a + edge - 0.5
            if a - 1 <= gx <= b + 1:
                ax.axvline(gx, color=MUT, lw=0.8, ls=":", alpha=0.9)

    ax = axes[0]
    ax.plot(f, arr[a:b + 1, 7], color=C_LEFT, ls="-", lw=1.4, marker="o", ms=2.6, label="Left")
    ax.plot(f, arr[a:b + 1, 15], color=C_RIGHT, ls="--", lw=1.4, marker="o", ms=2.6, label="Right")
    ax.set_ylabel("gripper\n(0 closed ~ 1 open)")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8, ncol=2, loc="lower right")

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


def load_success(path: Path) -> dict[int, bool]:
    """从 RoboDojo _result.json 读 {env_idx: success}（layout_id 即 env_idx）。"""
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    return {int(k): bool(v.get("success")) for k, v in d.get("details", {}).items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="extract_policy_log_csv.py 输出的 CSV")
    ap.add_argument("--envs", default="all", help="env_idx，逗号分隔，或 all")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--chunk", type=int, default=None, help="chunk 长度，默认取配置 chunk_size")
    ap.add_argument("--task-index", type=int, default=2, help="任务编号（stack_bowls=2）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--result-json", default=str(DEFAULT_RESULT),
                    help="用于标注成功/失败的 _result.json")
    ap.add_argument("--no-plot", action="store_true", help="只出清单不出图")
    ap.add_argument("--pdf", action="store_true", help="额外合并一份 contact-sheet PDF 便于验收")
    args = ap.parse_args()

    ch_cfg = json.loads(Path(args.config).read_text())
    chunk_size = args.chunk or int(ch_cfg["chunk_size"])
    gcfg = ch_cfg.get("gripper_motion", {})
    delta_min = float(gcfg.get("delta_min", 0.10))
    rate_min = float(gcfg.get("rate_min", 0.01))

    seg_cfg = load_config()                      # 夹爪锚点 detect 参数沿用训练集配置
    params = {**task_params(seg_cfg, args.task_index), **ch_cfg["segment"]}

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    success = load_success(Path(args.result_json))
    envs = (sorted(int(x) for x in df["env_idx"].unique())
            if args.envs == "all" else [int(x) for x in args.envs.split(",")])

    manifest, stats, n_png, pdf_pages = [], [], 0, []
    for e in envs:
        grp = df[df["env_idx"] == e]
        if grp.empty:
            print(f"  ! env {e} 不存在，跳过"); continue
        uuid = str(grp["episode_uuid"].iloc[0]) if "episode_uuid" in grp else ""
        arr = build_action_array(grp)
        res = segment_episode(e, args.task_index, arr, np.arange(len(arr)), seg_cfg, params)
        events = [{**ev, "t0": int(ev[EVENT_MOMENT_KEY[ev["event"]]])} for ev in res["events"]]

        recs = analyse_chunks(arr, events, chunk_size, delta_min, rate_min)
        meta = {"task": str(grp["task"].iloc[0]), "env": e, "uuid": uuid,
                "T": len(arr), "success": success.get(e)}
        n_hit = sum(r["hit"] for r in recs)
        tag = {True: "成功", False: "失败", None: "未知"}[meta["success"]]
        print(f"\n=== env {e} {uuid} [{tag}]  T={meta['T']}  事件={len(events)}  "
              f"chunk={len(recs)}  命中(口径C)={n_hit} ({n_hit/len(recs)*100:.0f}%)  "
              f"| 口径A={sum(r['hit_interval'] for r in recs)} "
              f"口径B={sum(r['hit_moment'] for r in recs)} ===")
        lowT = [r for r in recs if r["hit_interval"] and not r["hit"]]
        if lowT:
            print(f"  [口径A 判命中但口径C 判否 — 用户指出的假阳性类] {len(lowT)} 个: "
                  + ", ".join(f"c{r['chunk']}(Δ{r['d_grip'][r['arm']]:.3f})" for r in lowT))

        stats.append({"env": e, "success": meta["success"], "T": meta["T"],
                      "n_chunks": len(recs), "n_hit": n_hit,
                      "hit_ratio": round(n_hit / len(recs), 4),
                      "n_interval_only": len(lowT)})

        for r in recs:
            if not r["hit"]:
                continue
            d = r["d_grip"][r["arm"]]
            edge = d < 2 * delta_min
            ev_txt = "|".join(
                f"{ev['arm']}:{ev['event']}:{ev['approach_start']}-{ev['confirm_end']}"
                f"@t0={ev['t0']}" for ev in r["interval_hits"])
            manifest.append({
                "env": e, "uuid": uuid, "success": meta["success"], "T": meta["T"],
                "chunk": r["chunk"], "start": r["start"], "end": r["end"],
                "arm": r["arm"], "dGrip_L": round(r["d_grip"]["left"], 4),
                "dGrip_R": round(r["d_grip"]["right"], 4), "dGrip": round(d, 4),
                "edge_case": edge, "hit_interval": r["hit_interval"],
                "hit_moment": r["hit_moment"], "n_events": len(r["interval_hits"]),
                "events": ev_txt,
            })
            if not args.no_plot:
                p = out_dir / f"ep{e}_{uuid}_c{r['chunk']:03d}_f{r['start']}-{r['end']}.png"
                plot_chunk(arr, r, meta, chunk_size, delta_min, p)
                n_png += 1
                pdf_pages.append(p)

    mf = out_dir / "chunk_keyframe_manifest.csv"
    if manifest:
        pd.DataFrame(manifest).to_csv(mf, index=False)
        print(f"\n清单 -> {mf}  ({len(manifest)} 行)")
    sf = out_dir / "chunk_keyframe_stats.csv"
    if stats:
        pd.DataFrame(stats).to_csv(sf, index=False)
        s = pd.DataFrame(stats)
        print("\n=== 成功 vs 失败 分组统计（口径 C，chunk_size="
              f"{chunk_size}, delta_min={delta_min}）===")
        g = s.groupby("success")["hit_ratio"].agg(["count", "mean", "min", "max"])
        for key, lab in ((True, "成功"), (False, "失败")):
            if key in g.index:
                r = g.loc[key]
                tot_h = s[s.success == key]["n_hit"].sum()
                tot_c = s[s.success == key]["n_chunks"].sum()
                print(f"  {lab}: {int(r['count'])} 集  命中占比 均值={r['mean']*100:.1f}%  "
                      f"范围 {r['min']*100:.0f}~{r['max']*100:.0f}%  "
                      f"合计 {int(tot_h)}/{int(tot_c)} chunk")
        print(f"  被排除的「爪夹无变化」chunk: "
              + "  ".join(f"{'成功' if r['success'] else '失败'}env{r['env']}={r['n_chunks']-r['n_hit']}"
                          f"({(r['n_chunks']-r['n_hit'])/r['n_chunks']*100:.0f}%)"
                          for r in stats))
        print(f"统计 -> {sf}")

    if args.pdf and pdf_pages:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        pdf = out_dir / "chunk_keyframes_contact_sheet.pdf"
        with PdfPages(pdf) as pp:
            for p in pdf_pages:
                img = mpimg.imread(p)
                fig = plt.figure(figsize=(img.shape[1] / 130, img.shape[0] / 130))
                ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
                pp.savefig(fig); plt.close(fig)
        print(f"PDF  -> {pdf}  ({len(pdf_pages)} 页)")

    print(f"图   -> {out_dir}  ({n_png} 张)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
