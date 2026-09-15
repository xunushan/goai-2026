#!/usr/bin/env python3
"""基于末端 XYZ + 夹爪曲线的事件划分 (move / grasp / place)。

方法以 docs/xyz_gripper_event_segmentation.md 为准, 参数来自
configs/xyz_segment_config.json (不写死在代码里)。流程:

  ① 夹爪锚点   —— 复用 tools/keyframe_detect.detect_gripper_keyframes (§3.2):
                  close_start(开始闭合) / close_end(闭合结束=hold_start) /
                  open_start(开始张开=hold_end) / open_end(完全张开=open_full)
  ② XYZ 反查   —— find_approach_start (§6/§9.1): 以动作锚点处的末端位置为参考,
                  向前找「最后一次离开邻域(D_out)的下一帧」, 再要求其后连续 K_d 帧
                  落在 D_in 内; 回看不超过 M 帧。
  ③ 状态机     —— (§8) MOVE_EMPTY/GRASP_APPROACH/GRIP_CLOSING → move/grasp,
                  MOVE_HOLDING/PLACE_APPROACH/RELEASING → move/place。
                  抓取区间 = [grasp_start, close_end + K_confirm];
                  放置区间 = [place_start, open_end + K_confirm]。
                  open_full == INCOMPLETE(-1) 的周期无真实释放 → 不产生放置区间。

产物 (默认 outputs/xyz_segment_<slug>/):
  events_ep<EP>.json     §12 格式的事件区间 + 动作锚点 + 参考位置 (逐臂/逐事件)
  labels_ep<EP>.csv      逐帧 left_stage / right_stage (move|grasp|place)
  segment_ep<EP>.png     逐臂预览: 夹爪开度 + 到锚点的距离 + 三类标签色带
  boundary/              边界帧图片, 按时间命名 (取自三视角合成视频)
  boundary_sheet_ep<EP>.png  边界帧拼图 (标注臂/事件/边界名/时间)
  summary.json           参数 + 每 episode 事件数/区间长度统计

用法:
    python tools/xyz_gripper_segment.py --task 2 --episodes 200,201,202,203,204,205
    python tools/xyz_gripper_segment.py --episodes 200 --d-out 0.08 --out outputs/xyz_try
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import keyframe_events as ke  # noqa: E402
from tools import keyframe_detect as kd  # noqa: E402

ROOT = ke.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "xyz_segment_config.json"
DEFAULT_OUT = ROOT / "outputs" / "xyz_segment"
DEFAULT_VIDEO_DIR = ROOT / "outputs" / "episode_insight" / "interactive_sim" / "videos"

# state 16 维: [l_xyz(3) l_quat(4) l_grip(1) | r_xyz(3) r_quat(4) r_grip(1)]
XYZ_SLICE = {"left": slice(0, 3), "right": slice(8, 11)}
GRIP_IDX = {"left": ke.GRIP_L, "right": ke.GRIP_R}
SIDES = ("left", "right")
STAGES = ("move", "grasp", "place")
STAGE_COLOR = {"move": "#eceff1", "grasp": "#4c78a8", "place": "#e45756"}
STAGE_CN = {"move": "移动", "grasp": "抓取", "place": "放置"}

# 合成视频 (640x720) 分区: 上 640x480 = cam_high, 左下 320x240 = 左腕, 右下 = 右腕。
# 由 200-205 的区块运动能量与对应臂 |v| 的相关性确认
# (左下 vs 左臂 corr=0.58/0.64, 右下 vs 右臂 corr=0.36/0.71, 交叉项均 <=0)。
VIEW_BOX = {
    "cam_high": (slice(0, 480), slice(0, 640)),
    "left_wrist": (slice(480, 720), slice(0, 320)),
    "right_wrist": (slice(480, 720), slice(320, 640)),
}


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def load_config(path: Path | None = None) -> dict:
    return json.loads((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def task_params(config: dict, ti: int) -> dict:
    """合并 defaults 与 tasks.<ti> 覆盖, 返回 {grasp:{...}, place:{...}, k_confirm, slug, cn}。"""
    p = {k: dict(v) for k, v in config["defaults"].items() if isinstance(v, dict)}
    p["k_confirm"] = int(config["defaults"]["k_confirm"])
    p["slug"] = f"task{ti}"
    p["cn"] = f"task{ti}"
    for k, v in (config.get("tasks", {}).get(str(ti)) or {}).items():
        if isinstance(v, dict):
            p.setdefault(k, {}).update(v)
        else:
            p[k] = v
    return p


# ---------------------------------------------------------------------------
# §6 / §9.1  XYZ 反查最终操作邻域起点
# ---------------------------------------------------------------------------

def find_approach_start(
    xyz: np.ndarray,
    anchor: int,
    *,
    d_in: float,
    d_out: float,
    max_lookback: int,
    k_d: int,
) -> int:
    """从动作锚点向前反查最后一次进入最终操作邻域的帧。

    参考文档 docs/xyz_gripper_event_segmentation.md §6 迟滞 + §9.1 参考实现:
      1) 以 xyz[anchor] 为参考位置 p_ref, 在 [anchor-M, anchor] 上算距离 d_t;
      2) 取最后一个 d_t > D_out 的帧, 其下一帧为候选起点 (迟滞外阈, 防边界抖动);
      3) 自候选起点向后推进, 直到其后连续 K_d 帧均满足 d_t < D_in;
      4) 若窗口内始终无法满足, 退化为「最后一个 d_t > D_in 的帧的下一帧」;
         仍不满足则返回回看窗起点(§9.1 的 min 形式兜底)。
    """
    anchor = int(anchor)
    begin = max(0, anchor - int(max_lookback))
    d = np.linalg.norm(xyz[begin:anchor + 1] - xyz[anchor], axis=-1)
    n = len(d)
    if n <= 1:
        return begin

    def last_exit(thr: float) -> int:
        idx = np.flatnonzero(d > thr)
        return int(idx[-1]) + 1 if idx.size else 0

    s = last_exit(d_out)
    while s + k_d <= n and not bool((d[s:s + k_d] < d_in).all()):
        s += 1
    if s + k_d > n:
        s = last_exit(d_in)
        if s + k_d > n:
            s = 0
    return begin + min(int(s), n - 1)


# ---------------------------------------------------------------------------
# 状态机: 三标签
# ---------------------------------------------------------------------------

def segment_episode(
    episode_index: int,
    ti: int,
    state: np.ndarray,
    frame: np.ndarray,
    config: dict,
    params: dict | None = None,
) -> dict:
    """单 episode 事件划分。返回 {params, events, labels, per_side, T, frame}。

    params 非空时直接用作参数集 (供 CLI 覆盖), 否则由 config 按 task 推导。
    """
    p = params if params is not None else task_params(config, ti)
    detect_kwargs = dict(config.get("detect", {}))
    T = len(state)
    per_side = ke.detect_sides(state, frame, detect_kwargs)

    labels = {s: np.full(T, "move", dtype=object) for s in SIDES}
    events: list[dict] = []
    notes: list[str] = []

    for s in SIDES:
        xyz = state[:, XYZ_SLICE[s]]
        cycles = ke.side_cycles(per_side, s)
        for i, (t_cs, t_ce, t_os, t_oe) in enumerate(cycles):
            # ---- 抓取: 锚点 = close_start, 结束 = close_end + K_confirm ----
            g = p["grasp"]
            g_start = find_approach_start(xyz, t_cs, **g)
            g_end = min(T - 1, t_ce + p["k_confirm"])
            if g_start > g_end:
                notes.append(f"{s} cycle{i}: grasp 起点({g_start}) 晚于终点({g_end}), 已丢弃")
            else:
                labels[s][g_start:g_end + 1] = "grasp"
                events.append({
                    "episode_index": episode_index, "arm": s, "cycle": i, "event": "grasp",
                    "approach_start": int(g_start),
                    "close_start": int(t_cs),
                    "close_end": int(t_ce),
                    "confirm_end": int(g_end),
                    "reference_xyz": [round(float(v), 6) for v in xyz[t_cs]],
                    "approach_lead_frames": int(t_cs - g_start),
                })

            # ---- 放置: 锚点 = open_start; open_full 为哨兵(周期未回开)则不产生 ----
            if t_oe == ke.INCOMPLETE:
                notes.append(f"{s} cycle{i}: 周期未回开(open_full=INCOMPLETE), 无放置区间")
                continue
            q = p["place"]
            p_start = find_approach_start(xyz, t_os, **q)
            p_end = min(T - 1, t_oe + p["k_confirm"])
            if p_start <= g_end:                      # §9.2 冲突处理: 放置区间不得侵入抓取区间
                notes.append(f"{s} cycle{i}: place 起点({p_start}) <= grasp 终点({g_end}), "
                             f"已右移到 {g_end + 1}")
                p_start = g_end + 1
            if p_start > p_end:
                notes.append(f"{s} cycle{i}: place 起点({p_start}) 晚于终点({p_end}), 已丢弃")
                continue
            labels[s][p_start:p_end + 1] = "place"
            events.append({
                "episode_index": episode_index, "arm": s, "cycle": i, "event": "place",
                "approach_start": int(p_start),
                "open_start": int(t_os),
                "open_end": int(t_oe),
                "confirm_end": int(p_end),
                "reference_xyz": [round(float(v), 6) for v in xyz[t_os]],
                "approach_lead_frames": int(t_os - p_start),
            })

    return {"episode_index": episode_index, "task_index": ti, "params": p,
            "events": events, "labels": labels, "per_side": per_side,
            "T": T, "frame": frame, "state": state, "notes": notes}


def stage_shares(labels: dict, T: int) -> dict:
    return {s: {k: round(float((labels[s] == k).mean() * 100), 2) for k in STAGES}
            for s in SIDES}


def rle(seq: np.ndarray, max_runs: int = 14) -> str:
    """标签序列的行程编码, 如 'move[0,43) grasp[43,65) ...'。"""
    out, i = [], 0
    seq = list(seq)
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        out.append(f"{seq[i]}[{i},{j + 1})")
        i = j + 1
    return " ".join(out[:max_runs]) + (" ..." if len(out) > max_runs else "")


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.edgecolor": kd.MUT, "axes.labelcolor": kd.INK, "axes.titlecolor": kd.INK,
        "text.color": kd.INK, "xtick.color": kd.INK, "ytick.color": kd.INK,
        "figure.facecolor": kd.SURF, "axes.facecolor": kd.SURF, "grid.color": kd.GRID,
        "font.family": "sans-serif", "figure.dpi": 110,
        "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Heiti SC",
                            "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    return plt


def plot_segment(res: dict, task_name: str, out_path: Path) -> None:
    """4 行: [左臂 夹爪+距离] [左臂 标签带] [右臂 夹爪+距离] [右臂 标签带]。"""
    plt = _setup_mpl()
    T, labels, per_side = res["T"], res["labels"], res["per_side"]
    state, frame = res["state"], res["frame"]
    x = np.arange(T)

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 0.30, 1.5, 0.30]})
    fig.suptitle(
        f"{task_name}  episode {res['episode_index']}  ({T} frames)   "
        f"事件划分: 移动/抓取/放置   D_in={res['params']['grasp']['d_in']} "
        f"D_out={res['params']['grasp']['d_out']} "
        f"M={res['params']['grasp']['max_lookback']}/{res['params']['place']['max_lookback']} "
        f"K_d={res['params']['grasp']['k_d']} K_confirm={res['params']['k_confirm']}",
        fontsize=11, y=0.975)

    # keyframe_detect 的锚点名 → 本文档 §3.2 的锚点名
    anchors = {k: (c, ls, f"{k} = 文档 {cn}") for k, c, ls, cn in kd.KEYFRAME_STYLES
               for cn in [{"close_start": "close_start", "hold_start": "close_end",
                           "hold_end": "open_start", "open_full": "open_end"}[k]]}

    for row, s in ((0, "left"), (2, "right")):
        ax = axes[row]
        col = kd.C_LEFT if s == "left" else kd.C_RIGHT
        grip = state[:, GRIP_IDX[s]]
        ax.plot(x, grip, color=col, lw=1.8, label=f"{s} 夹爪开度")
        ax.set_ylim(-0.06, 1.12)
        ax.set_ylabel("夹爪开度", color=col)
        ax.tick_params(axis="y", labelcolor=col)
        ax.grid(True, lw=0.5)

        ax2 = ax.twinx()
        xyz = state[:, XYZ_SLICE[s]]
        for ev in [e for e in res["events"] if e["arm"] == s]:
            a = ev["approach_start"]
            anch = ev["close_start"] if ev["event"] == "grasp" else ev["open_start"]
            d = np.linalg.norm(xyz[a:anch + 1] - xyz[anch], axis=-1)
            ax2.plot(np.arange(a, anch + 1), d, color="#5f5f5f", lw=1.2, ls=":",
                     label=f"{s} 到锚点距离" if ev is res["events"][0] else None)
            ax2.axhline(res["params"][ev["event"]]["d_out"], color="#9e9e9e", lw=0.7, ls="--")
        ax2.set_ylabel("到锚点距离 (m)", color="#5f5f5f")
        ax2.tick_params(axis="y", labelcolor="#5f5f5f")
        ax2.set_ylim(bottom=0)

        for key, arr in per_side[s].items():
            color, ls, _ = anchors[key]
            for t0 in arr:
                if int(t0) < 0:
                    continue
                ax.axvline(int(t0), color=color, lw=1.1, ls=ls, alpha=0.85)
        for ev in [e for e in res["events"] if e["arm"] == s]:
            is_g = ev["event"] == "grasp"
            ax.axvspan(ev["approach_start"], ev["confirm_end"],
                       color=STAGE_COLOR[ev["event"]], alpha=0.13)
            ax.annotate(f"{'抓取' if is_g else '放置'} lead={ev['approach_lead_frames']}f",
                        xy=(ev["approach_start"], 1.02), fontsize=7.5, color=STAGE_COLOR[ev["event"]],
                        rotation=0, ha="left", va="bottom")
        handles = [plt.Line2D([], [], color=c, ls=ls, lw=1.4, label=lbl)
                   for c, ls, lbl in anchors.values()]
        handles += [plt.Line2D([], [], color=col, lw=1.8, label=f"{s} 夹爪开度"),
                    plt.Line2D([], [], color="#5f5f5f", lw=1.2, ls=":", label="到锚点距离")]
        ax.legend(handles=handles, loc="upper right", fontsize=7.5, ncol=3, framealpha=0.9)

        axb = axes[row + 1]
        vals = labels[s]
        for k in STAGES:
            idx = np.flatnonzero(vals == k)
            if idx.size:
                axb.broken_barh([(int(idx[0]), int(idx[-1]) - int(idx[0]) + 1)],
                                (0, 1), facecolors=STAGE_COLOR[k])
        for m in np.flatnonzero(vals[1:] != vals[:-1]) + 1:
            axb.axvline(int(m), color="#ffffff", lw=0.8)
        axb.set_ylim(0, 1)
        axb.set_yticks([])
        axb.set_ylabel(f"{s}\n标签", fontsize=8, rotation=0, ha="right", va="center")
        axb.grid(False)

    axes[3].set_xlabel("frame_index")
    handles = [plt.Line2D([], [], color=STAGE_COLOR[k], lw=8, label=STAGE_CN[k]) for k in STAGES]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    fig.savefig(out_path)
    plt.close(fig)


def plot_boundary_sheet(rows: list, out_path: Path, episode_index: int, fps: float) -> None:
    """边界帧拼图: 每行一个边界, 左=合成原图, 右=cam_high 放大。"""
    plt = _setup_mpl()
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(9.5, 2.45 * n),
                             gridspec_kw={"width_ratios": [0.62, 1.0]})
    axes = np.atleast_2d(axes)
    fig.suptitle(f"episode {episode_index} 边界帧 (共 {n} 个)", fontsize=11, y=0.999)
    for i, r in enumerate(rows):
        axes[i, 0].imshow(r["img"])
        axes[i, 0].set_ylabel("")
        axes[i, 1].imshow(r["img"][VIEW_BOX["cam_high"]])
        for j in (0, 1):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values():
                sp.set_color(STAGE_COLOR[r["event"]]); sp.set_linewidth(2.2)
        axes[i, 1].set_title(
            f"{r['side']}  {STAGE_CN[r['event']]}  {r['boundary']}   "
            f"t={r['frame'] / fps:.3f}s (frame {r['frame']})",
            fontsize=9, pad=3)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 合成视频取帧
# ---------------------------------------------------------------------------

def read_video_frames(path: Path, want: set[int]) -> dict:
    """解码 mp4, 返回 {frame_no: rgb ndarray}, 只保留 want 中的帧。"""
    import av
    out: dict[int, np.ndarray] = {}
    if not path.exists():
        return out
    with av.open(str(path)) as c:
        n = 0
        for f in c.decode(video=0):
            if n in want:
                out[n] = f.to_ndarray(format="rgb24")
            n += 1
            if n > max(want):
                break
    return out


def boundary_frames(events: list, T: int) -> list:
    """把事件区间展开为 (boundary_name, frame) 列表 (去重后按帧号排序)。"""
    bounds = {"grasp": [("approach_start", "approach_start"), ("close_start", "close_start"),
                        ("close_end", "close_end"), ("confirm_end", "confirm_end")],
              "place": [("approach_start", "approach_start"), ("open_start", "open_start"),
                        ("open_end", "open_end"), ("confirm_end", "confirm_end")]}
    rows = []
    for ev in events:
        for name, key in bounds[ev["event"]]:
            f = int(ev[key])
            if 0 <= f < T:
                rows.append({"side": ev["arm"], "event": ev["event"], "cycle": ev["cycle"],
                             "boundary": name, "frame": f})
    rows.sort(key=lambda r: (r["frame"], r["side"]))
    return rows


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default=None,
                    help="逗号分隔的 episode_index (全局编号, stack=200..299)")
    ap.add_argument("--task", type=int, default=None, help="task_index, 用于取参数/命名")
    ap.add_argument("--csv", default=str(ke.DEFAULT_CSV), help="训练集主表 (含 observation.state)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR),
                    help="三视角合成视频目录 (<slug>_<ep>.mp4)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--d-in", type=float, default=None, help="覆盖 grasp+place 的 D_in")
    ap.add_argument("--d-out", type=float, default=None, help="覆盖 grasp+place 的 D_out")
    ap.add_argument("--max-lookback", type=int, default=None, help="覆盖 grasp+place 的 M")
    ap.add_argument("--k-d", type=int, default=None, help="覆盖 grasp+place 的 K_d")
    ap.add_argument("--k-confirm", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="不导出边界帧图片")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--dataset-label-dir", default=None,
                    help="把「每 episode 左右臂锚点帧 + 逐帧三类标签」落盘到该数据集子目录供复用"
                         " (如 data/sim_lerobot_v30_ee/xyz_segment); 不给则只在 outputs/ 出图")
    args = ap.parse_args()

    if not args.episodes:
        raise SystemExit("请用 --episodes 指定 episode (如 --episodes 200,201,202,203,204,205)")

    config = load_config(Path(args.config))
    cfg_detect = dict(config.get("detect", {}))
    eps = [int(v) for v in args.episodes.split(",") if v.strip()]

    small = ke.load_state_cols(Path(args.csv))
    tasks = ke.load_tasks()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bnd_dir = out_dir / "boundary"
    bnd_dir.mkdir(exist_ok=True)

    summary = {"config": str(args.config),
               "detect": cfg_detect, "episodes": {}}
    ds_rows: list[dict] = []       # 每臂每周期的锚点表 (落数据集目录复用)
    ds_labels: list[pd.DataFrame] = []

    for ep in eps:
        state, frame = kd.episode_state(small, ep)
        if args.task is not None:
            ti = args.task
        else:
            ti = int(small[small["episode_index"] == ep]["task_index"].iloc[0])
        res = segment_episode(ep, ti, state, frame, config,
                              params=_apply_overrides(task_params(config, ti), args))

        slug = res["params"]["slug"]
        task_name = tasks.get(ti, f"task{ti}")
        fps = 25.0

        # ---- JSON (doc §12) ----
        (out_dir / f"events_ep{ep}.json").write_text(json.dumps({
            "episode_index": ep, "task_index": ti, "task_name": task_name,
            "num_frames": res["T"], "fps": fps, "arm": "both",
            "params": res["params"], "events": res["events"], "notes": res["notes"],
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        # ---- 逐帧标签 CSV ----
        pd.DataFrame({
            "episode_index": ep, "frame_index": frame,
            "timestamp": np.round(np.asarray(frame, dtype=float) / fps, 4),
            "left_stage": res["labels"]["left"], "right_stage": res["labels"]["right"],
        }).to_csv(out_dir / f"labels_ep{ep}.csv", index=False)

        # ---- 预览图 ----
        if not args.no_plot:
            plot_segment(res, task_name, out_dir / f"segment_ep{ep}.png")

        # ---- 边界帧图片 ----
        rows = boundary_frames(res["events"], res["T"])
        n_png = 0
        if not args.no_video:
            vid = Path(args.video_dir) / f"{slug}_{ep}.mp4"
            imgs = read_video_frames(vid, {r["frame"] for r in rows})
            if not imgs:
                print(f"  [warn] 未找到/无法解码合成视频: {vid} (跳过边界帧导出)")
            for r in rows:
                img = imgs.get(r["frame"])
                if img is None:
                    continue
                r["img"] = img
                name = (f"ep{ep}_t{r['frame'] / fps:07.3f}s_{r['side']}_"
                        f"{r['event']}_{r['boundary']}.png")
                _save_png(img, bnd_dir / name)
                n_png += 1
            keep = [r for r in rows if "img" in r]
            if keep:
                plot_boundary_sheet(keep, out_dir / f"boundary_sheet_ep{ep}.png", ep, fps)

        # ---- 数据集内锚点表 (供后续复用) ----
        ds_rows.extend(anchor_rows(res, fps))
        ds_labels.append(pd.DataFrame({
            "episode_index": ep, "frame_index": frame,
            "timestamp": np.round(np.asarray(frame, dtype=float) / fps, 4),
            "left_stage": res["labels"]["left"], "right_stage": res["labels"]["right"],
        }))

        # ---- 汇总 ----
        summary["episodes"][str(ep)] = {
            "task_index": ti, "num_frames": res["T"],
            "n_events": len(res["events"]),
            "grasp_lead_frames": [e["approach_lead_frames"] for e in res["events"]
                                  if e["event"] == "grasp"],
            "place_lead_frames": [e["approach_lead_frames"] for e in res["events"]
                                  if e["event"] == "place"],
            "stage_share_pct": stage_shares(res["labels"], res["T"]),
            "n_boundary_stills": n_png,
            "notes": res["notes"],
        }

        print(f"\n=== episode {ep}  task={ti} ({task_name})  T={res['T']} ===")
        for s in SIDES:
            print(f"  {s:5s} 标签: {rle(res['labels'][s])}")
            print(f"        {s:5s} 占比: " + "  ".join(
                f"{STAGE_CN[k]}={summary['episodes'][str(ep)]['stage_share_pct'][s][k]:5.1f}%"
                for k in STAGES))
        for e in res["events"]:
            key = "close_start" if e["event"] == "grasp" else "open_start"
            print(f"  {e['arm']:5s} {e['event']:5s} cyc{e['cycle']}: "
                  f"[{e['approach_start']:3d},{e['confirm_end']:3d}]  "
                  f"锚点 {key}={e[key]:3d}  lead={e['approach_lead_frames']:2d}f  "
                  f"ref_xyz={e['reference_xyz']}")
        for n in res["notes"]:
            print(f"  [note] {n}")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.dataset_label_dir:
        dl = Path(args.dataset_label_dir)
        dl.mkdir(parents=True, exist_ok=True)
        kinds = sorted({r["task_index"] for r in ds_rows})
        tag = "_".join(str(t) for t in kinds) if kinds else "none"
        anch = pd.DataFrame(ds_rows)
        anch.to_csv(dl / f"gripper_anchors_task{tag}.csv", index=False)
        labels = pd.concat(ds_labels, ignore_index=True).sort_values(
            ["episode_index", "frame_index"]).reset_index(drop=True)
        labels.to_csv(dl / f"stage_labels_task{tag}.csv", index=False)
        (dl / f"meta_task{tag}.json").write_text(json.dumps({
            "generated_from": "tools/xyz_gripper_segment.py",
            "method": "docs/xyz_gripper_event_segmentation.md",
            "config": str(args.config), "config_body": config,
            "episodes": eps, "task_indices": kinds, "fps": 25.0,
            "columns": {
                "gripper_anchors": "每臂每抓取周期一行的锚点帧 (close_start/close_end/"
                                   "open_start/open_end) + 本方案事件区间 (grasp_*/place_*) "
                                   "+ 参考位置; -1 = 不适用/未回开",
                "stage_labels": "逐帧三分类 left_stage/right_stage ∈ {move, grasp, place}",
            }}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n数据集标签目录: {dl}  "
              f"(gripper_anchors_task{tag}.csv: {len(anch)} 行锚点, "
              f"stage_labels_task{tag}.csv: {len(labels)} 帧)")

    print(f"产物目录: {out_dir}")


def anchor_rows(res: dict, fps: float) -> list[dict]:
    """把一个 episode 的事件实例摊平成「每臂每周期一行」的锚点表 (供数据集内复用)。

    列语义: close_start/close_end 见 doc §3.2 (close_end == 检测器的 hold_start);
            open_start/open_end 见 §3.2 (open_start == hold_end, open_end == open_full);
            grasp_start/grasp_end、place_start/place_end 为本方案的事件区间 (§4/§5);
            open_end = -1 表示周期未回开 (哨兵), 此时无放置区间。
    """
    T, frame = res["T"], np.asarray(res["frame"], dtype=float)
    rows = []
    for s in SIDES:
        for i, (t_cs, t_ce, t_os, t_oe) in enumerate(ke.side_cycles(res["per_side"], s)):
            grasps = [e for e in res["events"]
                      if e["arm"] == s and e["cycle"] == i and e["event"] == "grasp"]
            places = [e for e in res["events"]
                      if e["arm"] == s and e["cycle"] == i and e["event"] == "place"]
            g = grasps[0] if grasps else None
            q = places[0] if places else None
            rows.append({
                "episode_index": res["episode_index"], "task_index": res["task_index"],
                "side": s, "cycle": i,
                "close_start": int(t_cs), "close_end": int(t_ce),
                "open_start": int(t_os), "open_end": int(t_oe),
                "cycle_complete": int(t_oe != ke.INCOMPLETE),
                "close_start_s": round(float(frame[t_cs]) / fps, 3),
                "grasp_start": g["approach_start"] if g else -1,
                "grasp_end": g["confirm_end"] if g else -1,
                "grasp_lead_frames": g["approach_lead_frames"] if g else -1,
                "ref_xyz_grasp_x": g["reference_xyz"][0] if g else np.nan,
                "ref_xyz_grasp_y": g["reference_xyz"][1] if g else np.nan,
                "ref_xyz_grasp_z": g["reference_xyz"][2] if g else np.nan,
                "place_start": q["approach_start"] if q else -1,
                "place_end": q["confirm_end"] if q else -1,
                "place_lead_frames": q["approach_lead_frames"] if q else -1,
                "ref_xyz_place_x": q["reference_xyz"][0] if q else np.nan,
                "ref_xyz_place_y": q["reference_xyz"][1] if q else np.nan,
                "ref_xyz_place_z": q["reference_xyz"][2] if q else np.nan,
            })
    return rows


def _apply_overrides(params: dict, args: argparse.Namespace) -> dict:
    """把 CLI 的 D_in/D_out/M/K_d/K_confirm 覆盖套到参数集上 (调试阈值用)。"""
    for ev in ("grasp", "place"):
        for k in ("d_in", "d_out", "max_lookback", "k_d"):
            v = getattr(args, k)
            if v is not None:
                params[ev][k] = v
    if args.k_confirm is not None:
        params["k_confirm"] = args.k_confirm
    return params


def _save_png(img: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(img).save(path)


if __name__ == "__main__":
    main()
