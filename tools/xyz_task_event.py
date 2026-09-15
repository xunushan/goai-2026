#!/usr/bin/env python3
"""第 2 层：任务级事件映射 (doc §12) —— 每帧输出唯一任务标签。

第 1 层 (tools/xyz_gripper_segment.py) 只产出任务无关的单臂五状态
{move, grasp, place, hold, idle}。本模块在其上做任务级映射

    E_t^task = F_task(s_t^left, s_t^right, c_t)

c_t = 持物历史 (holding) + 事件区间 (grasp/place 的起止帧, 取自 anchors 表)。
映射规则全部来自 docs/xyz_gripper_event_segmentation.md §12, 代码里不加任务知识:

  §12.1 fill_pen_holder  —— 角色 (持筒臂/笔臂) 由 episode 自动判定;
                            优先级 笔臂 grasp/place > 笔臂 move/hold(持笔) >
                            持筒臂 grasp/place > 持筒臂 move/hold(持筒) > move > idle
  §12.2 plug_in_charger  —— 优先级 handover > insert > grasp_plug/place_plug/hold_plug
                            > move > idle
                            handover = 递交侧 place ∪ 接取侧 grasp 的**并集**
                            insert   = 最后一个非交接、且在末次 handover 之后的 place
  §12.3 stack_bowl       —— 优先级 place_bowl > grasp_bowl > move_bowl > hold_bowl
                            > move > idle (不按左右臂设固定优先级)

★ fill 角色是数据事实, 不是配置常量: doc §12.1 的示例配置把职责写死成
  left=holder / right=pen, 但 sim_lerobot_v30_ee 的 100 个 fill episode 中
  46 个是左臂持筒、54 个是右臂持筒。故此处复用关键帧权重管线同一套判定
  ke.resolve_fill_roles (保持跨度最大的臂为持筒臂), 而不是照抄配置里的固定映射。

产物: 把 left_state / right_state / task_event 三列按 (episode_index, frame_index)
合并进 data/sim_lerobot_v30_ee/frame_weight.csv, 原有列原样不动。

用法:
    python tools/xyz_task_event.py --dry-run          # 只统计, 不写文件
    python tools/xyz_task_event.py                    # 写回 frame_weight.csv
    python tools/xyz_task_event.py --tasks 1 --out-csv outputs/x.csv
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

ROOT = ke.ROOT
SEG_DIR = ROOT / "data" / "sim_lerobot_v30_ee" / "xyz_segment"
FRAME_WEIGHT = ROOT / "data" / "sim_lerobot_v30_ee" / "frame_weight.csv"
DEFAULT_OUT = ROOT / "outputs" / "xyz_task_event"

SIDES = ("left", "right")
K_GAP = 25              # §12.2 条件 1 允许的配对间隔 (帧)
NEW_COLS = ("left_state", "right_state", "task_event")

# 任务级标签的中文名 (仅供展示)
EVENT_CN = {
    "move": "移动", "idle": "静止",
    "grasp_holder": "抓笔筒", "place_holder": "放笔筒",
    "hold_holder": "持笔筒静止", "move_holder": "运笔筒",
    "grasp_pen": "抓笔", "place_pen": "放笔",
    "hold_pen": "持笔静止", "move_pen": "运笔",
    "handover": "交接", "insert": "插入",
    "grasp_plug": "抓插头", "place_plug": "放插头", "hold_plug": "持插头静止",
    "grasp_bowl": "抓碗", "place_bowl": "放碗",
    "hold_bowl": "持碗静止", "move_bowl": "运碗",
}


# ---------------------------------------------------------------------------
# §12.2 交接配对 / 插入判定
# ---------------------------------------------------------------------------

def handover_pairings(anchors_ep: pd.DataFrame, k_gap: int = K_GAP) -> list[dict]:
    """枚举一臂 place × 另一臂 grasp 的交接候选配对 (§12.2)。

    条件 1: 两区间有重叠, 或间隔 <= k_gap;
    条件 2: 接取侧 close_end <= 递交侧 open_end (接取手先闭合接稳, 递交手才完成打开);
    条件 3: 双臂末端距离 < D_handover —— doc 未给出数值阈值, 故不参与筛选
            (实测窗口内最近逼近 p10/p50/p90 = 0.224/0.285/0.302 m)。
    """
    out = []
    for A in SIDES:
        B = "right" if A == "left" else "left"
        for _, pa in anchors_ep[(anchors_ep.side == A) & (anchors_ep.place_start >= 0)].iterrows():
            for _, gb in anchors_ep[(anchors_ep.side == B) & (anchors_ep.grasp_start >= 0)].iterrows():
                p0, p1 = int(pa.place_start), int(pa.place_end)
                g0, g1 = int(gb.grasp_start), int(gb.grasp_end)
                if max(g0 - p1, p0 - g1) > k_gap:
                    continue
                if int(gb.close_end) > int(pa.open_end):
                    continue
                out.append({
                    "releaser": A, "receiver": B,
                    "releaser_cycle": int(pa.cycle), "receiver_cycle": int(gb.cycle),
                    "place": (p0, p1), "grasp": (g0, g1),
                    "u": (min(p0, g0), max(p1, g1)),
                    "overlap": max(0, min(p1, g1) - max(p0, g0)),
                })
    return sorted(out, key=lambda d: d["u"][0])


def handover_mask(pairs: list[dict], T: int) -> np.ndarray:
    """§12.2 交接范围 = place ∪ grasp 的并集区间 (不是两者的交集)。"""
    m = np.zeros(T, dtype=bool)
    for pr in pairs:
        m[pr["u"][0]: pr["u"][1] + 1] = True
    return m


def insert_event(anchors_ep: pd.DataFrame, pairs: list[dict],
                 k_gap: int = K_GAP) -> dict | None:
    """§12.2 插入 = 最后一个非交接、且在末次 handover 之后的 place 事件。

    返回 {"side","cycle","place_start","place_end","open_end"} 或 None;
    多个候选时取 place_end 最大者 (doc: 每条成功轨迹最终只插入一次)。
    """
    done = {(pr["releaser"], pr["releaser_cycle"]) for pr in pairs}
    last_ho = max((pr["u"][1] for pr in pairs), default=-1)
    cand = []
    for _, r in anchors_ep[anchors_ep.place_start >= 0].iterrows():
        if (r.side, int(r.cycle)) in done:
            continue
        if int(r.place_start) <= last_ho:       # 必须在末次交接之后
            continue
        cand.append(r)
    if not cand:
        return None
    r = max(cand, key=lambda x: int(x.place_end))
    return {"side": r.side, "cycle": int(r.cycle),
            "place_start": int(r.place_start), "place_end": int(r.place_end),
            "open_end": int(r.open_end), "n_candidates": len(cand)}


# ---------------------------------------------------------------------------
# §12 逐帧映射
# ---------------------------------------------------------------------------

def map_fill_pen_holder(left, right, left_holding, right_holding, holder) -> str:
    """§12.1 参考实现 (doc 里把 left 写死为 holder, 此处角色作为入参)。

    doc 的深层顺序: 笔臂 "place(持笔) > grasp > move(持笔) > hold(持笔)" 全部先于
    持筒臂, 因为右臂只有在抓笔/放笔/已持笔时才覆盖左臂的稳定持筒状态。
    """
    pen = "right" if holder == "left" else "left"
    st = {"left": left, "right": right}
    hd = {"left": left_holding, "right": right_holding}
    for arm, kind in ((pen, "pen"), (holder, "holder")):
        s = st[arm]
        if s == "place" and hd[arm]:
            return f"place_{kind}"
        if s == "grasp":
            return f"grasp_{kind}"
        if s == "move" and hd[arm]:
            return f"move_{kind}"
        if s == "hold" and hd[arm]:
            return f"hold_{kind}"
    return "move" if "move" in (left, right) else "idle"


def map_plug_base(left, right, left_holding, right_holding) -> str:
    """§12.2 非交接、非插入帧的单臂物体语义映射。

    优先级参照 §12.3 的"更精细的操作事件优先": place_plug > grasp_plug > hold_plug。
    move 不带物体后缀 (doc §12.2 只列出 grasp_plug/place_plug/hold_plug, §14 的
    move_* 分组里也没有 move_plug), 故运输插头仍输出 move。
    """
    st, hd = {"left": left, "right": right}, {"left": left_holding, "right": right_holding}
    for s in ("place", "grasp"):
        for arm in SIDES:
            if st[arm] == s:
                return f"{s}_plug"
    for arm in SIDES:
        if st[arm] == "hold" and hd[arm]:
            return "hold_plug"
    return "move" if "move" in (left, right) else "idle"


def map_stack_bowl(left, right, left_holding, right_holding) -> str:
    """§12.3 叠碗映射。对左右臂对称, 优先级 place_bowl > grasp_bowl > move_bowl
    > hold_bowl > move > idle (不按左右臂设固定优先级)。

    move_bowl 仅在"当前确实持碗"时输出 —— 空手移动不能带物体后缀 (doc §12.1)。
    """
    st, hd = {"left": left, "right": right}, {"left": left_holding, "right": right_holding}
    for s in ("place", "grasp"):
        for arm in SIDES:
            if st[arm] == s:
                return f"{s}_bowl"
    for arm in SIDES:
        if st[arm] == "move" and hd[arm]:
            return "move_bowl"
    for arm in SIDES:
        if st[arm] == "hold":
            return "hold_bowl"
    return "move" if "move" in (left, right) else "idle"


def per_side_from_anchors(anchors_ep: pd.DataFrame) -> dict:
    """把 anchors 表的四列还原成 detector 的 per_side 结构, 供 resolve_fill_roles 复用。

    映射: anchors.close_start -> close_start; anchors.close_end -> hold_start;
          anchors.open_start  -> hold_end;    anchors.open_end   -> open_full。
    """
    out = {}
    for s in SIDES:
        a = anchors_ep[anchors_ep.side == s].sort_values("cycle")
        out[s] = {"close_start": a.close_start.to_numpy(dtype=int),
                  "hold_start": a.close_end.to_numpy(dtype=int),
                  "hold_end": a.open_start.to_numpy(dtype=int),
                  "open_full": a.open_end.to_numpy(dtype=int)}
    return out


def layer2_episode(ti: int, states_ep: pd.DataFrame, anchors_ep: pd.DataFrame,
                   k_gap: int = K_GAP) -> tuple[np.ndarray, dict]:
    """一个 episode 的第 2 层映射 -> (逐帧 task_event, 上下文信息)。

    states_ep 需按 frame_index 升序, 列含 left_state/right_state/left_holding/right_holding。
    """
    T = len(states_ep)
    left = states_ep.left_state.to_numpy()
    right = states_ep.right_state.to_numpy()
    lh = states_ep.left_holding.to_numpy()
    rh = states_ep.right_holding.to_numpy()
    ev = np.empty(T, dtype=object)
    info: dict = {"task_index": ti, "num_frames": T}

    if ti == 0:
        roles = ke.resolve_fill_roles(np.zeros(T), per_side_from_anchors(anchors_ep))
        holder = roles["holder"]
        info["holder_side"] = holder
        info["pen_side"] = roles["pen"]
        for t in range(T):
            ev[t] = map_fill_pen_holder(left[t], right[t], lh[t], rh[t], holder)

    elif ti == 1:
        pairs = handover_pairings(anchors_ep, k_gap)
        hmask = handover_mask(pairs, T)
        ins = insert_event(anchors_ep, pairs, k_gap)
        imask = np.zeros(T, dtype=bool)
        if ins is not None:
            imask[ins["place_start"]: ins["place_end"] + 1] = True
        info["handovers"] = pairs
        info["insert"] = ins
        for t in range(T):
            if hmask[t]:
                ev[t] = "handover"
            elif imask[t]:
                ev[t] = "insert"
            else:
                ev[t] = map_plug_base(left[t], right[t], lh[t], rh[t])

    elif ti == 2:
        for t in range(T):
            ev[t] = map_stack_bowl(left[t], right[t], lh[t], rh[t])

    else:
        raise SystemExit(f"task_index={ti} 暂无第 2 层映射规则 (§12)")

    assert all(isinstance(x, str) and x for x in ev), "存在未映射的帧"
    return ev.astype(str), info


# ---------------------------------------------------------------------------
# 数据集级：读锚点/状态 → 映射 → 合并进 frame_weight.csv
# ---------------------------------------------------------------------------

def load_segments(seg_dir: Path, tasks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读三个任务的 anchors / arm_states 表并纵向拼接 (按 episode_index 判定任务)。"""
    anch, sts = [], []
    for ti in tasks:
        a = pd.read_csv(seg_dir / f"gripper_anchors_task{ti}.csv")
        s = pd.read_csv(seg_dir / f"arm_states_task{ti}.csv")
        a["task_index"] = ti
        s["task_index"] = ti
        anch.append(a)
        sts.append(s)
    return (pd.concat(anch, ignore_index=True), pd.concat(sts, ignore_index=True))


def build(tasks: list[int], seg_dir: Path, k_gap: int = K_GAP) -> tuple[pd.DataFrame, dict]:
    """跑完三个任务的第 2 层映射, 返回 (逐帧三列表, 供 review 的上下文 json)。"""
    anch_all, sts_all = load_segments(seg_dir, tasks)
    rows, ctx = [], {"k_gap": k_gap, "episodes": {}}
    for ti in tasks:
        a_ti = anch_all[anch_all.task_index == ti]
        s_ti = sts_all[sts_all.task_index == ti]
        meta = json.loads((seg_dir / f"meta_task{ti}.json").read_text(encoding="utf-8")) \
            if (seg_dir / f"meta_task{ti}.json").exists() else {}
        ctx["episodes"][str(ti)] = {"slug": meta.get("config_body", {}).get(
            "tasks", {}).get(str(ti), {}).get("slug", f"task{ti}"), "per_ep": {}}
        for ep in sorted(a_ti.episode_index.unique()):
            a_ep = a_ti[a_ti.episode_index == ep]
            s_ep = s_ti[s_ti.episode_index == ep].sort_values("frame_index")
            if len(s_ep) == 0:
                raise SystemExit(f"episode {ep}: 缺 arm_states 逐帧标签")
            ev, info = layer2_episode(ti, s_ep, a_ep, k_gap)
            rows.append(pd.DataFrame({
                "episode_index": int(ep),
                "frame_index": s_ep.frame_index.to_numpy(dtype=int),
                "left_state": s_ep.left_state.to_numpy(),
                "right_state": s_ep.right_state.to_numpy(),
                "task_event": ev,
            }))
            ctx["episodes"][str(ti)]["per_ep"][str(int(ep))] = {
                k: v for k, v in info.items() if k not in ("task_index", "num_frames")}
    all_rows = pd.concat(rows, ignore_index=True).sort_values(
        ["episode_index", "frame_index"]).reset_index(drop=True)
    return all_rows, ctx


def merge_into_frame_weight(fw_path: Path, l2: pd.DataFrame) -> pd.DataFrame:
    """把三列按 (episode_index, frame_index) 合并进 frame_weight.csv, 原列与行序不变。"""
    fw = pd.read_csv(fw_path)
    keep = [c for c in fw.columns if c not in NEW_COLS]      # 幂等: 重跑先丢旧列
    fw = fw[keep]
    out = fw.merge(l2, on=["episode_index", "frame_index"], how="left",
                   validate="one_to_one", sort=False)
    miss = out[list(NEW_COLS)].isna().any(axis=1)
    if miss.any():
        bad = out.loc[miss, ["episode_index", "frame_index"]].head(5).to_dict("records")
        raise SystemExit(f"对齐失败: {int(miss.sum())} 行在 xyz_segment 中无对应标签, "
                         f"例如 {bad}")
    if len(out) != len(fw):
        raise SystemExit(f"行数改变: {len(fw)} -> {len(out)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="第 2 层任务级事件映射 (§12)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="0,1,2", help="逗号分隔的 task_index")
    ap.add_argument("--seg-dir", default=str(SEG_DIR),
                    help="第 1 层产物目录 (gripper_anchors_task*.csv / arm_states_task*.csv)")
    ap.add_argument("--frame-weight", default=str(FRAME_WEIGHT), help="要写入的目标 CSV")
    ap.add_argument("--out-csv", default=None, help="写到别处 (默认原地更新 --frame-weight)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="review 上下文 json 目录")
    ap.add_argument("--k-gap", type=int, default=K_GAP, help="§12.2 条件1 允许的配对间隔")
    ap.add_argument("--dry-run", action="store_true", help="只统计, 不写回 CSV")
    args = ap.parse_args()

    tasks = [int(v) for v in args.tasks.split(",") if v.strip()]
    l2, ctx = build(tasks, Path(args.seg_dir), args.k_gap)

    # ---- 统计 ----
    print(f"逐帧任务级标签: {len(l2):,} 帧, {l2.episode_index.nunique()} episodes")
    for ti in tasks:
        eps = sorted(ctx["episodes"][str(ti)]["per_ep"].keys())
        sub = l2[l2.episode_index.isin([int(e) for e in eps])]
        print(f"\n=== task{ti} ({ctx['episodes'][str(ti)]['slug']})  "
              f"{len(eps)} episodes / {len(sub):,} 帧 ===")
        vc = sub.task_event.value_counts()
        for k, v in vc.items():
            print(f"  {k:14s} {EVENT_CN.get(k, ''):8s} {v:7,d}  {v / len(sub) * 100:5.2f}%")
        if ti == 0:
            hs = [ctx["episodes"]["0"]["per_ep"][e]["holder_side"] for e in eps]
            print(f"  持筒臂分布: left={hs.count('left')}  right={hs.count('right')}")
        if ti == 1:
            per = ctx["episodes"]["1"]["per_ep"]
            nh = {e: len(per[e]["handovers"]) for e in eps}
            ni = {e: (per[e]["insert"]["side"] if per[e]["insert"] else None) for e in eps}
            ncand = [per[e]["insert"]["n_candidates"] for e in eps if per[e]["insert"]]
            print(f"  每 episode 交接数分布: "
                  f"{ {k: list(nh.values()).count(k) for k in sorted(set(nh.values()))} }")
            print(f"  插入事件: 缺失 {sum(v is None for v in ni.values())} 个; "
                  f"插入臂 left={list(ni.values()).count('left')} "
                  f"right={list(ni.values()).count('right')}; "
                  f"候选唯一率 {sum(c == 1 for c in ncand)}/{len(ncand)}")

    # ---- review 上下文 ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layer2_context.json").write_text(
        json.dumps(ctx, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    if args.dry_run:
        print("\n[dry-run] 未写 CSV")
        return

    dst = Path(args.out_csv or args.frame_weight)
    merged = merge_into_frame_weight(Path(args.frame_weight), l2)
    merged.to_csv(dst, index=False)
    print(f"\n已写出: {dst}  {merged.shape[0]:,} 行 x {merged.shape[1]} 列")
    print(f"  列: {list(merged.columns)}")
    print(f"  review 上下文: {out_dir / 'layer2_context.json'}")


if __name__ == "__main__":
    main()
