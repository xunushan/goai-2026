#!/usr/bin/env python3
"""第 2 层：任务级事件映射 (doc §12) —— 每帧输出唯一任务标签。

第 1 层 (tools/xyz_gripper_segment.py) 只产出任务无关的单臂五状态
{move, grasp, place, hold, idle}。本模块在其上做任务级映射

    E_t^task = F_task(s_t^left, s_t^right, c_t)

c_t = 持物历史 (holding) + 事件区间 (grasp/place 的起止帧, 取自 anchors 表)。
映射规则全部来自 docs/xyz_gripper_event_segmentation.md §12, 代码里不加任务知识:

统一层级: grasp/place > move > hold > idle (move 优先于 hold —— 静止持物只是背景状态,
另一臂在移动时不能被它压住)。同一层级内再按任务自己的臂/物体优先级排。

  §12.1 fill_pen_holder  —— 角色 (持筒臂/笔臂) 由 episode 自动判定;
                            层级 grasp/place(笔臂先于持筒臂) > move_pen/move_holder/move
                            > hold_pen/hold_holder > idle
  §12.2 plug_in_charger  —— 优先级 handover > insert > grasp_plug/place_plug
                            > move > hold_plug > idle
                            handover = 递交侧 place ∪ 接取侧 grasp 的**并集**
                            insert   = 最后一个非交接、且在末次 handover 之后的 place
  §12.3 stack_bowl       —— 层级 place_bowl > grasp_bowl > move_bowl > move
                            > hold_bowl > idle (不按左右臂设固定优先级)

★ fill 角色是数据事实, 不是配置常量: doc §12.1 的示例配置把职责写死成
  left=holder / right=pen, 但 sim_lerobot_v30_ee 的 100 个 fill episode 中
  46 个是左臂持筒、54 个是右臂持筒。故此处复用关键帧权重管线同一套判定
  ke.resolve_fill_roles (保持跨度最大的臂为持筒臂), 而不是照抄配置里的固定映射。

产物: 把 left_state / right_state / task_event 三列按 (episode_index, frame_index)
合并进 data/sim_lerobot_v30_ee/frame_weight.csv, 原有列原样不动。

--------------------------------------------------------------------------
真实数据集 (--dataset real)
--------------------------------------------------------------------------
data/real_lerobot_v30_ee 的 6 个任务里只有 3 个在 doc §12 有映射规则, 另 3 个按
doc §2.2 的 `<状态>_<物体>` 通用层级补齐 (族名见 FAMILY_BY_SLUG):

  t0 fill_pen_holder        §12.1  持筒臂/笔臂 (角色由 resolve_fill_roles 自动判定)
  t1 put_objects_into_basket        通用 _object  (物体入筐)
  t2 stack_and_cover_blocks         通用 _block + 末尾 place_cup (扣杯)
  t3 stack_bowls            §12.3   _bowl
  t4 stand_up_bottles               通用 _bottle (立瓶子)
  t5 insert_charger         §12.2  handover / insert

真实侧两处与仿真不同:
① K_gap 从 25 放宽到 60 —— 真实 6 集的真交接 (递交侧 place ↔ 接取侧 grasp) 间隔实测
   为 36/40/40/43/48 帧 (ep502 为重叠 -30)，而次近的假候选 ≥122 帧，60 落在
   (48, 122) 内。依据与取值写在 configs/real_xyz_segment_config.json 的 layer2.k_gap。
② 输出是**新建**的 5 列表 (data/real_lerobot_v30_ee/frame_weight.csv)，不含仿真侧的
   关键帧权重列。

用法:
    python tools/xyz_task_event.py --dry-run          # 只统计, 不写文件
    python tools/xyz_task_event.py                    # 写回 sim frame_weight.csv
    python tools/xyz_task_event.py --tasks 1 --out-csv outputs/x.csv
    python tools/xyz_task_event.py --dataset real     # 生成真实侧 frame_weight.csv
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
SIDES = ("left", "right")
NEW_COLS = ("left_state", "right_state", "task_event")

# ---------------------------------------------------------------------------
# 数据集配置：仿真 3 任务 / 真实 6 任务
# ---------------------------------------------------------------------------
# 真实数据集 data/real_lerobot_v30_ee 的任务编号与仿真**不同**，所以任务分派一律
# 按 meta_task*.json 里的 slug 走 (见 FAMILY_BY_SLUG)，不按 task_index 硬编码。
#   仿真  0=fill_pen_holder  1=plug_in_charger  2=stack_bowls
#   真实  0=fill_pen_holder  1=put_objects_into_basket  2=stack_and_cover_blocks
#         3=stack_bowls  4=stand_up_bottles  5=insert_charger
DATASETS = {
    "sim": {
        "seg_dir": ROOT / "data" / "sim_lerobot_v30_ee" / "xyz_segment",
        "frame_weight": ROOT / "data" / "sim_lerobot_v30_ee" / "frame_weight.csv",
        "out": ROOT / "outputs" / "xyz_task_event",
        "tasks": (0, 1, 2),
        "k_gap": 25,            # §12.2 条件 1 允许的配对间隔 (帧)
        "fresh": False,         # False = 并入既有 frame_weight.csv (保留权重列)
    },
    "real": {
        "seg_dir": ROOT / "outputs" / "real_xyz_segment" / "_dataset_labels",
        "frame_weight": ROOT / "data" / "real_lerobot_v30_ee" / "frame_weight.csv",
        "out": ROOT / "outputs" / "real_xyz_task_event",
        "tasks": (0, 1, 2, 3, 4, 5),
        # K_gap 由 configs/real_xyz_segment_config.json 的 layer2.k_gap 覆盖 (默认值仅占位)
        "k_gap": 25,
        "fresh": True,          # True = 新建 5 列表 (episode/frame/左右状态/task_event)
        "config": ROOT / "configs" / "real_xyz_segment_config.json",
    },
}

# slug -> 映射族。doc §12 只定义了 fill / plug / bowl 三族；real 另外三个任务
# (t1 物体入筐 / t2 叠并盖积木 / t4 立瓶子) 按 doc §2.2 的 `<状态>_<物体>` 规则
# 补出 object / block / bottle 三族，通用层级与 §12.3 完全一致。
FAMILY_BY_SLUG = {
    "fill_pen_holder": "fill",              # §12.1
    "plug_in_charger": "plug",              # §12.2
    "insert_charger": "plug",               # §12.2 (real 的任务名)
    "stack_bowls": "bowl",                  # §12.3
    "put_objects_into_basket": "object",    # doc 未定义，按 §2.2 补
    "stack_and_cover_blocks": "block",      # doc 未定义，按 §2.2 补
    "stand_up_bottles": "bottle",           # doc 未定义，按 §2.2 补
}

SEG_DIR = DATASETS["sim"]["seg_dir"]
FRAME_WEIGHT = DATASETS["sim"]["frame_weight"]
DEFAULT_OUT = DATASETS["sim"]["out"]
K_GAP = DATASETS["sim"]["k_gap"]

# 任务级标签的中文名 (仅供展示)
EVENT_CN = {
    "move": "移动", "idle": "静止",
    "grasp_holder": "抓笔筒", "place_holder": "放笔筒",
    "hold_holder": "持笔筒静止", "move_holder": "运笔筒",
    "grasp_pen": "抓笔", "place_pen": "放笔",
    "hold_pen": "持笔静止", "move_pen": "运笔",
    "handover": "交接", "insert": "插入",
    "grasp_plug": "抓插头", "place_plug": "放插头",
    "hold_plug": "持插头静止", "move_plug": "运插头",
    "grasp_bowl": "抓碗", "place_bowl": "放碗",
    "hold_bowl": "持碗静止", "move_bowl": "运碗",
    "grasp_object": "抓物体", "place_object": "放物体",
    "hold_object": "持物体静止", "move_object": "运物体",
    "grasp_block": "抓积木", "place_block": "放积木",
    "hold_block": "持积木静止", "move_block": "运积木",
    "place_cup": "扣杯",
    "grasp_bottle": "抓瓶子", "place_bottle": "放瓶子",
    "hold_bottle": "持瓶静止", "move_bottle": "运瓶子",
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

    层级严格按 doc 的总优先级 grasp_* / place_* > move_* > hold_* > idle:

      第 1 层 grasp/place —— 笔臂先于持筒臂 (doc: 右臂只有在抓笔/放笔时才覆盖左臂);
      第 2 层 move        —— 任一手臂在移动时先出 move (持物带后缀, 空手为 move);
      第 3 层 hold        —— 两臂都不动时才轮到 hold;
      第 4 层 idle。

    ★ 持筒臂的 hold 是"稳定持筒"的背景状态 (doc §12.1), 只要另一臂在移动, 背景状态
      就不能压过 move —— 原参考实现把 hold_pen/hold_holder 排在 move 之前, 是错的。
    """
    pen = "right" if holder == "left" else "left"
    st = {"left": left, "right": right}
    hd = {"left": left_holding, "right": right_holding}
    # 1. grasp / place —— 笔臂先于持筒臂
    if st[pen] == "place" and hd[pen]:
        return "place_pen"
    if st[pen] == "grasp":
        return "grasp_pen"
    if st[holder] == "place" and hd[holder]:
        return "place_holder"
    if st[holder] == "grasp":
        return "grasp_holder"
    # 2. move —— 主动运动覆盖另一侧的背景持物
    if st[pen] == "move":
        return "move_pen" if hd[pen] else "move"
    if st[holder] == "move" and hd[holder]:
        return "move_holder"
    # 3. hold —— 两臂都不移动时才轮到背景持物状态
    if st[pen] == "hold" and hd[pen]:
        return "hold_pen"
    if st[holder] == "hold" and hd[holder]:
        return "hold_holder"
    # 4. 兜底
    if st[holder] == "move":
        return "move"
    return "idle"


def map_plug_base(left, right, left_holding, right_holding) -> str:
    """§12.2 非交接、非插入帧的单臂物体语义映射。

    层级 place_plug > grasp_plug > move_plug / move > hold_plug > idle —— move 先于
    hold (doc §12.2: 一只手臂静止持有插头、另一只手臂仍空手接近时, 标签应为 move,
    主动运动覆盖背景持物状态)。持插头运输带后缀 move_plug (doc 事件序列
    `move → handover → move_plug / hold_plug → insert`), 空手移动为 move。
    """
    st, hd = {"left": left, "right": right}, {"left": left_holding, "right": right_holding}
    for s in ("place", "grasp"):
        for arm in SIDES:
            if st[arm] == s:
                return f"{s}_plug"
    for arm in SIDES:
        if st[arm] == "move" and hd[arm]:
            return "move_plug"
    if "move" in (left, right):
        return "move"
    for arm in SIDES:
        if st[arm] == "hold" and hd[arm]:
            return "hold_plug"
    return "idle"


def map_stack_bowl(left, right, left_holding, right_holding) -> str:
    """§12.3 叠碗映射。对左右臂对称, 不按左右臂设固定优先级。

    层级 place_bowl > grasp_bowl > move_bowl > move > hold_bowl > idle ——
    move 先于 hold (持碗静止是背景状态, 另一臂在动时应出 move)。
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
    if "move" in (left, right):
        return "move"
    for arm in SIDES:
        if st[arm] == "hold":
            return "hold_bowl"
    return "idle"


def map_object_semantics(left, right, left_holding, right_holding, noun: str) -> str:
    """通用物体语义映射 (doc §2.2 的 `<状态>_<物体>`) —— 供 doc §12 未定义的任务复用。

    层级与 §12.3 stack_bowl 完全一致, 只换物体后缀:

        place_<noun> > grasp_<noun> > move_<noun> > move > hold_<noun> > idle

    对左右臂对称, 不按左右臂设固定优先级; move 先于 hold (持物静止只是背景状态,
    另一臂在动时应出 move, doc §12.1/§12.2/§12.3 三处都强调)。`move_<noun>` 仅在
    "当前确实持有该物体"时输出 —— 空手移动不能带物体后缀 (doc §12.1)。
    """
    st, hd = {"left": left, "right": right}, {"left": left_holding, "right": right_holding}
    for s in ("place", "grasp"):
        for arm in SIDES:
            if st[arm] == s:
                return f"{s}_{noun}"
    for arm in SIDES:
        if st[arm] == "move" and hd[arm]:
            return f"move_{noun}"
    if "move" in (left, right):
        return "move"
    for arm in SIDES:
        if st[arm] == "hold":
            return f"hold_{noun}"
    return "idle"


def cup_place_event(anchors_ep: pd.DataFrame,
                    pairs: list[dict] | None = None) -> dict | None:
    """叠积木任务里"扣杯子"那一次 place: 取该集**最后一次**非交接的 place。

    依据: 官方任务串 "Stack all three blocks into one vertical stack, then place the
    cup over the entire stack" (data/real_lerobot_v30_ee/meta/tasks.parquet), 扣杯发生在
    叠完三块之后。实测 6 集该 place 与前一次 place 的时间间隔为 130~250 帧 (其余相邻
    place 间隔都在 150 帧以内且成串出现), 是唯一一个孤立在末尾的 place。

    返回 {"side","cycle","place_start","place_end"} 或 None。
    """
    done = {(pr["releaser"], pr["releaser_cycle"]) for pr in (pairs or [])}
    cand = [r for _, r in anchors_ep[anchors_ep.place_start >= 0].iterrows()
            if (r.side, int(r.cycle)) not in done]
    if not cand:
        return None
    r = max(cand, key=lambda x: int(x.place_end))
    return {"side": r.side, "cycle": int(r.cycle),
            "place_start": int(r.place_start), "place_end": int(r.place_end)}


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


def layer2_episode(family: str, states_ep: pd.DataFrame, anchors_ep: pd.DataFrame,
                   k_gap: int = K_GAP) -> tuple[np.ndarray, dict]:
    """一个 episode 的第 2 层映射 -> (逐帧 task_event, 上下文信息)。

    states_ep 需按 frame_index 升序, 列含 left_state/right_state/left_holding/right_holding。
    family 见 FAMILY_BY_SLUG: fill(§12.1) / plug(§12.2) / bowl(§12.3)
    / object / block / bottle (doc 未定义, 按 §2.2 的 `<状态>_<物体>` 通用层级补)。
    """
    T = len(states_ep)
    left = states_ep.left_state.to_numpy()
    right = states_ep.right_state.to_numpy()
    lh = states_ep.left_holding.to_numpy()
    rh = states_ep.right_holding.to_numpy()
    ev = np.empty(T, dtype=object)
    info: dict = {"family": family, "num_frames": T}

    if family == "fill":                                    # §12.1
        roles = ke.resolve_fill_roles(np.zeros(T), per_side_from_anchors(anchors_ep))
        holder = roles["holder"]
        info["holder_side"] = holder
        info["pen_side"] = roles["pen"]
        for t in range(T):
            ev[t] = map_fill_pen_holder(left[t], right[t], lh[t], rh[t], holder)

    elif family == "plug":                                  # §12.2
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

    elif family == "bowl":                                  # §12.3
        for t in range(T):
            ev[t] = map_stack_bowl(left[t], right[t], lh[t], rh[t])

    elif family == "block":
        # 叠三块积木 (通用 _block 语义) + 末尾单独扣杯 (place_cup)
        cup = cup_place_event(anchors_ep)
        cmask = np.zeros(T, dtype=bool)
        if cup is not None:
            cmask[cup["place_start"]: cup["place_end"] + 1] = True
        info["place_cup"] = cup
        for t in range(T):
            ev[t] = map_object_semantics(left[t], right[t], lh[t], rh[t], "block")
        ev[cmask] = "place_cup"
        info["n_frames_place_cup"] = int(cmask.sum())

    elif family in ("object", "bottle"):
        for t in range(T):
            ev[t] = map_object_semantics(left[t], right[t], lh[t], rh[t], family)

    else:
        raise SystemExit(f"family={family!r} 暂无第 2 层映射规则 (§12 / §2.2)")

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
    """跑完各任务的第 2 层映射, 返回 (逐帧五列表, 供 review 的上下文 json)。

    任务分派按 meta_task*.json 里的 slug 走 (见 FAMILY_BY_SLUG), 不按 task_index 硬编码 ——
    仿真与真实的任务编号不同。
    """
    anch_all, sts_all = load_segments(seg_dir, tasks)
    rows, ctx = [], {"k_gap": k_gap, "episodes": {}}
    for ti in tasks:
        a_ti = anch_all[anch_all.task_index == ti]
        s_ti = sts_all[sts_all.task_index == ti]
        meta = json.loads((seg_dir / f"meta_task{ti}.json").read_text(encoding="utf-8")) \
            if (seg_dir / f"meta_task{ti}.json").exists() else {}
        slug = meta.get("config_body", {}).get(
            "tasks", {}).get(str(ti), {}).get("slug", f"task{ti}")
        if slug not in FAMILY_BY_SLUG:
            raise SystemExit(f"task{ti} slug={slug!r} 未登记映射族 (见 FAMILY_BY_SLUG)")
        family = FAMILY_BY_SLUG[slug]
        ctx["episodes"][str(ti)] = {"slug": slug, "family": family, "per_ep": {}}
        for ep in sorted(a_ti.episode_index.unique()):
            a_ep = a_ti[a_ti.episode_index == ep]
            s_ep = s_ti[s_ti.episode_index == ep].sort_values("frame_index")
            if len(s_ep) == 0:
                raise SystemExit(f"episode {ep}: 缺 arm_states 逐帧标签")
            ev, info = layer2_episode(family, s_ep, a_ep, k_gap)
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


def build_fresh(tasks: list[int], seg_dir: Path, l2: pd.DataFrame) -> pd.DataFrame:
    """新建 5 列 frame_weight.csv 的内容 (episode/frame/左右状态/task_event)。

    真实数据集没有既有 frame_weight.csv (权重列是仿真侧的产物)，所以这里直接以第 1 层
    的 arm_states_task*.csv 为骨架 —— 它恰好覆盖这 36 集全部帧，且自带 episode_index。
    """
    cols = ["episode_index", "frame_index", "left_state", "right_state"]
    parts = [pd.read_csv(seg_dir / f"arm_states_task{ti}.csv", usecols=cols)
             for ti in tasks]
    skel = pd.concat(parts, ignore_index=True).sort_values(
        ["episode_index", "frame_index"]).reset_index(drop=True)
    skel["episode_index"] = skel.episode_index.astype(int)
    skel["frame_index"] = skel.frame_index.astype(int)
    out = skel.merge(l2, on=["episode_index", "frame_index"], how="left",
                     validate="one_to_one", sort=False, suffixes=("_skel", "_l2"))
    miss = out.task_event.isna()
    if miss.any():
        bad = out.loc[miss, ["episode_index", "frame_index"]].head(5).to_dict("records")
        raise SystemExit(f"对齐失败: {int(miss.sum())} 行无 task_event, 例如 {bad}")
    if len(out) != len(skel):
        raise SystemExit(f"行数改变: {len(skel)} -> {len(out)}")
    # 骨架与第 2 层读的是同一份 arm_states_task*.csv, 两列状态必须逐帧一致 —— 不一致
    # 说明第 1 层产物在两次读取之间被换掉了, 必须报错而不是静默取一边。
    for c in ("left_state", "right_state"):
        bad = int((out[f"{c}_skel"] != out[f"{c}_l2"]).sum())
        if bad:
            raise SystemExit(f"{c}: 骨架与第 2 层结果逐帧对比不一致 ({bad} 帧)")
    out["left_state"], out["right_state"] = out.left_state_skel, out.right_state_skel
    return out[["episode_index", "frame_index", "left_state", "right_state", "task_event"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="第 2 层任务级事件映射 (§12)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sim", choices=sorted(DATASETS),
                    help="sim = data/sim_lerobot_v30_ee (并入既有 frame_weight.csv); "
                         "real = data/real_lerobot_v30_ee (新建 5 列 frame_weight.csv)")
    ap.add_argument("--tasks", default=None, help="逗号分隔的 task_index (默认按 --dataset)")
    ap.add_argument("--seg-dir", default=None,
                    help="第 1 层产物目录 (gripper_anchors_task*.csv / arm_states_task*.csv)")
    ap.add_argument("--frame-weight", default=None, help="要写入的目标 CSV")
    ap.add_argument("--out-csv", default=None, help="写到别处 (默认原地更新 --frame-weight)")
    ap.add_argument("--out", default=None, help="review 上下文 json 目录")
    ap.add_argument("--k-gap", type=int, default=None,
                    help="§12.2 条件1 允许的配对间隔; real 默认取配置的 layer2.k_gap")
    ap.add_argument("--dry-run", action="store_true", help="只统计, 不写回 CSV")
    args = ap.parse_args()

    dcfg = DATASETS[args.dataset]
    seg_dir = Path(args.seg_dir or dcfg["seg_dir"])
    frame_weight = Path(args.frame_weight or dcfg["frame_weight"])
    out_dir = Path(args.out or dcfg["out"])
    k_gap = args.k_gap
    if k_gap is None:
        k_gap = dcfg["k_gap"]
        cfg_path = dcfg.get("config")
        if cfg_path and Path(cfg_path).is_file():        # real: 配置是唯一来源
            body = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            k_gap = int(body.get("layer2", {}).get("k_gap", k_gap))
    tasks = ([int(v) for v in args.tasks.split(",") if v.strip()] if args.tasks
             else list(dcfg["tasks"]))
    fresh = dcfg["fresh"]

    print(f"数据集 = {args.dataset}   seg_dir = {seg_dir}")
    print(f"tasks = {tasks}   k_gap = {k_gap}   "
          f"输出 = {'新建 5 列表' if fresh else '并入既有 frame_weight.csv'}")
    l2, ctx = build(tasks, seg_dir, k_gap)

    # ---- 统计 ----
    print(f"\n逐帧任务级标签: {len(l2):,} 帧, {l2.episode_index.nunique()} episodes")
    for ti in tasks:
        meta = ctx["episodes"][str(ti)]
        eps = sorted(meta["per_ep"].keys())
        sub = l2[l2.episode_index.isin([int(e) for e in eps])]
        print(f"\n=== task{ti} ({meta['slug']} / {meta['family']})  "
              f"{len(eps)} episodes / {len(sub):,} 帧 ===")
        vc = sub.task_event.value_counts()
        for k, v in vc.items():
            print(f"  {k:14s} {EVENT_CN.get(k, ''):8s} {v:7,d}  {v / len(sub) * 100:5.2f}%")
        per = meta["per_ep"]
        if meta["family"] == "fill":
            hs = [per[e]["holder_side"] for e in eps]
            print(f"  持筒臂分布: left={hs.count('left')}  right={hs.count('right')}")
        if meta["family"] == "plug":
            nh = {e: len(per[e]["handovers"]) for e in eps}
            ni = {e: (per[e]["insert"]["side"] if per[e]["insert"] else None) for e in eps}
            ncand = [per[e]["insert"]["n_candidates"] for e in eps if per[e]["insert"]]
            print(f"  每 episode 交接数分布: "
                  f"{ {k: list(nh.values()).count(k) for k in sorted(set(nh.values()))} }")
            print(f"  交接并集窗口: " + "; ".join(
                f"ep{e}[{per[e]['handovers'][0]['u'][0]},{per[e]['handovers'][0]['u'][1]}]"
                for e in eps if per[e]["handovers"]))
            print(f"  插入事件: 缺失 {sum(v is None for v in ni.values())} 个; "
                  f"插入臂 left={list(ni.values()).count('left')} "
                  f"right={list(ni.values()).count('right')}; "
                  f"候选唯一率 {sum(c == 1 for c in ncand)}/{len(ncand)}")
        if meta["family"] == "block":
            print("  扣杯事件 (最后一次 place): " + "; ".join(
                f"ep{e} {per[e]['place_cup']['side']}{per[e]['place_cup']['cycle']}"
                f"[{per[e]['place_cup']['place_start']},{per[e]['place_cup']['place_end']}]"
                for e in eps if per[e]["place_cup"]))

    # ---- review 上下文 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layer2_context.json").write_text(
        json.dumps(ctx, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    if args.dry_run:
        print("\n[dry-run] 未写 CSV")
        return

    dst = Path(args.out_csv or frame_weight)
    out = (build_fresh(tasks, seg_dir, l2) if fresh
           else merge_into_frame_weight(frame_weight, l2))
    out.to_csv(dst, index=False)
    print(f"\n已写出: {dst}  {out.shape[0]:,} 行 x {out.shape[1]} 列")
    print(f"  列: {list(out.columns)}")
    print(f"  review 上下文: {out_dir / 'layer2_context.json'}")


if __name__ == "__main__":
    main()
