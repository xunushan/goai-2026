#!/usr/bin/env python3
"""导出 in-context learning 演示 (demo.json + 关键帧图)。

对每个任务选 1 个 episode, 按 frame_weight.csv 的 task_event 分段, 每个事件**类型**
取 1 个代表帧, 连同该帧的 state/action 与三路相机图, 组装成可供 in-context 使用的
demo.json。

两个数据集共用同一套输出结构, 差异集中在 PROFILES:

    sim   data/sim_lerobot_v30_ee   -> outputs/in_context_learning_sim    (3 任务)
    real  data/real_lerobot_v30_ee  -> outputs/in_context_learning_real   (6 任务)

差异主要在分段与代表帧的定法:
  - sim 的 frame_weight.csv 带 frame_weight_loss 列, 代表帧 = 关键帧平台末帧 (见
    segments_sim)
  - real 的只有 task_event, 没有权重列, 代表帧改从事件边界 + 夹爪轨迹推导 (见
    segments_real), 且真实数据的词表多了 hold_* 段, 各自单独成帧

数据来源:
- 事件分段: <dataset>/frame_weight.csv (列 task_event / 左右 state)
- state/action: <dataset>/data/chunk-000/*.parquet (16 维, 左 8 + 右 8)
- 图像: <dataset>/videos/observation.images.<cam>/chunk-000/file-XXX.mp4
        时间戳映射取自 meta/episodes 的 videos/<cam>/from_timestamp, 帧率 25fps

★ 不直接裁切 outputs/episode_insight/*/videos 的合成图: 那是多路画面的拼图,
  画质有损 (sim 那份的右腕还被压扁)。

用法 (须用 lerobot 环境: parquet 由 arrow 25 写出, base 的 pyarrow 19 读会报
"Repetition level histogram size mismatch"):
    /opt/anaconda3/envs/lerobot/bin/python tools/in_context_demo.py --dataset sim --all
    /opt/anaconda3/envs/lerobot/bin/python tools/in_context_demo.py --dataset real --all
    /opt/anaconda3/envs/lerobot/bin/python tools/in_context_demo.py --dataset real \\
        --task stack_bowls --episode 302
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FPS = 25.0
WEIGHT_BASE = 1.0   # frame_weight_loss 的窗口外取值 (见 tools/keyframe_events.py)

# ---------------------------------------------------------------- sim 的词表

# task_event -> 该段在动作链里的阶段名
STAGE_BY_EVENT = {
    "move": "approach",
    "move_bowl": "transport",
    "move_pen": "transport",
    "move_holder": "transport",
    "move_plug": "transport",
    "grasp_bowl": "grasp",
    "grasp_pen": "grasp",
    "grasp_holder": "grasp",
    "grasp_plug": "grasp",
    "grasp_charger": "grasp",
    "place_bowl": "place",
    "place_pen": "place",
    "place_holder": "place",
    "place_plug": "place",
    "insert": "insert",
    "handover": "handover",
}

MOVEMENT_EVENTS = {"move", "move_bowl", "move_pen", "move_holder", "move_plug"}

# ---------------------------------------------------------------- real 的词表

# real 的事件名都是 <动词>_<对象> (grasp_holder / hold_pen / place_bowl ...),
# 阶段直接按前缀归类, 见 segments_real

GRIP_TOL = 0.05     # 夹爪"仍闭合/仍张开"的容差

# 相机键名与数据集一致 (info.json 的 features)
CAMS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
POS = {"left": slice(0, 3), "right": slice(8, 11)}
QUAT = {"left": slice(3, 7), "right": slice(11, 15)}
GRIP = {"left": 7, "right": 15}
STATE_LAYOUT = "left[x,y,z,qw,qx,qy,qz,gripper] + right[same]"

# ---------------------------------------------------------------- 人工标注

# 人工标注 (逐帧看过 head 相机画面后撰写): (task, episode, frame_index) -> (observation, result)
# 未列出的 keyframe 只输出数值, observation/result 为空

# --- sim ---
ANNOTATIONS_SIM = {
    # --- stack_bowls, episode 203 (右臂叠前两只, 左臂叠第三只) ---
    # 键 = (task, episode, frame_index), 帧号比去重后的序号稳定
    ("stack_bowls", 203, 24): ("right jaws open, closing in on the first upright bowl; left arm parked at home", "aligned"),
    ("stack_bowls", 203, 52): ("right jaws just closed on the bowl rim, bowl tilted as it leaves the table", "grasped"),
    ("stack_bowls", 203, 74): ("right arm carrying the bowl left across the table toward the centre; left arm parked", "transported"),
    ("stack_bowls", 203, 103): ("right arm has set the bowl down and its jaws are about to open; left arm parked", "placed"),
    ("stack_bowls", 203, 366): ("both arms back at their home poses; the three bowls rest as one nested stack at the table centre", "task_complete"),

    # --- fill_pen_holder, episode 0 (左臂持筒, 右臂插第一支笔: 青色) ---
    # 后 3 个插笔循环与第一组同型, 按事件类型去重时已丢弃
    ("fill_pen_holder", 0, 37): ("left jaws open, reaching for the orange pen holder lying on the table; four pens scattered to the right; right arm parked", "aligned"),
    ("fill_pen_holder", 0, 84): ("left jaws have closed on the pen holder, lifting it off the table", "grasped"),
    ("fill_pen_holder", 0, 112): ("left arm carrying the holder upright to its working position at the table centre", "transported"),
    ("fill_pen_holder", 0, 188): ("left arm holds the holder upright off the table; right jaws just closed on the first (teal) pen, lifting it off the table", "grasped"),
    ("fill_pen_holder", 0, 263): ("right arm carrying the teal pen upright toward the holder held by the left arm", "transported"),
    ("fill_pen_holder", 0, 304): ("right arm lowering the teal pen into the holder", "inserted"),
    ("fill_pen_holder", 0, 878): ("all four pens seated; left arm lowering the loaded holder back onto the table", "placed"),
    ("fill_pen_holder", 0, 933): ("both arms at their home poses; the holder stands upright on the table with all four pens in it", "task_complete"),

    # --- plug_in_charger, episode 150 (右臂取充电器, 交给左臂, 左臂插入插座) ---
    ("plug_in_charger", 150, 15): ("right jaws open, closing in on the charger lying on the table; the power strip sits to the left; left arm parked", "aligned"),
    ("plug_in_charger", 150, 33): ("right jaws just closed on the charger, lifting it off the table", "grasped"),
    ("plug_in_charger", 150, 68): ("right arm carrying the charger up and toward the table centre; left arm parked", "transported"),
    ("plug_in_charger", 150, 106): ("left jaws closing on the charger while the right jaws still hold it — both arms meet at the exchange point", "received"),
    ("plug_in_charger", 150, 157): ("left jaws hold the charger against the power-strip socket and are just starting to open", "inserted"),
    ("plug_in_charger", 150, 191): ("both arms back at their home poses; the charger sits inserted in the power strip", "task_complete"),
}

# --- real --- (逐帧看过 head 相机画面后撰写)
ANNOTATIONS_REAL = {
    # --- fill_pen_holder, episode 0 (右臂持筒, 左臂插笔 —— 与 sim 的左右相反) ---
    # 后 2 支笔的循环与第一组同型, 按事件类型去重时已丢弃
    ("fill_pen_holder", 0, 215): ("right jaws open, closing in on the teal pen holder lying on the table; the pens are scattered to its right; left arm parked at home", "aligned"),
    ("fill_pen_holder", 0, 254): ("right jaws have just closed on the pen holder and are lifting it off the table", "grasped"),
    ("fill_pen_holder", 0, 260): ("right arm holds the pen holder clear of the table, kept upright", "held"),
    ("fill_pen_holder", 0, 288): ("right arm carries the holder upright toward its working position at the centre of the table", "transported"),
    ("fill_pen_holder", 0, 481): ("right arm still holding the holder; left jaws have just closed on the first pen lying on the table", "grasped"),
    ("fill_pen_holder", 0, 483): ("left arm holds the pen clear of the table", "held"),
    ("fill_pen_holder", 0, 525): ("left arm carries the pen toward the holder held by the right arm", "transported"),
    ("fill_pen_holder", 0, 546): ("left arm lowers the pen into the holder", "inserted"),
    ("fill_pen_holder", 0, 1069): ("all pens seated; right arm lowering the loaded holder back onto the table", "placed"),
    ("fill_pen_holder", 0, 1213): ("both arms back at their home poses; the pen holder stands upright at the centre of the table with the pens in it", "task_complete"),

    # --- put_objects_into_basket, episode 105 (左臂单臂完成, 6 个物件反复抓放) ---
    # 该任务 frame_weight.csv 只标了这一集; 后 5 轮与第一轮同型, 去重时已丢弃
    ("put_objects_into_basket", 105, 177): ("left jaws open, closing in on the white mug at the left of the table; the wicker basket sits at the centre, the remaining objects are still spread around it; right arm parked", "aligned"),
    ("put_objects_into_basket", 105, 209): ("left jaws have just closed on the mug and are lifting it off the table", "grasped"),
    ("put_objects_into_basket", 105, 211): ("left arm holds the mug clear of the table", "held"),
    ("put_objects_into_basket", 105, 261): ("left arm carries the mug over the rim of the basket", "transported"),
    ("put_objects_into_basket", 105, 272): ("left jaws are opening to drop the mug into the basket", "placed"),
    ("put_objects_into_basket", 105, 1548): ("both arms back at their home poses; the table is clear and every object has ended up inside the basket", "task_complete"),

    # --- stack_and_cover_blocks, episode 294 (左臂单臂叠块, 最后盖杯) ---
    ("stack_and_cover_blocks", 294, 146): ("left jaws open, closing in on the blocks scattered over the table; the cup rests at the left; right arm parked", "aligned"),
    ("stack_and_cover_blocks", 294, 201): ("left jaws have just closed on the first block and are lifting it off the table", "grasped"),
    ("stack_and_cover_blocks", 294, 217): ("left arm carries the block toward the centre of the table", "transported"),
    ("stack_and_cover_blocks", 294, 236): ("left arm holds the block clear of the table at the centre", "held"),
    ("stack_and_cover_blocks", 294, 255): ("left jaws are opening to set the block down — the first block of the stack", "placed"),
    ("stack_and_cover_blocks", 294, 918): ("the blocks are stacked; left arm is lowering the cup over the stack", "placed"),
    ("stack_and_cover_blocks", 294, 1000): ("both arms back at their home poses; the cup sits over the block stack at the centre of the table", "task_complete"),

    # --- stack_bowls, episode 302 (左臂单臂叠碗) ---
    # approach 帧是从 grasp 段内合成的 (见 segments_real): CSV 的片段从夹持前一刻才开始
    ("stack_bowls", 302, 69): ("left jaws open, swinging in on the first bowl; all three bowls are still upright on the table; right arm parked", "aligned"),
    ("stack_bowls", 302, 132): ("left jaws have just closed on the bowl rim and are lifting it off the table", "grasped"),
    ("stack_bowls", 302, 156): ("left arm carries the bowl, held clear of the table", "transported"),
    ("stack_bowls", 302, 182): ("left arm holds the bowl upright in mid air", "held"),
    ("stack_bowls", 302, 204): ("left jaws are opening to set the bowl down", "placed"),
    ("stack_bowls", 302, 629): ("both arms back at their home poses; the bowls rest as one nested stack at the centre of the table", "task_complete"),

    # --- stand_up_bottles, episode 400 (左臂单臂把躺倒的瓶子逐支扶正) ---
    # approach 帧是从 grasp 段内合成的 (见 segments_real): 段前的 move 59-86 是双臂回位
    ("stand_up_bottles", 400, 158): ("left jaws open, reaching for the leftmost bottle lying on the table; the other two still lie flat; right arm parked", "aligned"),
    ("stand_up_bottles", 400, 228): ("left jaws have just closed on the bottle and are lifting it off the table", "grasped"),
    ("stand_up_bottles", 400, 233): ("left arm holds the bottle clear of the table", "held"),
    ("stand_up_bottles", 400, 249): ("left arm carries the bottle upright", "transported"),
    ("stand_up_bottles", 400, 348): ("left jaws are opening to release the bottle, which is left standing upright on the table", "placed"),
    ("stand_up_bottles", 400, 1115): ("both arms back at their home poses; all three bottles stand upright on the table", "task_complete"),

    # --- insert_charger, episode 503 (左臂取充电器 -> 交给右臂 -> 右臂插入插排) ---
    # 6 集中段结构最短者; 首次 grasp_plug 不是空抓: ep503 左爪 140-160 由 0.993 收到
    # 0.468, 确实夹住了台面左侧的白色充电器(腕部相机可见)
    ("insert_charger", 503, 131): ("left jaws open, closing in on the white charger lying at the left of the table; the power strip with its coiled cable lies at the centre; right arm parked", "aligned"),
    ("insert_charger", 503, 163): ("left jaws have just closed on the charger and are lifting it off the table", "grasped"),
    ("insert_charger", 503, 167): ("left arm holds the charger clear of the table", "held"),
    ("insert_charger", 503, 183): ("left arm carries the charger toward the centre of the table", "transported"),
    ("insert_charger", 503, 316): ("the two grippers meet above the power strip: the right jaws are closing on the charger while the left jaws open to release it", "received"),
    ("insert_charger", 503, 521): ("right arm brings the charger down onto the power strip and its jaws are opening", "placed"),
    ("insert_charger", 503, 1230): ("right arm pushes the plug fully home into the socket of the power strip", "inserted"),
    ("insert_charger", 503, 1336): ("both arms back at their home poses; the charger is plugged into the power strip with its cable connected", "task_complete"),
}

# ---------------------------------------------------------------- 分段


def load_frame_weight(prof, episode: int):
    """frame_weight.csv 里某 episode 的逐帧行, 按 frame_index 排序。"""
    with open(prof["root"] / "frame_weight.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["episode_index"]) == episode]
    rows.sort(key=lambda r: int(r["frame_index"]))
    return rows


def group_segments(rows, min_len: int = 3):
    """按 task_event 切段, 丢弃 idle 与短于 min_len 的碎片 (真实数据里有大量 1-2 帧的
    标注抖动)。"""
    segs = []
    for ev, grp in itertools.groupby(rows, key=lambda r: r["task_event"]):
        g = list(grp)
        if len(g) < min_len or ev in ("idle",):
            continue
        segs.append({
            "event": ev,
            "start": int(g[0]["frame_index"]),
            "end": int(g[-1]["frame_index"]),
            "rep": int(g[-1]["frame_index"]),
            "rows": g,
        })
    return segs


def dedupe_by_type(segs):
    """每个事件**类型**只留 1 帧, 取该类型首次出现的那一段。

    类型键 = (stage, event)。用 event 而非只 stage, 是为了让 fill 的
    grasp_holder / grasp_pen、place_holder / place_pen 各自成类 —— 否则
    多次插笔循环会被压成同一个 grasp/place, 持筒与持笔就分不开了。
    后续重复的同类段 (第二次插笔、再夹一只碗...) 随之丢弃。
    """
    seen, out = set(), []
    for s in segs:
        key = (s["stage"], s["event"])
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def segments_sim(rows, arr):
    """sim: 代表帧 = 关键帧平台 [t0+Pl, t0+Pr] 的**末**帧 = 事件锚点 t0。

    configs/keyframe_weight_config.json: grasp 取 hold_start = 有效抓取形成帧;
    place/insert/handover-release 取 hold_end = 夹爪开始张开帧。v2 窗口
    Pl=-5, Pr=0, 故平台末帧即 t0。实测: ep0 grasp_holder 平台 79-84, 闭爪在
    80-84 完成 → t0=84; ep203 place_bowl 平台 98-103, 夹爪 104 才张开 → t0=103。
    """
    segs = group_segments(rows)
    for s in segs:
        wmax = max(float(r["frame_weight_loss"]) for r in s["rows"])
        # wmax > 1 说明本段含自己事件的权重平台 (抓取/放置类); 否则是纯移动段
        anchored = wmax > WEIGHT_BASE
        last_max = max(s["rows"], key=lambda r: (float(r["frame_weight_loss"]), int(r["frame_index"])))
        s["rep"] = int(last_max["frame_index"]) if anchored else s["end"]
    # 首段的 move -> approach; 末段的 move -> home
    if segs and segs[0]["event"] == "move":
        segs[0]["stage"] = "approach"
    if segs and segs[-1]["event"] == "move" and len(segs) > 1:
        segs[-1]["stage"] = "home"
    for i, s in enumerate(segs):
        s.setdefault("stage", STAGE_BY_EVENT.get(s["event"], s["event"]))
        # 移动段没有自己的锚点, 代表帧一律取段末(到位瞬间)。注意不能依赖
        # anchored: 相邻事件的梯形窗口会溢出到移动段里(如 plug 的 insert 窗口
        # 137-162 盖住 transport 尾段), 使移动段"看起来"有关键帧。
        if s["event"] in MOVEMENT_EVENTS:
            s["rep"] = s["end"]
            # 若与下一事件锚点挤在一起(plug: transport 止于 151 / insert 锚点 157),
            # 两帧几乎重复, 退回段中
            if i + 1 < len(segs) and segs[i + 1]["rep"] - s["rep"] < 10:
                s["rep"] = (s["start"] + s["end"]) // 2
    return dedupe_by_type(segs)


def grip_trace(arr, side: str):
    """整段 episode 的单臂夹爪开度序列, 下标即 frame_index。"""
    return np.stack(arr["observation.state"].to_numpy())[:, GRIP[side]]


def release_rep(grip, a: int, b: int) -> int:
    """place_*/insert/handover 的代表帧 = **松手前最后一帧**。

    real 的 frame_weight.csv 没有权重列, 段边界也不落在张开瞬间: 实测 ep503
    place_plug 504-533 的夹爪 504-519 恒为 0.015, 520 才开始张开。故取本段内
    张开幅度最大的那只手臂 (即交出方), 再取它仍贴近本段最小开度的最后一帧 ——
    与 sim 的 hold_end (夹爪开始张开的前一帧) 同义。
    """
    arm = max(("left", "right"), key=lambda s: grip[s][b] - grip[s][a])
    g = grip[arm]
    lo = min(g[a:b + 1])
    ts = [t for t in range(a, b + 1) if g[t] <= lo + GRIP_TOL]
    return ts[-1] if ts else b


def reach_rep(pos, grip, a: int, b: int) -> int:
    """首个 grasp 段内"接近"阶段的代表帧。

    stack_bowls 六个 episode 的录制都从夹持前一刻才开始, CSV 里根本没有 move 段 ——
    接近过程被并进了首个 grasp_* 段 (实测 ep302 的 grasp_bowl 14-132 里, 前 ~100 帧
    是臂从 home 摆到碗边、夹爪只是预整形, 110 才开始真正闭合)。取闭合开始前手臂速度
    最大的那一帧: 臂在途中、夹爪仍张, 正是 sim 里 approach 帧的样子。
    """
    closed = [f for f in range(a, b + 1) if grip[f] < 0.5]
    hi = closed[0] - 1 if closed else b
    if hi <= a:
        return a
    spd = np.linalg.norm(np.diff(pos[a:hi + 1], axis=0), axis=1)
    return a + int(np.argmax(spd))


def segments_real(rows, arr):
    """real: 靠事件边界 + 夹爪轨迹定锚 (无 frame_weight_loss 可用)。

    - grasp_X            : 段末帧 —— 夹爪闭合到位, 同 sim 的 hold_start
    - hold_X             : 段中点 —— 保持平台
    - move_X             : 段末帧 —— 搬运到位瞬间
    - place_X / insert / handover : 松手前最后一帧 (见 release_rep)
    - 无类型的 move      : 首个 grasp 之前的最后一段 -> approach; 全局最后一段 ->
                           home; 其余同类丢弃 (都是重新定位的小动作, 与 approach 同型)
    - 代表帧上不是抓取手在动的 move 段不算 approach (取段末帧的 <arm>_state 判定):
      ep400 的 move 59-86 是双臂回位, 末帧 f86 左臂已 idle, 而抓瓶子的是左臂
    - 找不到合格的 approach 段时, 从首个 grasp 段内按 reach_rep 合成一帧, 标 synthetic。
      两种触发情形: (a) 该 grasp 之前没有 move 段 (stack_bowls 六个 episode 都如此,
      录制从夹持前一刻才开始); (b) move 段不是抓取手在做 (stand_up_bottles ep400)
    """
    segs = group_segments(rows)
    if not segs:
        return []
    grip = {s: grip_trace(arr, s) for s in ("left", "right")}

    for s in segs:
        ev, a, b = s["event"], s["start"], s["end"]
        if ev == "move":
            s["stage"], s["rep"] = "approach", b
        elif ev.startswith("move_"):
            s["stage"], s["rep"] = "transport", b
        elif ev.startswith("grasp_"):
            s["stage"], s["rep"] = "grasp", b
        elif ev.startswith("hold_"):
            s["stage"], s["rep"] = "hold", (a + b) // 2
        elif ev == "insert" or ev == "handover" or ev.startswith("place_"):
            s["stage"] = "place" if ev.startswith("place_") else ev
            s["rep"] = release_rep(grip, a, b)
        else:
            s["stage"], s["rep"] = ev, b

    first_grasp = next((i for i, s in enumerate(segs) if s["event"].startswith("grasp_")), None)
    # 抓取手 = 首个 grasp 段里闭合幅度大的那只
    grasp_arm = None
    if first_grasp is not None:
        s = segs[first_grasp]
        grasp_arm = max(("left", "right"), key=lambda k: grip[k][s["start"]] - grip[k][s["end"]])

    def moving_at(seg, arm: str) -> bool:
        """该 move 段的代表帧(段末)上确实是 arm 在动 (CSV 的 <arm>_state 非 idle)。"""
        return seg["rows"][-1][f"{arm}_state"] not in ("idle", "none", "")

    moves = [i for i, s in enumerate(segs) if s["event"] == "move"]
    approach_i = next((i for i in reversed(moves)
                       if first_grasp is not None and i < first_grasp
                       and (grasp_arm is None or moving_at(segs[i], grasp_arm))), None)
    home_i = moves[-1] if (moves and first_grasp is not None and moves[-1] > first_grasp) else None
    for i in moves:
        if i not in (approach_i, home_i):
            segs[i]["drop"] = True
    if approach_i is not None:
        segs[approach_i]["stage"] = "approach"
    if home_i is not None:
        segs[home_i]["stage"] = "home"

    if first_grasp is not None and approach_i is None:
        # 接近过程并进了首个 grasp 段: 从段内合成一帧 approach
        s = segs[first_grasp]
        a, b = s["start"], s["end"]
        pos = np.stack(arr["observation.state"].to_numpy())[:, POS[grasp_arm]]
        f = reach_rep(pos, grip[grasp_arm], a, b)
        segs.insert(first_grasp, {"event": "move", "stage": "approach", "start": f, "end": f,
                                  "rep": f, "rows": [], "synthetic": True})

    for i, s in enumerate(segs):
        if s["event"].startswith("move_"):
            # 与下一段的代表帧挤在一起时 (真实数据里 hold_* 常紧贴 move_* 结束),
            # 两帧几乎重复, 退回段中
            if i + 1 < len(segs) and segs[i + 1]["rep"] - s["rep"] < 10:
                s["rep"] = (s["start"] + s["end"]) // 2
    return dedupe_by_type([s for s in segs if not s.get("drop")])


# ---------------------------------------------------------------- 数据集档案

PROFILES = {
    "sim": {
        "root": ROOT / "data" / "sim_lerobot_v30_ee",
        "out": ROOT / "outputs" / "in_context_learning_sim",
        "segments": segments_sim,
        "annotations": ANNOTATIONS_SIM,
        # slug -> (episode 区间, 英文任务描述)
        "tasks": {
            "fill_pen_holder": ((0, 99), "Hold the pen holder upright, insert all four pens tip-up one by one, place it down, then return to origin."),
            "plug_in_charger": ((100, 199), "Insert the charger fully upright into the power-strip socket, release it, then return to origin."),
            "stack_bowls": ((200, 299), "Nest all three bowls upright, release them, then return to origin."),
        },
        # 每个任务选定的演示 episode (均为该任务视频目录 outputs/episode_insight/interactive_sim/videos
        # 里已有的全集之一; 选结构最完整者):
        #   stack_bowls 203 —— 三次完整的 抓→移→放 循环 + 收尾回位
        #   fill_pen_holder 0 —— 起筒/插 4 支笔/放筒/回位, 全流程齐全
        #   plug_in_charger 150 —— 单次交接后直接插入 (145/146/148 多一次多余的来回递接)
        "default_episodes": {"fill_pen_holder": 0, "plug_in_charger": 150, "stack_bowls": 203},
    },
    "real": {
        "root": ROOT / "data" / "real_lerobot_v30_ee",
        "out": ROOT / "outputs" / "in_context_learning_real",
        "segments": segments_real,
        "annotations": ANNOTATIONS_REAL,
        # slug 与 outputs/episode_insight/interactive_real/videos 的文件名前缀一致;
        # 任务描述原样取自 meta/tasks.parquet
        "tasks": {
            "fill_pen_holder": ((0, 99), "Pick up and hold the pen holder upright, insert all three pens one by one, then place the filled holder on the table."),
            "put_objects_into_basket": ((100, 199), "Place all objects on the table into the basket one by one."),
            "stack_and_cover_blocks": ((200, 299), "Stack all three blocks into one vertical stack, then place the cup over the entire stack."),
            "stack_bowls": ((300, 399), "Nest all three bowls upright at the center of the table."),
            "stand_up_bottles": ((400, 499), "Place all three bottles stably upright on the table."),
            "insert_charger": ((500, 599), "Insert the charger plug into the power strip, then connect the charging cable to the charger plug."),
        },
        # frame_weight.csv 只覆盖 36 集 (即 interactive_real/videos 里那 36 个),
        # 每个任务各 6 集; 选段结构最短、事件齐全的那一集:
        #   put_objects_into_basket 105 —— 该任务只有这一集有标注
        #   stack_and_cover_blocks 294 / stack_bowls 302 / stand_up_bottles 400
        #   insert_charger 503 —— 分段数最少 (34), 交接后只多 2 个来回
        "default_episodes": {
            "fill_pen_holder": 0, "put_objects_into_basket": 105,
            "stack_and_cover_blocks": 294, "stack_bowls": 302,
            "stand_up_bottles": 400, "insert_charger": 503,
        },
        # 写进 index.json, 说明本数据集上两个需要读者留意的点
        "notes": [
            "gripper: 闭合极值随被夹物粗细而异 —— 细的笔能收到 0.45 以下记 closed,"
            " 而笔筒(~0.69)、瓶子(~0.60)、充电器(~0.47) 这类粗件收不到行程末端,"
            " 即使确实夹住了也只记 partial。判读 real 的 grasp/hold 帧请以 observation 文本为准。",
            "approach: CSV 里没有对应 move 段 (或该 move 段不是抓取手在做) 时,"
            " approach 帧从首个 grasp 段内按手臂速度峰值合成, 该帧带 synthetic: true。",
        ],
    },
}


# ---------------------------------------------------------------- 数据读取

_ARRAY_CACHE: dict = {}


def load_episode_arrays(prof, episode: int):
    """某 episode 的逐帧 state/action。整表只读一次 (real 有 66 万行)。"""
    key = str(prof["root"])
    if key not in _ARRAY_CACHE:
        _ARRAY_CACHE[key] = pd.read_parquet(
            prof["root"] / "data" / "chunk-000" / "file-000.parquet",
            columns=["episode_index", "frame_index", "observation.state", "action"],
        )
    df = _ARRAY_CACHE[key]
    return df[df.episode_index == episode].sort_values("frame_index").reset_index(drop=True)


def video_index(prof, episode: int):
    ep = pd.read_parquet(prof["root"] / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    r = ep.set_index("episode_index").loc[episode]
    out = {}
    for feat in CAMS:
        out[feat] = {
            "path": prof["root"] / "videos" / feat / "chunk-000" / f"file-{int(r[f'videos/{feat}/file_index']):03d}.mp4",
            "from": float(r[f"videos/{feat}/from_timestamp"]),
        }
    return out


def grab(video: Path, ts: float, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{ts:.3f}", "-i", str(video),
         "-frames:v", "1", "-q:v", "2", str(dst)],
        check=True,
    )


# ---------------------------------------------------------------- 离散标签


def roles_of(lstate: str, rstate: str):
    return {
        "left": "idle" if lstate in ("idle", "none", "") else "active",
        "right": "idle" if rstate in ("idle", "none", "") else "active",
    }


def grip_state(g: float) -> str:
    """夹爪开合 → 离散标签 (两个数据集里 1.0 = 全开, 0.0 = 全闭)。

    阈值按实测分布定: 闭合极值随对象而异 —— sim 的碗/笔能收到 ~0.07, 充电器对象
    薄, 夹到 0.36 就到头; real 的笔筒(较粗)持筒只收到 ~0.69, 充电器插头 ~0.31。
    故 closed 取 0.45 以覆盖薄件, partial 留给"夹住但未到行程末端"。
    """
    if g >= 0.85:
        return "open"
    if g <= 0.45:
        return "closed"
    return "partial"


def grip_action(a: float, s: float) -> str:
    """动作里的夹爪目标 → 相对当前状态的离散指令。"""
    if abs(a - s) <= 0.05:
        return "keep"
    return "close" if a < s else "open"


# ---------------------------------------------------------------- 组装


def build(prof, task: str, episode: int):
    rows = load_frame_weight(prof, episode)
    row_by_frame = {int(r["frame_index"]): r for r in rows}
    arr = load_episode_arrays(prof, episode)
    segs = prof["segments"](rows, arr)
    vidx = video_index(prof, episode)
    out_dir = prof["out"] / task
    # 去重后序号会变, 旧图残留会与 demo.json 对不上 —— 先清空该任务图像目录
    shutil.rmtree(out_dir / "images", ignore_errors=True)
    keyframes = []
    for i, s in enumerate(segs):
        fidx = s["rep"]
        row = arr.iloc[fidx]
        st, ac = row["observation.state"], row["action"]
        lstate = row_by_frame[fidx]
        stage = s["stage"]
        imgs = {}
        for feat, info in vidx.items():
            tag = feat.replace("observation.images.", "")
            name = f"{i:03d}-{stage}-{tag}.jpg"
            grab(info["path"], info["from"] + fidx / FPS, out_dir / "images" / name)
            imgs[feat] = f"images/{name}"
        obs, res = prof["annotations"].get((task, episode, fidx), ("", ""))
        kf = {
            "frame_index": fidx,
            "event": s["event"],
            "stage": stage,
            "event_span": [s["start"], s["end"]],
            "observation": obs,
            "roles": roles_of(lstate["left_state"], lstate["right_state"]),
            "result": res,
            "state": {
                side: {
                    "position": [round(float(v), 4) for v in st[POS[side]]],
                    "orientation": [round(float(v), 4) for v in st[QUAT[side]]],
                    "gripper": grip_state(float(st[GRIP[side]])),
                } for side in ("left", "right")
            },
            "action": {
                side: {
                    "position": [round(float(v), 4) for v in ac[POS[side]]],
                    "orientation": [round(float(v), 4) for v in ac[QUAT[side]]],
                    "gripper": grip_action(float(ac[GRIP[side]]), float(st[GRIP[side]])),
                } for side in ("left", "right")
            },
            "images": imgs,
        }
        if s.get("synthetic"):
            kf["synthetic"] = True
        keyframes.append(kf)
    demo = {
        "task": prof["tasks"][task][1],
        "task_slug": task,
        "dataset": str(prof["root"].relative_to(ROOT)),
        "episode_index": episode,
        "fps": FPS,
        "state_layout": STATE_LAYOUT,
        "keyframes": keyframes,
    }
    out = out_dir / "demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(demo, indent=2, ensure_ascii=False))
    return {"demo": demo, "path": out}


def write_index(prof, entries):
    """index.json: task -> demo 文件 / 图像目录 的映射。"""
    index = {
        "dataset": str(prof["root"].relative_to(ROOT)),
        "fps": FPS,
        "cameras": list(CAMS),
        "gripper_labels": {"state": ["open", "partial", "closed"], "action": ["open", "close", "keep"]},
        "state_layout": STATE_LAYOUT,
        "notes": prof.get("notes", []),
        "tasks": entries,
    }
    p = prof["out"] / "index.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(PROFILES), default="sim")
    ap.add_argument("--task")
    ap.add_argument("--episode", type=int)
    ap.add_argument("--all", action="store_true", help="每个任务导 1 个 episode (取 default_episodes)")
    args = ap.parse_args()

    prof = PROFILES[args.dataset]
    jobs = list(prof["tasks"]) if args.all else [args.task]
    entries = []
    for task in jobs:
        episode = args.episode if args.episode is not None else prof["default_episodes"][task]
        res = build(prof, task, episode)
        d = res["demo"]
        entries.append({
            "slug": task,
            "task": prof["tasks"][task][1],
            "episode_index": episode,
            "demo": f"{task}/demo.json",
            "images_dir": f"{task}/images",
            "n_keyframes": len(d["keyframes"]),
            "stages": [k["stage"] for k in d["keyframes"]],
        })
        print(res["path"], f"({len(d['keyframes'])} keyframes)")
    print(write_index(prof, entries))


if __name__ == "__main__":
    main()
