#!/usr/bin/env python3
"""导出 in-context learning 演示 (demo.json + 关键帧图)。

对每个任务选 1 个 episode, 按 frame_weight.csv 的 task_event 分段, 每段取一个
代表帧, 连同该帧的 state/action 与三路相机图, 组装成可供 in-context 使用的
demo.json。

数据来源:
- 事件分段: data/sim_lerobot_v30_ee/frame_weight.csv (列 task_event / 左右 state)
- state/action: data/sim_lerobot_v30_ee/data/chunk-000/*.parquet (16 维, 左 8 + 右 8)
- 图像: data/sim_lerobot_v30_ee/videos/observation.images.<cam>/chunk-000/file-XXX.mp4
        时间戳映射取自 meta/episodes 的 videos/<cam>/from_timestamp, 帧率 25fps

★ 不直接裁切 outputs/episode_insight/interactive_sim/videos 的合成图: 那是
  cam_high + cam_left_wrist(纵向压扁) 的拼图, 右腕视角缺失且画质有损。

用法 (须用 lerobot 环境: parquet 由 arrow 25 写出, base 的 pyarrow 19 读会报
"Repetition level histogram size mismatch"):
    /opt/anaconda3/envs/lerobot/bin/python tools/in_context_demo.py --task stack_bowls --episode 203
    /opt/anaconda3/envs/lerobot/bin/python tools/in_context_demo.py --all
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "sim_lerobot_v30_ee"
OUT_ROOT = ROOT / "outputs" / "in_context_learning_sim"
FPS = 25.0
WEIGHT_BASE = 1.0   # frame_weight_loss 的窗口外取值 (见 tools/keyframe_events.py)

# 任务 slug -> (episode 区间, 英文任务描述)
TASKS = {
    "fill_pen_holder": ((0, 99), "Hold the pen holder upright, insert all four pens tip-up one by one, place it down, then return to origin."),
    "plug_in_charger": ((100, 199), "Insert the charger fully upright into the power-strip socket, release it, then return to origin."),
    "stack_bowls": ((200, 299), "Nest all three bowls upright, release them, then return to origin."),
}

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

# 人工标注 (逐帧看过 head 相机画面后撰写): (task, episode, frame_index) -> (observation, result)
# 未列出的 keyframe 只输出数值, observation/result 为空
ANNOTATIONS = {
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

# 相机键名与数据集一致 (info.json 的 features)
MOVEMENT_EVENTS = {"move", "move_bowl", "move_pen", "move_holder", "move_plug"}

# 每个任务选定的演示 episode (均为该任务视频目录 outputs/episode_insight/interactive_sim/videos
# 里已有的全集之一; 选结构最完整者):
#   stack_bowls 203 —— 三次完整的 抓→移→放 循环 + 收尾回位
#   fill_pen_holder 0 —— 起筒/插 4 支笔/放筒/回位, 全流程齐全
#   plug_in_charger 150 —— 单次交接后直接插入 (145/146/148 多一次多余的来回递接)
DEFAULT_EPISODES = {"fill_pen_holder": 0, "plug_in_charger": 150, "stack_bowls": 203}

CAMS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
SIDES = {0: "left", 1: "right"}
POS = {"left": slice(0, 3), "right": slice(8, 11)}
QUAT = {"left": slice(3, 7), "right": slice(11, 15)}
GRIP = {"left": 7, "right": 15}


def load_frame_weight(episode: int):
    """frame_weight.csv 里某 episode 的逐帧行, 按 frame_index 排序。"""
    with open(DATASET / "frame_weight.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["episode_index"]) == episode]
    rows.sort(key=lambda r: int(r["frame_index"]))
    return rows


def load_episode_segments(rows, min_len: int = 3):
    """按 task_event 切段, 每段定一个代表帧。

    代表帧 = 关键帧平台 [t0+Pl, t0+Pr] 的**末**帧 = 事件锚点 t0
    (configs/keyframe_weight_config.json: grasp 取 hold_start = 有效抓取形成帧;
     place/insert/handover-release 取 hold_end = 夹爪开始张开帧)。v2 窗口
    Pl=-5, Pr=0, 故平台末帧即 t0。实测: ep0 grasp_holder 平台 79-84, 闭爪在
    80-84 完成 → t0=84; ep203 place_bowl 平台 98-103, 夹爪 104 才张开 → t0=103。
    """
    segs = []
    for ev, grp in itertools.groupby(rows, key=lambda r: r["task_event"]):
        g = list(grp)
        if len(g) < min_len or ev in ("idle",):
            continue
        wmax = max(float(r["frame_weight_loss"]) for r in g)
        # wmax > 1 说明本段含自己事件的权重平台 (抓取/放置类); 否则是纯移动段
        anchored = wmax > WEIGHT_BASE
        last_max = max(g, key=lambda r: (float(r["frame_weight_loss"]), int(r["frame_index"])))
        segs.append({
            "event": ev,
            "start": int(g[0]["frame_index"]),
            "end": int(g[-1]["frame_index"]),
            "rep": int(last_max["frame_index"]) if anchored else int(g[-1]["frame_index"]),
            "anchored": anchored,
        })
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


def dedupe_by_type(segs):
    """每个事件**类型**只留 1 帧, 取该类型首次出现的那一段。

    类型键 = (stage, event)。用 event 而非只 stage, 是为了让 fill 的
    grasp_holder / grasp_pen、place_holder / place_pen 各自成类 —— 否则
    4 个插笔循环会被压成同一个 grasp/place, 持筒与持笔就分不开了。
    后续重复的插笔循环(grasp_pen / move_pen / place_pen 再次出现)随之丢弃。
    """
    seen, out = set(), []
    for s in segs:
        key = (s["stage"], s["event"])
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def load_episode_arrays(episode: int):
    df = pd.read_parquet(DATASET / "data" / "chunk-000" / "file-000.parquet")
    e = df[df.episode_index == episode].sort_values("frame_index").reset_index(drop=True)
    return e


def video_index(episode: int):
    ep = pd.read_parquet(DATASET / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    r = ep.set_index("episode_index").loc[episode]
    out = {}
    for feat in CAMS:
        out[feat] = {
            "path": DATASET / "videos" / feat / "chunk-000" / f"file-{int(r[f'videos/{feat}/file_index']):03d}.mp4",
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


def roles_of(lstate: str, rstate: str):
    return {
        "left": "idle" if lstate in ("idle", "none", "") else "active",
        "right": "idle" if rstate in ("idle", "none", "") else "active",
    }


def grip_state(g: float) -> str:
    """夹爪开合 → 离散标签 (数据集里 1.0 = 全开, 0.0 = 全闭)。

    阈值按本数据集实测分布定: 三任务的闭合极值不同 —— 碗/笔能收到 ~0.07,
    但充电器对象薄, 夹到 0.36 就到头; 笔筒较粗, 持筒只收到 ~0.6。
    故 closed 取 0.45 以覆盖充电器, partial 留给"夹住但未到行程末端"(如持筒)。
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


def build(task: str, episode: int):
    rows = load_frame_weight(episode)
    row_by_frame = {int(r["frame_index"]): r for r in rows}
    segs = load_episode_segments(rows)
    arr = load_episode_arrays(episode)
    vidx = video_index(episode)
    # 去重后序号会变, 旧图残留会与 demo.json 对不上 —— 先清空该任务图像目录
    shutil.rmtree(OUT_ROOT / task / "images", ignore_errors=True)
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
            grab(info["path"], info["from"] + fidx / FPS, OUT_ROOT / task / "images" / name)
            imgs[feat] = f"images/{name}"
        obs, res = ANNOTATIONS.get((task, episode, fidx), ("", ""))
        keyframes.append({
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
        })
    demo = {
        "task": TASKS[task][1],
        "task_slug": task,
        "episode_index": episode,
        "fps": FPS,
        "state_layout": "left[x,y,z,qw,qx,qy,qz,gripper] + right[same]",
        "keyframes": keyframes,
    }
    out = OUT_ROOT / task / "demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(demo, indent=2, ensure_ascii=False))
    return {"demo": demo, "path": out}


def write_index(entries):
    """index.json: task -> demo 文件 / 图像目录 的映射。"""
    index = {
        "dataset": "data/sim_lerobot_v30_ee",
        "fps": FPS,
        "cameras": list(CAMS),
        "gripper_labels": {"state": ["open", "partial", "closed"], "action": ["open", "close", "keep"]},
        "state_layout": "left[x,y,z,qw,qx,qy,qz,gripper] + right[same]",
        "tasks": entries,
    }
    p = OUT_ROOT / "index.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--episode", type=int)
    ap.add_argument("--all", action="store_true", help="每个任务导 1 个 episode (任务区间首集)")
    args = ap.parse_args()

    jobs = list(TASKS.items()) if args.all else [(args.task, None)]
    entries = []
    for task, span in jobs:
        episode = args.episode if args.episode is not None else DEFAULT_EPISODES[task]
        res = build(task, episode)
        d = res["demo"]
        entries.append({
            "slug": task,
            "task": TASKS[task][1],
            "episode_index": episode,
            "demo": f"{task}/demo.json",
            "images_dir": f"{task}/images",
            "n_keyframes": len(d["keyframes"]),
            "stages": [k["stage"] for k in d["keyframes"]],
        })
        print(res["path"], f"({len(d['keyframes'])} keyframes)")
    print(write_index(entries))


if __name__ == "__main__":
    main()
