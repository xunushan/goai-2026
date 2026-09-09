#!/usr/bin/env python3
"""sim_lerobot_v30_ee 数据集关键帧标定：追加 is_keyframe 列 + 统计占比 + 可视化。

对每个 episode 的左右爪夹（observation.state idx7=left, idx15=right）复用
tools/keyframe_detect.py 的 detect_gripper_keyframes（参数一致）检测四类关键点
close_start / hold_start / hold_end / open_full（不完整周期的 open_full=-1 不计）,
把"任一爪任一关键点命中该帧"的帧标记为关键帧：

    is_keyframe = 1    帧命中任一关键点
    is_keyframe = 0    其它帧

与 tools/frame_weight.py 不同: frame_weight 把 hold 前后扩成加权窗口, 本脚本只标
离散关键点单帧（即 keyframe_detect.py 竖线画出的那些点）, 统计其占全帧的比例,
供评估"关键帧采样/加权"是否够密集。

输出（默认全部落在 outputs/keyframe_annotate_sim_v30ee/, 不改动 data/ 源文件）:
    sim_lerobot_v30_ee_keyframed.csv   源 CSV 全列副本 + is_keyframe(0/1) 列
    keyframe_stats.json                每任务/每 episode 关键帧明细与占比统计
    plots/{slug}_ep{ep:03d}.png        每任务抽样 episode 的关键帧时序可视化

sim 数据集与 tools/keyframe_detect.py 针对的 real 12 任务不同, 只有 3 个 sim 任务
(fill_pen_holder / plug_in_charger / stack_bowls), 故 slug/任务名从
meta/tasks.parquet 加载, slug 按本数据集固定映射。

用法:
    python tools/keyframe_annotate.py                          # 全任务, 每任务 6 ep
    python tools/keyframe_annotate.py --per-task 6 --seed 0 --out outputs/xxx
    python tools/keyframe_annotate.py --no-incomplete          # 不标记结尾未回开的抓取
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

from tools.keyframe_detect import (  # noqa: E402
    GRIP_L,
    GRIP_R,
    INCOMPLETE,
    detect_gripper_keyframes,
    plot_episode_keyframes,
)

DEFAULT_CSV = ROOT / "data" / "sim_lerobot_v30_ee" / "sim_lerobot_v30_ee.csv"
DEFAULT_TASKS = ROOT / "data" / "sim_lerobot_v30_ee" / "meta" / "tasks.parquet"
DEFAULT_SPLIT = ROOT / "data" / "sim_lerobot_v30_ee" / "train_val_split.json"
DEFAULT_OUT = ROOT / "outputs" / "keyframe_annotate_sim_v30ee"

# sim 数据集固定 3 任务 task_index -> slug（与 data/sim_lerobot_v30_ee 一致）
TASK_SLUGS: dict[int, str] = {
    0: "fill_pen_holder",
    1: "plug_in_charger",
    2: "stack_bowls",
}

KEYFRAME_TYPES = ["close_start", "hold_start", "hold_end", "open_full"]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_tasks(tasks_parquet: Path) -> dict[int, str]:
    """读取 tasks.parquet -> {task_index: 完整指令}。"""
    import pyarrow.parquet as pq

    t = pq.read_table(str(tasks_parquet)).to_pandas()
    return {int(v): str(k) for k, v in t["task_index"].items()}


def load_state_cols(csv_path: Path) -> pd.DataFrame:
    """读取 CSV 中检测所需的列 (episode/task/frame/state)。"""
    return pd.read_csv(
        csv_path,
        usecols=["episode_index", "task_index", "frame_index", "observation.state"],
    )


def parse_state_series(s: pd.Series) -> np.ndarray:
    """把 observation.state 字符串列解析为 (N,16) 数组。"""
    return np.asarray(
        [np.fromstring(x.strip("[]"), sep=",", dtype=float) for x in s],
        dtype=float,
    )


def load_train_episodes(split_json: Path) -> set[int] | None:
    """读取 train_val_split.json -> train episode_index 集合 (无则 None)。"""
    if not split_json.is_file():
        return None
    d = json.loads(split_json.read_text(encoding="utf-8"))
    train: set[int] = set()
    for task_cfg in d["tasks"].values():
        train.update(int(e) for e in task_cfg["train_episode_idx"])
    return train


# ---------------------------------------------------------------------------
# 关键帧检测（逐 episode）
# ---------------------------------------------------------------------------

def _to_frame_set(kf: dict[str, np.ndarray], frame: np.ndarray) -> tuple[set[int], int]:
    """单爪夹关键点 position 数组 -> (命中 frame_index 集合, 其中不完整 open_full 数)。"""
    hit: set[int] = set()
    incomplete = 0
    for t in KEYFRAME_TYPES:
        for p in kf[t]:
            if p < 0:  # INCOMPLETE 哨兵(-1), 无对应帧
                incomplete += 1
                continue
            hit.add(int(frame[p]))
    return hit, incomplete


def annotate_episode(
    state: np.ndarray,
    frame: np.ndarray,
    detect_kwargs: dict,
) -> dict:
    """单个 episode 标注 -> dict(is_keyframe 逐行 bool, 每侧关键点明细)。"""
    per_side = {}
    union: set[int] = set()
    for side, idx in (("left", GRIP_L), ("right", GRIP_R)):
        kf = detect_gripper_keyframes(state[:, idx], frame, **detect_kwargs)
        hit, n_incomplete = _to_frame_set(kf, frame)
        per_side[side] = {
            t: int(np.sum(kf[t] >= 0)) for t in KEYFRAME_TYPES
        }
        per_side[side]["n_cycles"] = int(len(kf["close_start"]))
        per_side[side]["n_incomplete"] = n_incomplete
        per_side[side]["incomplete_any"] = bool((kf["open_full"] == INCOMPLETE).any())
        union |= hit
    mask = np.zeros(len(state), dtype=bool)
    if union:
        # frame 为逐行 frame_index; union 存 frame_index -> 反查行号
        pos = np.where(np.isin(frame, sorted(union)))[0]
        mask[pos] = True
    return {"mask": mask, "n_keyframes": int(mask.sum()), "per_side": per_side}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="sim CSV 路径")
    parser.add_argument("--tasks", default="all",
                        help="要标注的 task_index, 逗号分隔 (如 '0,2') 或 'all'")
    parser.add_argument("--per-task", type=int, default=6,
                        help="每任务抽样绘制的 episode 数")
    parser.add_argument("--min-prominence", type=float, default=0.2,
                        help="运动段幅度下限(低于此的抖动/浅捏并入平段)")
    parser.add_argument("--hold-min-len", type=int, default=3,
                        help="水平保持段最少帧数")
    parser.add_argument("--open-level", type=float, default=0.9,
                        help="视作打开的阈值(周期起点/终点回到该值)")
    parser.add_argument("--no-incomplete", action="store_true",
                        help="不标记结尾未回开(不完整)的抓取周期")
    parser.add_argument("--seed", type=int, default=0, help="抽样种子(可复现)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="输出目录 (不写源数据)")
    args = parser.parse_args()

    detect_kwargs = {
        "min_prominence": args.min_prominence,
        "hold_min_len": args.hold_min_len,
        "open_level": args.open_level,
        "allow_incomplete": not args.no_incomplete,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_names = load_tasks(DEFAULT_TASKS) if DEFAULT_TASKS.is_file() else {}
    train_eps = load_train_episodes(DEFAULT_SPLIT)

    print(f"loading {args.csv} ...")
    csv_path = Path(args.csv)
    small = load_state_cols(csv_path)
    episodes_by_task = (
        small.groupby("task_index")["episode_index"]
        .apply(lambda s: np.sort(s.unique()))
        .to_dict()
    )
    selected_tasks = (
        list(episodes_by_task)
        if args.tasks == "all"
        else [int(x) for x in args.tasks.split(",")]
    )

    # ---- 1) 逐 episode 检测, 汇总标注掩码与统计 ----
    key_rows: list[pd.DataFrame] = []
    n_ep = n_frame = n_kf = 0
    train_frame = train_kf = 0
    per_episode: dict[str, dict] = {}
    per_task: dict[str, dict] = {}
    for ti in selected_tasks:
        if ti not in episodes_by_task:
            print(f"  ! task_index={ti} 不存在, 跳过")
            continue
        eps = episodes_by_task[ti]
        name = task_names.get(ti, TASK_SLUGS.get(ti, f"task_{ti:03d}"))
        t_frames = t_kf = 0
        side_total = {"left": {t: 0 for t in KEYFRAME_TYPES},
                      "right": {t: 0 for t in KEYFRAME_TYPES}}
        for ep in [int(e) for e in eps]:
            sub = small[small["episode_index"] == ep].sort_values("frame_index")
            state = parse_state_series(sub["observation.state"])
            frame = sub["frame_index"].to_numpy()
            res = annotate_episode(state, frame, detect_kwargs)
            key_rows.append(pd.DataFrame({
                "episode_index": ep,
                "frame_index": frame,
                "is_keyframe": res["mask"].astype(np.uint8),
            }))
            n_frame += len(frame)
            n_kf += res["n_keyframes"]
            t_frames += len(frame)
            t_kf += res["n_keyframes"]
            if train_eps is not None and ep in train_eps:
                train_frame += len(frame)
                train_kf += res["n_keyframes"]
            for side in ("left", "right"):
                for t in KEYFRAME_TYPES:
                    side_total[side][t] += res["per_side"][side][t]
            per_episode[f"task{ti}_ep{ep}"] = {
                "task_index": ti, "episode_index": ep,
                "n_frames": int(len(frame)),
                "n_keyframes": res["n_keyframes"],
                "keyframe_pct": 100.0 * res["n_keyframes"] / len(frame),
                **{f"{side}.{t}": res["per_side"][side][t]
                   for side in ("left", "right") for t in KEYFRAME_TYPES},
                "incomplete_left": res["per_side"]["left"]["incomplete_any"],
                "incomplete_right": res["per_side"]["right"]["incomplete_any"],
            }
        n_ep += len(eps)
        per_task[str(ti)] = {
            "instruction": name,
            "episodes": int(len(eps)),
            "frames": int(t_frames),
            "keyframes": int(t_kf),
            "keyframe_pct": 100.0 * t_kf / t_frames,
            **{f"left.{t}": side_total["left"][t] for t in KEYFRAME_TYPES},
            **{f"right.{t}": side_total["right"][t] for t in KEYFRAME_TYPES},
        }

    # ---- 2) 写标注副本 CSV ----
    full = pd.read_csv(str(csv_path))
    key = pd.concat(key_rows, ignore_index=True)
    full = full.merge(key, on=["episode_index", "frame_index"], how="left",
                      validate="one_to_one")
    missing = int(full["is_keyframe"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} 行未对齐到 episode/frame key")
    keyframed_path = out_dir / f"{csv_path.stem}_keyframed.csv"
    full.to_csv(keyframed_path, index=False)
    print(f"annotated copy -> {keyframed_path}")

    # ---- 3) 统计与 JSON ----
    total_pct = 100.0 * n_kf / n_frame
    stats = {
        "csv": str(csv_path),
        "schema": "is_keyframe(0/1): 帧命中左右爪任一关键点(close_start/hold_start/"
                  "hold_end/open_full)为1, 不完整周期open_full(-1)不计",
        "detect_kwargs": detect_kwargs,
        "total": {"episodes": n_ep, "frames": n_frame,
                  "keyframes": n_kf, "keyframe_pct": round(total_pct, 4)},
        "per_task": per_task,
        "per_episode": per_episode,
    }
    if train_eps is not None:
        stats["train_subset"] = {
            "episodes": len(train_eps),
            "frames": train_frame, "keyframes": train_kf,
            "keyframe_pct": round(100.0 * train_kf / train_frame, 4),
        }
    stats_path = out_dir / "keyframe_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    # 控制台表格
    print(f"{'task':>4} {'slug':<18} {'episodes':>8} {'frames':>8} "
          f"{'keyframes':>9} {'pct':>7}")
    for ti in selected_tasks:
        s = per_task.get(str(ti))
        if not s:
            continue
        slug = TASK_SLUGS.get(ti, f"task_{ti:03d}")
        print(f"{ti:>4} {slug:<18} {s['episodes']:>8} {s['frames']:>8} "
              f"{s['keyframes']:>9} {s['keyframe_pct']:>6.2f}%")
    print(f"总计  : {n_ep:>5} episodes / {n_frame} frames, "
          f"{n_kf} keyframes ({total_pct:.2f}%)")
    if train_eps is not None:
        tr = stats["train_subset"]
        print(f"train  : {tr['episodes']} episodes / {tr['frames']} frames, "
              f"{tr['keyframes']} keyframes ({tr['keyframe_pct']:.2f}%)")

    # ---- 4) 每任务抽样绘制关键帧时序图 ----
    rng = np.random.default_rng(args.seed)
    plot_dir = out_dir / "plots"
    n_plot = 0
    for ti in selected_tasks:
        if ti not in episodes_by_task:
            continue
        eps = episodes_by_task[ti]
        pick = rng.choice(eps, size=min(args.per_task, len(eps)), replace=False)
        slug = TASK_SLUGS.get(ti, f"task_{ti:03d}")
        name = task_names.get(ti, slug)
        for ep in [int(x) for x in pick]:
            sub = small[small["episode_index"] == ep].sort_values("frame_index")
            state = parse_state_series(sub["observation.state"])
            frame = sub["frame_index"].to_numpy()
            kf = {
                "left": detect_gripper_keyframes(state[:, GRIP_L], frame,
                                                 **detect_kwargs),
                "right": detect_gripper_keyframes(state[:, GRIP_R], frame,
                                                  **detect_kwargs),
            }
            out_png = plot_dir / f"{slug}_ep{ep:03d}.png"
            plot_episode_keyframes(name, ep, frame, state, kf, out_png)
            n_plot += 1
    print(f"\n{len(selected_tasks)} 任务 × 每任务 {args.per_task} episode 可视化 "
          f"-> {plot_dir} ({n_plot} 张)")
    print(f"stats -> {stats_path}")


if __name__ == "__main__":
    main()
