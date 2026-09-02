"""分析 lerobot 数据集元数据：任务分布、episode 覆盖的视频文件，及最小视频集合。

步骤:
1. 打印 info.json / tasks.parquet 摘要
2. 打印每任务 episode 数
3. 找出指定任务集合的指定数量 episode，映射到三路相机视频文件
4. 用贪心/精确法计算覆盖这些 episode 所需的最少视频文件集合

用法 (默认 data/real_lerobot_v30_ee; 其他数据集用 --data 指定):
    conda run -n lerobot python scripts/analyze_real_lerobot.py --tasks 0,1,2,3,4,5
    conda run -n lerobot python scripts/analyze_real_lerobot.py --data data/sim_lerobot_v30_ee --tasks 0,1,2 --per-task 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

BASE = Path("data/real_lerobot_v30_ee")

CAMERA_PREFIX = {
    "high": "videos/observation.images.cam_high",
    "left": "videos/observation.images.cam_left_wrist",
    "right": "videos/observation.images.cam_right_wrist",
}


def load_episode_meta() -> dict:
    """读取 episodes 元数据 -> {episode_index: 行dict}"""
    path = BASE / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(str(path)).to_pandas()
    # 行索引=episode_index, 需要按 episode_index 列
    if "episode_index" in t.columns:
        return {int(r["episode_index"]): r.to_dict() for _, r in t.iterrows()}
    # 某些数据集没有 episode_index 列, 行索引即 episode
    return {int(i): r.to_dict() for i, r in t.iterrows()}


def main() -> None:
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(BASE), help="数据集根目录")
    parser.add_argument("--tasks", default=None,
                        help="要分析的任务索引, 逗号分隔 (默认: 打印全部)")
    parser.add_argument("--per-task", type=int, default=6, help="每任务 episode 数")
    args = parser.parse_args()

    BASE = Path(args.data)
    info = json.load(open(BASE / "meta" / "info.json"))
    print("=== info.json ===")
    for k in ["total_episodes", "total_frames", "total_tasks", "fps"]:
        print(f"  {k}: {info[k]}")

    tasks = pq.read_table(str(BASE / "meta" / "tasks.parquet")).to_pandas()
    print("\n=== tasks.parquet ===")
    task_names = {}
    for idx, row in tasks.iterrows():
        task_names[int(row["task_index"])] = str(idx)
        print(f"  task_index={row['task_index']}: {idx}")

    meta = load_episode_meta()
    print(f"\n=== episodes: {len(meta)} total ===")

    # 需要读数据表确定 episode -> task 映射（episodes 元数据不一定含 task_index）
    data = pq.read_table(str(BASE / "data" / "chunk-000" / "file-000.parquet")).to_pandas()
    cols = data.columns
    print("  data columns sample:", [c for c in cols if "task" in c.lower() or "episode" in c.lower()][:10])

    # 确认 episode -> task
    if "task_index" in data.columns:
        ep_task = {int(e): int(g) for e, g in data.groupby("episode_index")["task_index"].first().items()}
    else:
        ep_task = {e: -1 for e in meta}

    from collections import Counter, defaultdict
    task_eps = defaultdict(list)
    for ep, ti in ep_task.items():
        task_eps[ti].append(ep)
    print("\n=== 每任务 episode 数 ===")
    for ti in sorted(task_eps):
        print(f"  task {ti:2d} ({task_names.get(ti,'?'):25s}): {len(task_eps[ti]):3d} episodes")

    # 视频覆盖分析: 每个 episode 对应三路相机的 (chunk, file)
    ep_video_file = {}
    for ep, row in meta.items():
        f = {}
        for key, prefix in CAMERA_PREFIX.items():
            chunk = int(row[f"{prefix}/chunk_index"])
            filei = int(row[f"{prefix}/file_index"])
            f[key] = (chunk, filei)
        ep_video_file[ep] = f
    # 统计有多少不同视频文件
    all_files = set()
    for f in ep_video_file.values():
        all_files.update((key, c, i) for key, (c, i) in f.items())
    print(f"\n=== 全部 episode 覆盖的唯一视频文件: {len(all_files)} (每路相机一个文件也算) ===")

    # 给定任务集合, 选 per-task 个 episode, 求最少覆盖这些 episode 的视频文件
    sel_tasks = [int(x) for x in args.tasks.split(",")] if args.tasks else None
    if sel_tasks is not None:
        print(f"\n=== 选定任务: {sel_tasks}, 每任务 {args.per_task} 个 episode ===")
        import numpy as np
        from itertools import combinations

        for ti in sel_tasks:
            eps = task_eps.get(ti, [])
            print(f"  task {ti} ({task_names.get(ti,'?'):25s}): {len(eps)} episodes available")
            if len(eps) >= args.per_task:
                print(f"     示例 episodes: {eps[:10]}")

        # 对每个任务选 per-task 个 episode, 找到覆盖所有选定 episode 的最少视频文件
        # 先选择每个任务的前 per_task 个 episode (按 episode_index 排序)
        chosen = {}
        for ti in sel_tasks:
            eps = sorted(task_eps.get(ti, []))[:args.per_task]
            chosen[ti] = eps
        need_eps = set(e for eps in chosen.values() for e in eps)
        print(f"\n  选定 episode 集合: {sorted(need_eps)}")

        # episode -> 需要的视频文件 (三路)
        need_videos = set()
        for ep in need_eps:
            for key, (c, i) in ep_video_file[ep].items():
                need_videos.add((key, c, i))
        print(f"  三路相机共涉及 {len(need_videos)} 个视频文件:")
        from collections import defaultdict
        by_cam = defaultdict(list)
        for key, c, i in need_videos:
            by_cam[key].append(i)
        for key in by_cam:
            print(f"    {key}: {sorted(set(by_cam[key]))}")

        # 最小覆盖: 每个相机独立找最小文件集合覆盖该相机的文件索引
        # 由于每路相机文件是独立下载的, 总最小 = 各相机最小文件数之和 (无共享)
        total = 0
        for key in ["high", "left", "right"]:
            files_needed = sorted(set(by_cam[key]))
            # 该相机需要下载这些文件
            print(f"    -> 相机 {key} 需下载 {len(files_needed)} 个文件: "
                  f"{[f'file-{i:03d}.mp4' for i in files_needed]}")
            total += len(files_needed)
        print(f"  总需下载视频文件数: {total}")

        # 供 episode_video_sync 使用的 episode 白名单输出
        print(f"\n  --episodes 白名单: {','.join(map(str, sorted(need_eps)))}")


def _debug_file_mapping():
    """打印三路相机 文件 -> episode 范围 映射 (调试用)。"""
    from collections import defaultdict
    meta = load_episode_meta()
    data = pq.read_table(str(BASE / "data" / "chunk-000" / "file-000.parquet")).to_pandas()
    ep_task = {int(e): int(g) for e, g in data.groupby("episode_index")["task_index"].first().items()}
    for cam in ["high", "left_wrist", "right_wrist"]:
        prefix = f"videos/observation.images.cam_{cam}"
        agg = defaultdict(list)
        for _, row in meta.items():
            e = row["episode_index"]
            agg[(row[f"{prefix}/chunk_index"], row[f"{prefix}/file_index"])].append(e)
        print(f"=== {cam}: file -> episode 范围 ===")
        for (c, i), eps in sorted(agg.items()):
            eps = sorted(eps)
            t = ep_task.get(eps[0])
            print(f"  {cam} file-{i:03d}: episodes {min(eps)}-{max(eps)} (n={len(eps)}), task={t}")


if __name__ == "__main__":
    main()
