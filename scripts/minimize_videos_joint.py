"""3 任务 × 6 episode 的最少视频文件联合优化。

每个 episode 在 3 路相机各对应一个文件, 三元组 (high,left,right)。
一致 episode 集合 = 每个任务选 6 个 episode, 下载文件 = 这 18 个 episode 三路文件的并集。

对每任务: 统计 (high,left,right) 三元组 -> episode 列表。
若某任务存在含 >=6 episode 的三元组, 则该任务可用 3 个文件覆盖 6 个 episode。
联合优化: 每任务挑一个三元组, 使三任务文件并集最小。

用法:
    conda run -n lerobot python scripts/minimize_videos_joint.py
    conda run -n lerobot python scripts/minimize_videos_joint.py --tasks 0,3,5
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pyarrow.parquet as pq

BASE = Path("data/real_lerobot_v30_ee")


def load() -> tuple[dict, dict]:
    meta = pq.read_table(str(BASE / "meta" / "episodes" / "chunk-000" / "file-000.parquet")).to_pandas()
    data = pq.read_table(str(BASE / "data" / "chunk-000" / "file-000.parquet")).to_pandas()
    ep_task = {int(e): int(g) for e, g in data.groupby("episode_index")["task_index"].first().items()}
    cams = ["high", "left_wrist", "right_wrist"]
    ep_files = {}
    for _, r in meta.iterrows():
        e = int(r["episode_index"])
        ep_files[e] = tuple(
            (int(r[f"videos/observation.images.cam_{c}/chunk_index"]),
             int(r[f"videos/observation.images.cam_{c}/file_index"]))
            for c in cams
        )
    return ep_task, ep_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=None,
                        help="要覆盖的任务 (如 0,3,5 或 0,1,2,3,4,5); 缺省遍历全部 3 任务组合")
    parser.add_argument("--per-task", type=int, default=6)
    args = parser.parse_args()

    ep_task, ep_files = load()
    all_tasks = sorted(set(ep_task.values()))

    # 每任务: 三元组 -> episodes
    triple_eps = {}
    for t in all_tasks:
        agg = defaultdict(list)
        for ep, ti in ep_task.items():
            if ti == t:
                agg[ep_files[ep]].append(ep)
        triple_eps[t] = {tr: sorted(eps) for tr, eps in agg.items()}

    def union_files(sel: dict) -> set:
        """sel: {task: (triple, eps)} -> (camera, chunk, file) 并集"""
        return {
            (CAM_KEYS[ci], tr[ci][0], tr[ci][1])
            for tr, _ in sel.values()
            for ci in range(3)
        }

    combos = [tuple(int(x) for x in args.tasks.split(","))] if args.tasks else list(combinations(all_tasks, 3))

    # 三元组中的相机索引顺序: high, left_wrist, right_wrist
    CAM_KEYS = ["high", "left_wrist", "right_wrist"]

    results = []
    for combo in combos:
        # 每任务可选三元组: 只保留含 >=per_task 个 episode 的
        cand = [
            [(tr, eps) for tr, eps in triple_eps[t].items() if len(eps) >= args.per_task]
            for t in combo
        ]
        # 若无解, 记录并跳过
        if any(len(c) == 0 for c in cand):
            results.append((float("inf"), combo, None))
            print(f"任务 {combo}: 无解 (某任务无 >= {args.per_task} episode 的三元组)")
            continue
        best_sel, best_n = None, float("inf")
        import itertools
        for pick in itertools.product(*cand):
            sel = dict(zip(combo, pick))
            n = len(union_files(sel))
            if n < best_n:
                best_n, best_sel = n, sel
        results.append((best_n, combo, best_sel))
        print(f"任务 {combo}: 需下载 {best_n} 个视频文件")

    if len(results) > 1:
        print("\n=== Top5 最少文件组合 ===")
        for n, combo, _ in sorted(results)[:5]:
            print(f"  任务 {combo}: {n} 文件")

    best_n, best_combo, best_sel = min(results)
    print(f"\n=== 最优: 任务 {best_combo}, {best_n} 个视频文件 ===")
    all_files = set()
    for t in best_combo:
        tr, eps = best_sel[t]
        print(f"  task {t}: 三元组 {tr}, episodes {eps[:args.per_task]} (共{len(eps)}个可用)")
        for ci in range(3):
            all_files.add((CAM_KEYS[ci], tr[ci][0], tr[ci][1]))
    print("\n=== 需下载的视频文件 ===")
    for cam, c, i in sorted(all_files):
        print(f"  videos/observation.images.cam_{cam}/chunk-{c:03d}/file-{i:03d}.mp4")

    # 一致 episode 白名单
    eps_whitelist = sorted(e for t in best_combo for e in best_sel[t][1][:args.per_task])
    print(f"\n--episodes 白名单: {','.join(map(str, eps_whitelist))}")


if __name__ == "__main__":
    main()
