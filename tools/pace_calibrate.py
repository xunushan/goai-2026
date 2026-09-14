#!/usr/bin/env python
"""PACE 离线标定与验证（arXiv:2606.00537v2）。

两个子命令：

- ``calibrate``：按论文 §3.3「每个任务用训练示范标定一次 δ_T」，在
  ``data/sim_lerobot_v30_ee`` 的**训练** split 上统计示范动作 chunk 的谷值
  prominence 分布，取第 ρ 百分位作为该任务的 δ_T，输出可直接粘进 deploy.yml
  ``pace.threshold_delta_by_task`` 的 YAML 片段 + JSON。
- ``validate``：在**验证** split 上做相位边界命中率验证。地面真值是
  ``tools/keyframe_events.py`` 引擎给出的 per-arm 关键事件帧 t0
  （grasp/place/insert 等），检验 PACE 选出的重规划边界是否落在这些相位切换点
  附近，并与固定 horizon 基线（含论文 Table 4 的 matched-horizon 对照）比较。

口径说明（论文未规定、这里的取法）

- 示范 chunk 长度取``--chunk-len``，默认与 X-VLA 的 ``num_actions`` 一致（30），
  这样标定出的 δ_T 与推理时在 length-L chunk 上算出的 Φ 同尺度；标定按
  ``--stride``（默认 1）在每条示范轨迹上滑窗。
- 命中判定：事件 t0 与任一同侧重规划边界的距离 ``<= --tolerance``（默认 5 帧，
  25 fps 下 = 0.2 s）算命中。
- 只统计 ``t0 <= T - chunk_len`` 的事件（滑窗模拟到轨迹末尾前一个完整 chunk）。

运行环境：``/opt/anaconda3/envs/lerobot/bin/python``（pandas/numpy/matplotlib）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 仓库根目录上 path，使 `tools.` 命名空间包可导入（keyframe_events 内部依赖
# `from tools.keyframe_detect import ...`，与 tools/frame_weight_trapezoid.py 同法）；
# 再把 X_VLA 策略目录上 path，直接复用策略服务 import 的那一份 pace.py。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RoboDojo" / "XPolicyLab" / "policy" / "X_VLA"))
sys.path.insert(0, str(ROOT))

from tools import keyframe_events as ke  # noqa: E402  (共享事件引擎)

from pace import (  # noqa: E402  (与被策略服务 import 的同一份实现)
    ARM_XYZ_IDX,
    PaceConfig,
    arm_speed_profile,
    select_execution_horizon,
    smooth_profile,
    valley_prominences,
)

DEFAULT_CSV = ke.DEFAULT_CSV
DEFAULT_SPLIT = ROOT / "data" / "sim_lerobot_v30_ee" / "train_val_split.json"
DEFAULT_OUT = ROOT / "outputs" / "pace_validation"

# 数据集 task_index → 评测侧任务名（RoboDojo task/RoboDojo/config/<name>.yml，
# 也是 deploy.yml 的 task_name）。stack_blocks 无示范数据，只能吃全局兜底 δ_T。
TASK_INDEX_TO_NAME: dict[int, str] = {
    0: "fill_pen_holder",
    1: "plug_in_charger",
    2: "stack_bowls",
}

# 标定报告的百分位网格。注意默认 ρ=5（论文 Table 3 的默认值）在本数据上**不可用**：
# 绝对 prominence 池被静止段的 float32 量化抖动占满（示范动作以 float32 存储，静止步
# 位移是 2^-24 的整数倍，实测左臂 23.6% 的步位移为 0、39.5% < 1e-5），p5 ≈ 1e-8
# 落在浮点噪声层，会让几乎所有谷值被接受、h 塌到 h_min。见 PACE_NOISE_FLOOR 告警。
# 复现论文报告的「平均执行视野 24.3/50」需要 ρ≈70~85，见 tools/README 的验证记录。
QUANTILE_GRID: tuple[int, ...] = (
    0, 1, 2, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 97, 99,
)

# 低于该位移（米/步）的 prominence 不可能是真实减速：示范静止步的 float32 量化抖动
# 恰好在这个量级。标定结果落在此处时给出告警（不自动改值，保持论文口径可复现）。
PACE_NOISE_FLOOR = 1e-6

ARMS = ("left", "right")
# sim_lerobot_v30_ee.csv 的 16 维布局（meta/info.json features.observation.state.names）：
#   每臂 xyz(3) + 四元数 wxyz(4) + 夹爪(1)，左臂 0:8、右臂 8:16。
CSV_ARM_SLICES = {"left": slice(0, 8), "right": slice(8, 16)}


# ---------------------------------------------------------------------------
# 数据加载：把 CSV 的 16 维（xyz + wxyz + 夹爪）铺成 X-VLA 的 20 维布局
# （xyz + rotate6d + 夹爪），使离线工具与推理走完全相同的 ARM_XYZ_IDX。
# ---------------------------------------------------------------------------

def _quat_wxyz_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """wxyz → rotate6d（= 旋转矩阵前两列按列优先展平）。

    与 model.py::quat_to_rotate6d 同一约定；这里只用 numpy 实现，避免把 torch
    依赖带进离线工具。PACE 默认只吃 xyz（`profile: xyz`），rot6d 仅用于把 20 维
    布局填完整。
    """
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = np.where(norm < 1e-8, np.array([1.0, 0.0, 0.0, 0.0]), quat)
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    rot = np.empty((quat.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot[:, :, :2].reshape(quat.shape[0], 6)


def to_action_20d(actions16: np.ndarray) -> np.ndarray:
    """[T,16] → [T,20]（X-VLA EE6DActionSpace 布局）。"""
    actions16 = np.asarray(actions16, dtype=np.float64)
    out = np.zeros((actions16.shape[0], 20), dtype=np.float64)
    for arm, sl in CSV_ARM_SLICES.items():
        block = actions16[:, sl]
        xyz = block[:, :3]
        rot6d = _quat_wxyz_to_rot6d(block[:, 3:7])
        gripper = block[:, 7:8]
        idx = ARM_XYZ_IDX[arm]
        out[:, list(idx)] = xyz
        out[:, 9 if arm == "left" else 19] = gripper[:, 0]
        rot_start = 3 if arm == "left" else 13
        out[:, rot_start : rot_start + 6] = rot6d
    return out


def load_dataset(csv_path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """返回 (task_by_ep, frame_by_ep, state16_by_ep, action20_by_ep)。"""
    print(f"loading {csv_path} ...", flush=True)
    df = pd.read_csv(
        csv_path,
        usecols=[
            "episode_index",
            "task_index",
            "frame_index",
            "observation.state",
            "action",
        ],
    ).sort_values(["episode_index", "frame_index"], kind="stable")
    task_by_ep: dict[int, np.ndarray] = {}
    frame_by_ep: dict[int, np.ndarray] = {}
    state_by_ep: dict[int, np.ndarray] = {}
    action_by_ep: dict[int, np.ndarray] = {}
    for ep, sub in df.groupby("episode_index", sort=True):
        ep = int(ep)
        frames = sub["frame_index"].to_numpy()
        state = np.vstack([np.asarray(v, dtype=np.float64) for v in ke.parse_state_series(sub["observation.state"])])
        action = np.vstack([np.asarray(v, dtype=np.float64) for v in ke.parse_state_series(sub["action"])])
        task_by_ep[ep] = np.full(len(frames), int(sub["task_index"].iloc[0]))
        frame_by_ep[ep] = frames
        state_by_ep[ep] = state
        action_by_ep[ep] = to_action_20d(action)
    print(f"  {len(action_by_ep)} episodes loaded", flush=True)
    return task_by_ep, frame_by_ep, state_by_ep, action_by_ep


def load_split(split_path: Path) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """返回 (train_by_task, val_by_task)。"""
    data = json.loads(split_path.read_text(encoding="utf-8"))
    train: dict[int, list[int]] = {}
    val: dict[int, list[int]] = {}
    for key, info in data["tasks"].items():
        ti = int(info.get("task_index", key))
        train[ti] = [int(e) for e in info["train_episode_idx"]]
        val[ti] = [int(e) for e in info["val_episode_idx"]]
    return train, val


def build_config(args) -> PaceConfig:
    return PaceConfig(
        enabled=True,
        h_max=int(args.chunk_len),
        h_min=int(args.h_min),
        d_min=int(args.d_min),
        smooth=str(args.smooth),
        smooth_window=int(args.smooth_window),
        threshold_delta=None,
        threshold_delta_by_task={},
    )


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

def collect_prominences(
    cfg: PaceConfig, episodes: list[int], action_by_ep: dict[int, np.ndarray],
    chunk_len: int, stride: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """在示范轨迹上滑窗收集所有谷值的 prominence（论文 §3.3 的标定池）。"""
    pool: list[np.ndarray] = []
    stats = {"windows": 0, "valleys": 0, "episodes": 0}
    for ep in episodes:
        actions = action_by_ep[ep]
        if actions.shape[0] < chunk_len + 1:
            continue
        stats["episodes"] += 1
        for arm in ARMS:
            xyz_idx = ARM_XYZ_IDX[arm]
            for start in range(0, actions.shape[0] - chunk_len + 1, stride):
                chunk = actions[start : start + chunk_len]
                profile = arm_speed_profile(chunk, xyz_idx)
                smoothed = smooth_profile(profile, cfg.smooth_window, cfg.smooth)
                _, prominences = valley_prominences(smoothed, lo=0, hi=cfg.h_max - 1)
                if prominences:
                    pool.append(np.asarray(prominences, dtype=np.float64))
                stats["windows"] += 1
                stats["valleys"] += len(prominences)
    if not pool:
        return np.zeros(0, dtype=np.float64), stats
    return np.concatenate(pool), stats


def cmd_calibrate(args) -> None:
    cfg = build_config(args)
    train_by_task, _ = load_split(Path(args.split))
    if args.tasks != "all":
        keep = {int(x) for x in args.tasks.split(",")}
        train_by_task = {k: v for k, v in train_by_task.items() if k in keep}
    _, _, _, action_by_ep = load_dataset(Path(args.csv))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "chunk_len": cfg.h_max,
        "stride": args.stride,
        "d_min": cfg.d_min,
        "smooth": cfg.smooth,
        "smooth_window": cfg.smooth_window,
        "percentile": args.percentile,
        "split": str(args.split),
        "csv": str(args.csv),
        "tasks": {},
    }
    global_pool: list[np.ndarray] = []
    for ti in sorted(train_by_task):
        pool, stats = collect_prominences(
            cfg, train_by_task[ti], action_by_ep, cfg.h_max, args.stride
        )
        global_pool.append(pool)
        name = TASK_INDEX_TO_NAME.get(ti, f"task_{ti}")
        quantiles = {}
        if pool.size:
            quantiles = {
                f"p{q}": float(np.percentile(pool, q)) for q in QUANTILE_GRID
            }
        report["tasks"][name] = {
            "task_index": ti,
            "n_prominences": int(pool.size),
            "threshold_delta": (
                float(np.percentile(pool, args.percentile)) if pool.size else None
            ),
            "prominence_quantiles": quantiles,
            "windows": stats["windows"],
            "episodes": stats["episodes"],
        }
        print(
            f"[{name}] valleys={pool.size} windows={stats['windows']} "
            f"delta_t(p{args.percentile})="
            f"{report['tasks'][name]['threshold_delta']}",
            flush=True,
        )
        if pool.size and report["tasks"][name]["threshold_delta"] < PACE_NOISE_FLOOR:
            noise_floor = 1.0 / (1 << 24)
            print(
                f"  [warn] {name}: delta_t={report['tasks'][name]['threshold_delta']:.3e} "
                f"低于噪声层 {PACE_NOISE_FLOOR:.0e} —— 该池被示范静止段的 float32 "
                f"量化抖动（位移为 {noise_floor:.1e} 的整数倍）占据，按其标定会让几乎"
                f"所有谷值被接受、h 塌到 h_min。请改用更大的 --percentile"
                f"（本数据约 70~85），或先对示范做静止段过滤。",
                flush=True,
            )

    merged = np.concatenate([p for p in global_pool if p.size]) if any(
        p.size for p in global_pool
    ) else np.zeros(0)
    report["global_fallback"] = {
        "n_prominences": int(merged.size),
        "threshold_delta": (
            float(np.percentile(merged, args.percentile)) if merged.size else None
        ),
    }
    print(
        f"[global fallback] valleys={merged.size} "
        f"delta_t(p{args.percentile})={report['global_fallback']['threshold_delta']}",
        flush=True,
    )

    (out_dir / "pace_calibration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    table = {
        name: info["threshold_delta"]
        for name, info in report["tasks"].items()
        if info["threshold_delta"] is not None
    }
    lines = [
        "# 由 tools/pace_calibrate.py calibrate 生成，粘进 deploy.yml 的 pace 段。",
        f"# chunk_len={cfg.h_max} stride={args.stride} d_min={cfg.d_min} "
        f"smooth={cfg.smooth}(w={cfg.smooth_window}) p{args.percentile}",
        "threshold_delta: "
        + (
            f"{report['global_fallback']['threshold_delta']:.6g}"
            if report["global_fallback"]["threshold_delta"] is not None
            else "null"
        ),
        "threshold_delta_by_task:",
    ]
    for name in sorted(table):
        lines.append(f"  {name}: {table[name]:.6g}")
    if len(TASK_INDEX_TO_NAME) < 4:
        lines.append(
            "  # stack_blocks 无示范数据 -> 走上面的 threshold_delta 全局兜底"
        )
    snippet = "\n".join(lines) + "\n"
    (out_dir / "pace_thresholds.yml").write_text(snippet, encoding="utf-8")
    print("\n--- deploy.yml 片段 ---\n" + snippet)
    print(f"产物: {out_dir / 'pace_calibration.json'}")
    print(f"      {out_dir / 'pace_thresholds.yml'}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def episode_events(
    ti: int, state: np.ndarray, frames: np.ndarray, config: dict, detect_kwargs: dict
) -> list[dict]:
    """per-arm 关键事件帧 t0（grasp/place/insert 等）。"""
    info = ke.episode_event_instances(ti, state, frames, detect_kwargs, config)
    return info["instances"]


def simulate_schedule(
    actions: np.ndarray, cfg: PaceConfig, delta_by_arm: dict[str, float], chunk_len: int
) -> dict:
    """按 PACE 规则在示范轨迹上模拟重规划：返回每臂边界、并集边界与视野序列。

    并集口径与论文 §3.3 一致：多臂候选取并集、接受后取**最早**的边界作为本次 h。
    """
    per_arm_boundaries: dict[str, list[int]] = {arm: [] for arm in ARMS}
    schedule_boundaries: list[int] = []
    horizons: list[int] = []
    per_arm_horizons: dict[str, list[int]] = {arm: [] for arm in ARMS}
    tau = 0
    total = actions.shape[0]
    while tau + chunk_len <= total:
        chunk = actions[tau : tau + chunk_len]
        arm_h = {}
        for arm in ARMS:
            decision = select_execution_horizon(
                chunk, cfg, delta_by_arm[arm], task_name=arm
            )
            arm_h[arm] = decision.h
            per_arm_horizons[arm].append(decision.h)
            per_arm_boundaries[arm].append(tau + decision.h)
        h = min(arm_h.values())
        horizons.append(h)
        schedule_boundaries.append(tau + h)
        tau += h
    return {
        "per_arm_boundaries": per_arm_boundaries,
        "per_arm_horizons": per_arm_horizons,
        "schedule_boundaries": schedule_boundaries,
        "horizons": horizons,
        "covered_until": tau,
    }


def hit_rate(
    events: list[int], boundaries: list[int], tolerance: int
) -> tuple[float, int, int]:
    """事件被边界命中的比例（任一边界距离 <= tolerance）。"""
    if not events:
        return float("nan"), 0, 0
    array = np.asarray(boundaries, dtype=np.int64)
    hits = 0
    for t0 in events:
        if array.size and np.min(np.abs(array - t0)) <= tolerance:
            hits += 1
    return hits / len(events), hits, len(events)


def fixed_boundaries(total: int, h: int, chunk_len: int) -> list[int]:
    """固定 horizon 的重规划边界（与 PACE 相同的 `while tau + L <= T` 口径）。"""
    out = []
    tau = 0
    while tau + chunk_len <= total:
        out.append(tau + h)
        tau += h
    return out


def cmd_validate(args) -> None:
    cfg = build_config(args)
    train_by_task, val_by_task = load_split(Path(args.split))
    if args.tasks != "all":
        keep = {int(x) for x in args.tasks.split(",")}
        train_by_task = {k: v for k, v in train_by_task.items() if k in keep}
        val_by_task = {k: v for k, v in val_by_task.items() if k in keep}
    task_by_ep, frame_by_ep, state_by_ep, action_by_ep = load_dataset(Path(args.csv))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # δ_T：优先用已标定的 JSON，否则就地按训练 split 标定（只用训练示范，不碰验证集）。
    calibration_path = Path(args.calibration) if args.calibration else (
        out_dir / "pace_calibration.json"
    )
    if args.delta_t is not None:
        delta_by_task = {ti: float(args.delta_t) for ti in train_by_task}
        delta_global = float(args.delta_t)
        calib_source = "cli"
    elif calibration_path.exists():
        calib = json.loads(calibration_path.read_text(encoding="utf-8"))
        delta_global = calib["global_fallback"]["threshold_delta"]
        delta_by_task = {}
        for ti in train_by_task:
            name = TASK_INDEX_TO_NAME.get(ti, f"task_{ti}")
            value = calib["tasks"].get(name, {}).get("threshold_delta")
            delta_by_task[ti] = delta_global if value is None else float(value)
        calib_source = str(calibration_path)
    else:
        print("no calibration json found; calibrating on the train split first ...")
        delta_by_task = {}
        pools = []
        for ti in sorted(train_by_task):
            pool, _ = collect_prominences(
                cfg, train_by_task[ti], action_by_ep, cfg.h_max, args.stride
            )
            pools.append(pool)
            delta_by_task[ti] = (
                float(np.percentile(pool, args.percentile)) if pool.size else 0.0
            )
        merged = np.concatenate([p for p in pools if p.size])
        delta_global = float(np.percentile(merged, args.percentile))
        calib_source = "inline(train)"

    keyframe_config = ke.load_config(Path(args.config) if args.config else None)
    detect_kwargs = {
        "min_prominence": args.min_prominence,
        "hold_min_len": args.hold_min_len,
        "open_level": args.open_level,
        "allow_incomplete": not args.no_incomplete,
    }

    report: dict = {
        "chunk_len": cfg.h_max,
        "h_min": cfg.h_min,
        "d_min": cfg.d_min,
        "smooth": cfg.smooth,
        "smooth_window": cfg.smooth_window,
        "tolerance": args.tolerance,
        "metrics_note": (
            "命中率随重规划边界数单调上升：fixed H=1 的边界数等于帧数，必然 100% "
            "命中，故 best_fixed_h/best_fixed_hit_rate 不可用于比较。唯一有判别力的"
            "口径是 budget_matched（PACE 与「固定 H = PACE 平均视野」用同样多的重规划"
            "次数）。注意本验证用示范动作充当预测 chunk（论文同法标定），因此只测"
            "相位边界的定位质量，不含「执行跨相位 chunk 导致后段动作失效」这一 PACE "
            "真正要解决的收益 —— 后者必须在仿真里做 A/B。"
        ),
        "calibration_source": calib_source,
        "global_delta_t": delta_global,
        "per_task_delta_t": {
            TASK_INDEX_TO_NAME.get(ti, f"task_{ti}"): delta_by_task[ti]
            for ti in sorted(delta_by_task)
        },
        "tasks": {},
    }

    horizon_all: list[int] = []
    matched_all: list[tuple[int, int, int, int]] = []
    for ti in sorted(val_by_task):
        name = TASK_INDEX_TO_NAME.get(ti, f"task_{ti}")
        delta = delta_by_task[ti]
        per_arm = {arm: {"events": 0, "hits": 0, "vals": []} for arm in ARMS}
        union_events = 0
        union_hits = 0
        horizons: list[int] = []
        arm_horizons = {arm: [] for arm in ARMS}
        baseline: dict[int, tuple[int, int]] = {h: (0, 0) for h in range(1, cfg.h_max + 1)}
        n_episodes = 0
        matched_pairs: list[tuple[int, int, int, int]] = []
        figs: list[tuple[int, Path]] = []

        for ep in val_by_task[ti]:
            actions = action_by_ep[ep]
            state = state_by_ep[ep]
            frames = frame_by_ep[ep]
            if actions.shape[0] < cfg.h_max + 1:
                continue
            n_episodes += 1
            instances = episode_events(ti, state, frames, keyframe_config, detect_kwargs)
            total = actions.shape[0]
            cutoff = total - cfg.h_max
            events_by_arm: dict[str, list[int]] = {arm: [] for arm in ARMS}
            for inst in instances:
                side = inst.get("side")
                t0 = int(inst["t0"])
                if side in events_by_arm and t0 <= cutoff:
                    events_by_arm[side].append(t0)
            sim = simulate_schedule(
                actions, cfg, {arm: delta for arm in ARMS}, cfg.h_max
            )
            horizons.extend(sim["horizons"])
            for arm in ARMS:
                arm_horizons[arm].extend(sim["per_arm_horizons"][arm])
                rate, hits, total_events = hit_rate(
                    events_by_arm[arm], sim["per_arm_boundaries"][arm], args.tolerance
                )
                per_arm[arm]["events"] += total_events
                per_arm[arm]["hits"] += hits
                if not np.isnan(rate):
                    per_arm[arm]["vals"].append(rate)

            all_events = sorted(events_by_arm["left"] + events_by_arm["right"])
            _, u_hits, u_total = hit_rate(
                all_events, sim["schedule_boundaries"], args.tolerance
            )
            union_events += u_total
            union_hits += u_hits

            for h in range(1, cfg.h_max + 1):
                _, b_hits, b_total = hit_rate(
                    all_events, fixed_boundaries(total, h, cfg.h_max), args.tolerance
                )
                prev_hits, prev_total = baseline[h]
                baseline[h] = (prev_hits + b_hits, prev_total + b_total)

            mean_h = float(np.mean(sim["horizons"])) if sim["horizons"] else float("nan")
            if np.isfinite(mean_h):
                matched = int(round(mean_h))
                _, m_hits, m_total = hit_rate(
                    all_events,
                    fixed_boundaries(total, matched, cfg.h_max),
                    args.tolerance,
                )
                _, p_hits, p_total = hit_rate(
                    all_events, sim["schedule_boundaries"], args.tolerance
                )
                # 聚合用命中数/事件数，而不是逐 episode 命中率的平均（后者会被
                # 事件数少的 episode 等权放大）。
                matched_pairs.append((matched, m_hits, p_hits, max(m_total, p_total)))

            if len(figs) < args.per_task:
                figs.append((ep, _plot_episode(ti, name, ep, actions, frames, events_by_arm,
                                               sim, cfg, delta, out_dir)))

        horizon_all.extend(horizons)
        matched_all.extend(matched_pairs)
        baseline_rates = {
            h: (hits / total if total else float("nan"))
            for h, (hits, total) in baseline.items()
        }
        best_h = max(baseline_rates, key=lambda h: (baseline_rates[h], -h)) if baseline_rates else None
        matched_h = int(round(float(np.mean(horizons)))) if horizons else None
        report["tasks"][name] = {
            "task_index": ti,
            "delta_t": delta,
            "n_val_episodes": n_episodes,
            "pace": {
                "mean_h": float(np.mean(horizons)) if horizons else None,
                "median_h": float(np.median(horizons)) if horizons else None,
                "min_h": int(min(horizons)) if horizons else None,
                "max_h": int(max(horizons)) if horizons else None,
                "n_queries": len(horizons),
                "event_hit_rate": union_hits / union_events if union_events else None,
                "n_events": union_events,
                "per_arm_hit_rate": {
                    arm: (
                        per_arm[arm]["hits"] / per_arm[arm]["events"]
                        if per_arm[arm]["events"]
                        else None
                    )
                    for arm in ARMS
                },
                "per_arm_mean_h": {
                    arm: (
                        float(np.mean(arm_horizons[arm])) if arm_horizons[arm] else None
                    )
                    for arm in ARMS
                },
            },
            "fixed_horizon_hit_rate": {
                str(h): baseline_rates[h] for h in sorted(baseline_rates)
            },
            "best_fixed_h": best_h,
            "best_fixed_hit_rate": baseline_rates.get(best_h) if best_h else None,
            "matched_h": matched_h,
            "matched_hit_rate": baseline_rates.get(matched_h) if matched_h else None,
            # 等预算对照（唯一有判别力的口径）：PACE 与「固定 H = PACE 平均视野」
            # 用同样多的重规划次数。命中率本身随边界数单调上升（H=1 时边界数=帧数，
            # 必然 100% 命中），故 best_fixed_h 恒为 1，不可用于比较。
            "budget_matched": {
                "n_events": sum(p[3] for p in matched_pairs),
                "pace_hit_rate": (
                    sum(p[2] for p in matched_pairs) / max(sum(p[3] for p in matched_pairs), 1)
                ),
                "fixed_hit_rate": (
                    sum(p[1] for p in matched_pairs) / max(sum(p[3] for p in matched_pairs), 1)
                ),
            },
            "figures": [str(p) for _, p in figs],
        }
        entry = report["tasks"][name]
        bm = entry["budget_matched"]
        print(
            f"[{name}] delta_t={delta:.4g} val_eps={n_episodes} events={union_events} "
            f"PACE hit={entry['pace']['event_hit_rate']:.3f} "
            f"mean_h={entry['pace']['mean_h']:.1f} "
            f"| 等预算 fixed H={matched_h}: {bm['fixed_hit_rate']:.3f} "
            f"vs PACE {bm['pace_hit_rate']:.3f}",
            flush=True,
        )

    n_matched = max(sum(p[3] for p in matched_all), 1)
    report["overall"] = {
        "mean_h": float(np.mean(horizon_all)) if horizon_all else None,
        "n_queries": len(horizon_all),
        "budget_matched": {
            "n_events": sum(p[3] for p in matched_all),
            "pace_hit_rate": sum(p[2] for p in matched_all) / n_matched,
            "fixed_hit_rate": sum(p[1] for p in matched_all) / n_matched,
        },
        "h_histogram": (
            {str(h): int((np.asarray(horizon_all) == h).sum()) for h in sorted(set(horizon_all))}
            if horizon_all
            else {}
        ),
    }
    (out_dir / "pace_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ov = report["overall"]
    print(f"\noverall mean_h={ov['mean_h']:.2f} n_queries={len(horizon_all)}")
    print(
        f"等预算对照（{ov['budget_matched']['n_events']} 事件）："
        f"fixed hit={ov['budget_matched']['fixed_hit_rate']:.3f} "
        f"vs PACE hit={ov['budget_matched']['pace_hit_rate']:.3f}"
    )
    print(f"产物: {out_dir / 'pace_validation.json'}")


def _plot_episode(
    ti: int,
    name: str,
    ep: int,
    actions: np.ndarray,
    frames: np.ndarray,
    events_by_arm: dict[str, list[int]],
    sim: dict,
    cfg: PaceConfig,
    delta: float,
    out_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    colors = {"left": "#1f77b4", "right": "#d62728"}
    for ax, arm in zip(axes, ARMS):
        profile = arm_speed_profile(actions, ARM_XYZ_IDX[arm])
        smoothed = smooth_profile(profile, cfg.smooth_window, cfg.smooth)
        ax.plot(np.arange(profile.size) + 1, smoothed, color=colors[arm], lw=1.2,
                label=f"{arm} smoothed speed (m/step)")
        idxs, proms = valley_prominences(smoothed, lo=0, hi=cfg.h_max - 1)
        accepted = [(j, p) for j, p in zip(idxs, proms) if p >= delta]
        for j, p in zip(idxs, proms):
            ax.plot(j + 1, smoothed[j], marker="v", ms=5, color="#7f7f7f", zorder=3)
            ax.annotate(f"{p:.3f}", (j + 1, smoothed[j]), textcoords="offset points",
                        xytext=(0, -12), fontsize=6, color="#555555", ha="center")
        for j, p in accepted:
            ax.plot(j + 1, smoothed[j], marker="v", ms=9, color="#2ca02c", zorder=4)
        for t0 in events_by_arm[arm]:
            ax.axvline(t0, color="#9467bd", ls="--", lw=1.0, alpha=0.8)
        for b in sim["per_arm_boundaries"][arm]:
            ax.axvline(b, color=colors[arm], ls=":", lw=1.1, alpha=0.9)
        ax.set_ylabel(f"{arm} speed")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_title(
        f"task={name}(#{ti}) episode={ep}  PACE delta_t={delta:.4g}  "
        f"d_min={cfg.d_min} smooth={cfg.smooth}(w={cfg.smooth_window})  "
        f"mean_h={np.mean(sim['horizons']):.1f}\n"
        "purple dashed = labeled event t0 (grasp/place/insert) | dotted = PACE replan boundary "
        "| green = accepted valley (prominence >= delta_t) | grey numbers = all valley prominences"
    )
    axes[-1].set_xlabel("frame_index")
    fig.tight_layout()
    path = out_dir / f"pace_validate_{name}_ep{ep:03d}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def common_parser_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="sim CSV 路径")
    p.add_argument("--split", default=str(DEFAULT_SPLIT), help="train/val 划分 JSON")
    p.add_argument("--tasks", default="all", help="task_index 逗号分隔或 'all'")
    p.add_argument("--chunk-len", type=int, default=30,
                   help="示范 chunk 长度 L（= X-VLA num_actions）")
    p.add_argument("--h-min", type=int, default=5, help="与 deploy.yml pace.h_min 一致")
    p.add_argument("--d-min", type=int, default=10, help="候选边界最小间隔")
    p.add_argument("--smooth", default="moving_average", choices=["moving_average", "none"])
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--percentile", type=float, default=5.0, help="标定百分位 rho")
    p.add_argument("--stride", type=int, default=1, help="标定滑窗步长")
    p.add_argument("--out", default=str(DEFAULT_OUT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="用训练示范标定每任务 δ_T")
    common_parser_args(p_cal)
    p_cal.set_defaults(func=cmd_calibrate)

    p_val = sub.add_parser("validate", help="验证集上的相位边界命中率验证")
    common_parser_args(p_val)
    p_val.add_argument("--calibration", default=None, help="calibrate 产出的 JSON")
    p_val.add_argument("--delta-t", type=float, default=None,
                       help="直接指定全局 δ_T（覆盖标定结果）")
    p_val.add_argument("--tolerance", type=int, default=5,
                       help="事件 t0 与边界的命中容差（帧）")
    p_val.add_argument("--per-task", type=int, default=2, help="每任务出图 episode 数")
    p_val.add_argument("--config", default=None, help="关键帧配置 JSON")
    p_val.add_argument("--min-prominence", type=float, default=0.2)
    p_val.add_argument("--hold-min-len", type=int, default=3)
    p_val.add_argument("--open-level", type=float, default=0.9)
    p_val.add_argument("--no-incomplete", action="store_true")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
