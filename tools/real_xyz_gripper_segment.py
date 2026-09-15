#!/usr/bin/env python3
"""真实数据集 (data/real_lerobot_v30_ee) 的单臂事件打标驱动 —— 复用 tools/xyz_gripper_segment.py。

**不修改任何现有文件**。本工具只做四件仿真版做不到的事，其余（XYZ 反查最终操作邻域、
五状态机、绘图、事件 JSON / 逐帧标签 CSV / summary / 边界帧拼图）全部原样复用
`tools/xyz_gripper_segment.py` 的 `main()`。

真实数据相对仿真多出的四处（前三处对应 configs/real_xyz_segment_config.json 的 _why_separate）：

  ① **逐臂归一化**。真实两臂闭合值不同且不为 0（task0 右臂停在 ≈0.67、task2 两臂 ≈0.35、
     task4 两臂 ≈0.55），绝对阈值下 task4 检出 0 个周期。按 doc §3.2『阈值应使用归一化开度』，
     对每个 (episode, 臂) 用自身 [p2, p98] 线性归一化后再送检测。
     —— 实现方式：替换 `keyframe_events.detect_sides`，把归一化后的 state 交给原函数。
  ② **逐任务检测参数**。真实各任务的夹爪行程与抖动幅度差异极大，统一参数必然顾此失彼
     （task5 要 min_prominence=0.01，task0/task2 要 0.15）。
     —— 实现方式：替换后的 detect_sides 按当前 task_index 覆盖 detect_kwargs。
  ③ **真实数据集路径与任务名**。仿真版把 DEFAULT_CSV / DEFAULT_TASKS / 视频目录写死在
     tools/keyframe_events.py 里。
     —— 实现方式：替换 `keyframe_events.load_tasks`；csv 与视频目录由命令行显式传入。
  ④ **锚点修正**（见 `_refine_anchors`）。`keyframe_detect.py` 的三处缺陷在真实数据上
     表现明显：保持平台取最长平台、close_start 回溯绑死在 open_level 上、保持/释放锚点
     落到别的阶段。真实数据上这三条会让事件区间把「手臂已停稳的保持段」和随后的
     「回 home」整段吞进来（最长的一个区间达 1170 帧、76% 是静止帧）。
     —— 实现方式：在替换后的 detect_sides 里就地修正锚点。

用法:
    python tools/real_xyz_gripper_segment.py                     # 6 个任务全跑
    python tools/real_xyz_gripper_segment.py --tasks 3           # 只跑叠碗
    python tools/real_xyz_gripper_segment.py --tasks 0,5 --no-video
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

from tools import keyframe_detect as kd          # noqa: E402
from tools import keyframe_events as ke          # noqa: E402
from tools import xyz_gripper_segment as xgs     # noqa: E402
from tools.gripper_keyframe_labels import normalize_per_arm  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "real_xyz_segment_config.json"
DEFAULT_OUT = ROOT / "outputs" / "real_xyz_segment"


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    ds = cfg["dataset"]
    for key in ("csv", "tasks_parquet", "video_dir"):
        ds["_" + key] = (ROOT / ds[key]) if not Path(ds[key]).is_absolute() else Path(ds[key])
    return cfg


def resolve_csv(raw: Path) -> Path:
    """兼顾 data/x.csv 与 data/x/x.csv 两种布局（配置里写的是前者）。"""
    if raw.is_file():
        return raw
    alt = raw.parent / raw.stem / raw.name
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"csv 不存在: {raw} (回退候选 {alt} 也不存在)")


# ---------------------------------------------------------------------------
# 运行时替换（只改本进程内的模块属性，磁盘上的现有文件一律不动）
# ---------------------------------------------------------------------------

_ORIG_DETECT_SIDES = ke.detect_sides
_ORIG_LOAD_TASKS = ke.load_tasks
_ORIG_LOAD_STATE_COLS = ke.load_state_cols

_STATE = {"ti": 0, "detect_by_task": {}, "lo": 2.0, "hi": 98.0, "df": None,
          "eps_flat": 0.002, "k_flat": 12, "tol_hold": 0.01,
          # 手臂侧：判「停稳」用（见 _refine_anchors）。n_settle 是「长静止」的最短帧数。
          "v_static": 0.002, "smooth": 5, "k_static": 5, "n_settle": 50}


def _side_params(kw: dict, side: str) -> dict:
    """检测参数支持逐臂覆盖：值若形如 {"left":..,"right":..} 则按 side 取。

    为什么必须逐臂：task5 左臂的保持电平是 0.44，右臂周期间的回开电平只有 0.52~0.60。
    单一 open_level 无法同时成立 —— 取 0.4 时左臂首个保持平段被判成「回开到打开」，
    整个周期被吞掉（ep502 左臂只检出 1 个周期）；取 0.6 时右臂的 4 个周期会粘连。
    """
    out = {}
    for k, v in kw.items():
        if isinstance(v, dict) and ("left" in v or "right" in v):
            if side in v:
                out[k] = v[side]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# ④ 锚点修正（keyframe_detect.py 的三处已知缺陷，只在真实数据侧补，不动原文件）
# ---------------------------------------------------------------------------

def _flat_runs(g: np.ndarray, eps_flat: float, k_flat: int) -> list[tuple[int, int, float]]:
    """|Δg| < eps_flat 的极大连续段（长度 >= k_flat），返回 (起点, 终点, 段内均值)。"""
    d = np.abs(np.diff(g, prepend=g[0]))
    flat = d < eps_flat
    runs, i, n = [], 0, len(g)
    while i < n:
        if flat[i]:
            j = i
            while j + 1 < n and flat[j + 1]:
                j += 1
            if j - i + 1 >= k_flat:
                runs.append((i, j, float(g[i:j + 1].mean())))
            i = j + 1
        else:
            i += 1
    return runs


def _hold_around_min(g: np.ndarray, lo: int, end: int, k_flat: int,
                     tol_hold: float) -> tuple[int, int, float] | None:
    """保持平台 = 以区间最低点为中心、电平落在 [min, min+tol] 内的极大连续段。

    为什么不用「严格平台」（|Δg| < eps 的连续段）：真实遥操的最后一段闭合经常是
    **缓慢蠕动**而非阶跃 —— task1 ep110 右臂从 0.914 用 65 帧慢慢爬到 0.620，
    再 5 帧掉到 0.138，底部那 0.138 只维持 15 帧。逐帧 |Δg| 全程超过 eps_flat，
    切不出任何平台，保持平台就落到了 0.914 的「放开后等待」平台上（见下）。
    以最低点 + 容差带取段则不受影响：抓取必然是本周期闭合得最深的位置。

    tol 取 3 档（tol_hold, 2×, 4×）逐级放宽，直到段长 >= k_flat，避免最低点附近
    只剩两三帧噪声时取出一个退化区间。三档都不够长则返回 None，交回原引擎值。
    """
    seg = g[lo:end + 1]
    if seg.size == 0:
        return None
    m = float(seg.min())
    i0 = int(np.argmin(seg))
    for t in (tol_hold, 2 * tol_hold, 4 * tol_hold):
        ok = seg <= m + t
        a = i0
        while a > 0 and ok[a - 1]:
            a -= 1
        b = i0
        while b + 1 < ok.size and ok[b + 1]:
            b += 1
        if b - a + 1 >= k_flat:
            return lo + a, lo + b, m
    return None


def _arm_rests(st: np.ndarray, side: str) -> list[tuple[int, int]]:
    """该臂「静止段」= 平滑末端速率 <= v_static 且连续 >= k_static 帧的极大段。

    与 tools/xyz_gripper_segment.py 里判 hold/idle 用的是同一套口径（同 smooth、
    同 v_static、同 k_static），保证锚点修正与状态层看到的是同一个「手臂在不在动」。
    """
    xyz = st[:, 0:3] if side == "left" else st[:, 8:11]
    sp = xgs.smoothed_speed(xyz, _STATE["smooth"])
    still = sp <= _STATE["v_static"]
    out, i, n = [], 0, len(still)
    while i < n:
        if still[i]:
            j = i
            while j + 1 < n and still[j + 1]:
                j += 1
            if j - i + 1 >= _STATE["k_static"]:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _first_long_rest(rests: list[tuple[int, int]], t_from: int,
                     t_to: int) -> int | None:
    """[t_from, t_to] 内第一次「停稳」的起点；没有则 None。

    上界 t_to 必不可少：不加就会被下一周期的停稳段吸走 —— 实测 task3 ep300 左臂
    cycle0 的 close_start=52，若一路搜到 episode 末尾的 (495,797)，会把 cycle1 的
    闭合段 (257,264) 当成 cycle0 的保持平台。

    两条判停稳的途径（任一成立即可）：
      (a) 段长 >= n_settle —— 常规情形，手臂真正停下来了；
      (b) 该段一直延伸到本周期末尾 —— 手臂停到周期结束，即使段长不足 n_settle。
    为什么需要 (b)：task4 ep404 左臂 cycle1 的整个周期只有 15/18 帧的碎片式停顿
    （706-720、725-742），拿不到任何 >=50 帧的段，会把回 home 途中那次全张（721-725）
    当成释放终点，区间仍旧多出 44 帧。而 (b) 取到 (725,742) → 释放终点回到 685。
    """
    for a, b in rests:
        if t_from <= a <= t_to and ((b - a + 1) >= _STATE["n_settle"] or b >= t_to):
            return a
    return None


def _steps(g: np.ndarray, sign: int, step: float) -> list[tuple[int, int]]:
    """爪夹单调变化段：sign=-1 取闭合（下降），+1 取开放（上升）。"""
    d = np.diff(g, prepend=g[0])
    flag = (d < -step) if sign < 0 else (d > step)
    out, i, n = [], 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and flag[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _last_step_end(steps: list[tuple[int, int]], t_from: int,
                   t_end: int | None) -> int | None:
    """[t_from, t_end) 内最后一次跳变段的末帧；没有则 None。

    下界 t_from 不能省：否则会取到**上一周期**的跳变（实测 task1 ep105 右臂 cycle2
    因此拿到 open_end=1214 < close_start=1352，锚点直接失序）。
    """
    if t_end is None:
        return None
    prev = [t for t in steps if t_from <= t[1] < t_end]
    return int(prev[-1][1]) if prev else None


def _refine_anchors(st: np.ndarray, per_side: dict, kw_by_side: dict) -> dict:
    """就地修正 close_start / 保持平台 / open_full（doc §4.2 语义）。

    原检测器 tools/keyframe_detect.py 有三处缺陷，在真实数据上表现明显：

    【缺陷 1 · 保持平台取成了最长平台】keyframe_detect.py:190
        `hold = max(closed_flats, key=lambda f: f[1] - f[0])`
      真实遥操「放开物体后手停在原地等待、最后才完全张开」，于是周期里会出现两个
      长度相近的平台：真保持平台（电平最低，抓住物体）与释放后的等待平台（电平较高）。
      取最长等于抛硬币。实测 task3 ep304 右臂 0.035 平台 81 帧 vs 0.355 平台 86 帧，
      引擎选了后者，导致 holder_start=500 / holder_end=594（用户指出应为 405 / 500）。
      → 修正：保持平台 = 以区间最低点为中心的容差平台（见 _hold_around_min），
        但**只在「手臂停稳」规则给不出结果时兜底**（见缺陷 3 的说明）。

    【缺陷 2 · close_start 回溯绑死在 open_level 上】keyframe_detect.py:193-198
        `while cs > 0 and g[cs - 1] < open_level: cs -= 1`
      该循环一旦 open_level 被压低（真实数据逐任务压到 0.35~0.7 才检得出周期），
      起点值早已高于 open_level，循环一次都不进，close_start 退化成「引擎切出的那段
      下降段的起点」——而下降段是从斜率超过 eps 处才开始的。实测 task3 ep301 右臂
      0.994 平台到 f315、f318 起开始闭合，引擎却给出 close_start=375（raw 0.544）。
      → 修正：close_start = 回溯区间内**电平最高**的持续平台的末帧 + 1，即「离开开口位
        置的那一帧」，与 open_level 解耦。平局取更晚的。

    【缺陷 3 · 保持/释放锚点落在别的阶段】—— 本次新增，是用户四条反馈的根因
      区间 = [approach_start, 锚点 + k_confirm]，锚点晚几百帧，区间就把「手臂已停稳的
      保持段」和随后的「回 home」一起吞进来。两种成因：
        (a) 全局最低点落在别的阶段。_hold_around_min 的搜索区间是
            [上一周期 open_full, 本周期 open_full]，一旦本周期 open_full 本身偏晚，
            最低点会落到**之后的放置阶段**。实测 task5 ep501 左臂：真抓取闭合在
            132~166（→0.461），全 episode 最低点 0.042 却在第 1300 帧（放置阶段），
            取到它 → close_end=1254，区间 [89,1258]。
        (b) 闭合后**缓慢蠕动**。fill ep0 右臂 0.689→0.680 爬了 800 帧，容差带从 721
            才开始 → close_end=721，而真值是 251。注意这一条是「修正反而做坏」：
            引擎原始检测给的就是 251。

      → 修正：锚点用**手臂运动**封边界，而不是用爪夹电平的全局极值。
        本周期第一次「长静止」（>= n_settle 帧，即手臂真正停稳）之前的：
          - 最后一次**闭合**跳变的末帧 = 保持平台起点 close_end；
          - 最后一次**开放**跳变的末帧 = 完全张开 open_end。
        长静止之后的一切同向跳变都是重抓/回 home 的副产物，不再是本事件的一部分。

    为什么 open_end 也必须修：引擎要求爪夹回到该任务的 open_level 才算「张开了」，
    但真实释放往往只张到**中间电平**、手臂随即离开，最后一次全张发生在回 home 途中。
    实测 fill ep0 右臂 place：真释放 1068~1074，引擎给 open_full=1203。

    回溯区间 = [本侧上一周期的 open_full, 本周期的保持平台起点)，首周期从 0 开始 ——
    这个下界防止跨周期取到上一周期的开口平台。
    """
    from scipy.ndimage import median_filter

    eps_flat, k_flat = _STATE["eps_flat"], _STATE["k_flat"]
    tol_hold = _STATE["tol_hold"]
    T = st.shape[0]
    for side, gi in (("left", ke.GRIP_L), ("right", ke.GRIP_R)):
        g = median_filter(st[:, gi].astype(float), size=3, mode="nearest")
        runs = _flat_runs(g, eps_flat, k_flat)
        step = float(kw_by_side.get(side, {}).get("eps", 0.02))
        desc_steps = _steps(g, -1, step)
        asc_steps = _steps(g, +1, step)
        rests = _arm_rests(st, side)
        kf = per_side[side]
        prev_end = 0
        for i in range(len(kf["close_start"])):
            cs, hs, he = (int(kf["close_start"][i]), int(kf["hold_start"][i]),
                          int(kf["hold_end"][i]))
            oe = int(kf["open_full"][i])
            end = T - 1 if oe == ke.INCOMPLETE else oe
            lo = prev_end

            # --- 缺陷 2：close_start = 回溯区间内电平最高的平台末帧 + 1 ---
            # 必须先用**原始** hold_start 定出「离开开口位置的那一帧」，不能拿引擎的
            # close_start 当回溯下界：引擎在 open_level 被压低时会把 close_start 给到
            # 周期末（实测 task5 ep501 左臂原始 close_start=1211，真值 129），而下面的
            # 停稳搜索又是从 close_start 起算的，用错值会让整条修正链一起跑偏。
            cand2 = [r for r in runs if r[0] >= lo and r[1] < hs]
            if cand2:
                t = max(cand2, key=lambda r: (round(r[2], 4), r[0]))
                cs = int(t[1]) + 1

            # --- 缺陷 3(a)：保持平台起点 = 本周期停稳前的最后一次闭合跳变末帧 ---
            cand = _last_step_end(desc_steps, cs, _first_long_rest(rests, cs, end))
            if cand is not None:
                hs = cand
            else:                                   # 兜底：缺陷 1 的最低点容差平台
                hold = _hold_around_min(g, lo, end, k_flat, tol_hold)
                if hold is not None:
                    hs = hold[0]

            # 释放起点 = 保持平台之后第一次开放跳变的起点（与 open_level 解耦）
            after = [t for t in asc_steps if hs <= t[0] <= end]
            if after:
                he = int(after[0][0])

            # --- 缺陷 3(b)：open_full = 释放之后、手臂停稳前的最后一次开放跳变末帧 ---
            if oe != ke.INCOMPLETE:
                cand_oe = _last_step_end(asc_steps, he, _first_long_rest(rests, he, end))
                if cand_oe is not None:
                    oe = cand_oe

            kf["close_start"][i], kf["hold_start"][i], kf["hold_end"][i] = cs, hs, he
            kf["open_full"][i] = oe
            prev_end = end
    return per_side


def _detect_sides_real(state: np.ndarray, frame: np.ndarray,
                       detect_kwargs: dict) -> dict:
    """逐臂归一化 + 逐任务参数覆盖后调用原 `detect_sides`，再做锚点修正。

    归一化只作用在送进检测的那份 state 副本上；调用方 `segment_episode` 持有的原 state
    不受影响，所以事件 JSON / 图 / CSV 里仍是原始开度（归一化是逐臂单调线性变换，
    锚点帧号不变）。
    """
    # 配置里的 "_note"/"_evidence" 等说明性键不是检测参数，必须剔除
    base = {k: v for k, v in detect_kwargs.items() if not k.startswith("_")}
    base.update({k: v for k, v in _STATE["detect_by_task"].get(str(_STATE["ti"]), {}).items()
                 if not k.startswith("_")})
    st = np.array(state, dtype=float, copy=True)
    for gi in (ke.GRIP_L, ke.GRIP_R):
        st[:, gi] = normalize_per_arm(state[:, gi], _STATE["lo"], _STATE["hi"])[0]

    # 逐臂参数：左右臂的夹爪行程差异可能大到单一阈值无法兼顾（task5 见 _side_params）
    per_side, kw_by_side = {}, {}
    for side, gi in (("left", ke.GRIP_L), ("right", ke.GRIP_R)):
        kw = {k: v for k, v in _side_params(base, side).items() if not k.startswith("_")}
        kw_by_side[side] = kw
        kf = kd.detect_gripper_keyframes(st[:, gi], frame, **kw)
        per_side[side] = {k: np.asarray(v, dtype=int) for k, v in kf.items()}
    return _refine_anchors(st, per_side, kw_by_side)


def _load_tasks_real(*_a, **_kw) -> dict:
    """仿真版无参调用 `load_tasks()`，这里强制走真实 tasks.parquet。"""
    return _ORIG_LOAD_TASKS(_STATE["tasks_parquet"])


def _load_state_cols_cached(path) -> pd.DataFrame:
    """6 个任务各跑一次 main()，主表只读一次。"""
    if _STATE["df"] is None:
        _STATE["df"] = _ORIG_LOAD_STATE_COLS(path)
    return _STATE["df"]


def install_real_hooks(cfg: dict, csv_path: Path) -> None:
    _STATE["detect_by_task"] = cfg.get("detect_by_task", {})
    _STATE["lo"] = float(cfg["normalize"]["lo_pct"])
    _STATE["hi"] = float(cfg["normalize"]["hi_pct"])
    _STATE["tasks_parquet"] = cfg["dataset"]["_tasks_parquet"]
    _STATE["df"] = None
    flat = cfg.get("defaults", {}).get("flat", {})
    _STATE["eps_flat"] = float(flat.get("eps_flat", 0.002))
    _STATE["k_flat"] = int(flat.get("k_flat", 12))
    _STATE["tol_hold"] = float(flat.get("tol_hold", 0.01))
    _STATE["n_settle"] = int(flat.get("hold_settle_frames", 50))
    sp = cfg.get("defaults", {}).get("speed", {})
    _STATE["v_static"] = float(sp.get("v_static", 0.002))
    _STATE["smooth"] = int(sp.get("smooth", 5))
    _STATE["k_static"] = int(sp.get("k_static", 5))
    ke.detect_sides = _detect_sides_real
    ke.load_tasks = _load_tasks_real
    ke.load_state_cols = _load_state_cols_cached
    # xgs 通过 `from tools import keyframe_events as ke` 引用同一模块对象，
    # 故上面的属性替换对 xgs.main() 内部同样生效。
    assert xgs.ke is ke, "xgs 与 ke 不是同一模块对象，挂钩失效"


# ---------------------------------------------------------------------------
# 逐任务调用 xgs.main()
# ---------------------------------------------------------------------------

def run_one_task(cfg: dict, csv_path: Path, ti: int, episodes: list[int],
                 out_root: Path, dataset_label_dir: Path,
                 *, no_video: bool, no_plot: bool) -> Path:
    """拼 argv 调 xgs.main()，产物落到 out_root/t<ti>_<slug>/。"""
    slug = cfg["tasks"][str(ti)]["slug"]
    out_dir = out_root / f"t{ti}_{slug}"
    _STATE["ti"] = ti

    argv = [
        "real_xyz_gripper_segment.py",
        "--episodes", ",".join(str(e) for e in episodes),
        "--task", str(ti),
        "--csv", str(csv_path),
        "--config", str(DEFAULT_CONFIG),
        "--video-dir", str(cfg["dataset"]["_video_dir"]),
        "--out", str(out_dir),
        "--dataset-label-dir", str(dataset_label_dir),
    ]
    if no_video:
        argv.append("--no-video")
    if no_plot:
        argv.append("--no-plot")

    print(f"\n{'=' * 78}\n=== task {ti}  {slug}  ({cfg['tasks'][str(ti)]['cn']})  "
          f"{len(episodes)} episodes\n{'=' * 78}")
    saved, sys.argv = sys.argv, argv
    try:
        xgs.main()
    finally:
        sys.argv = saved
    return out_dir


# ---------------------------------------------------------------------------
# 汇总 + 与人工真值对照
# ---------------------------------------------------------------------------

def collect(cfg: dict, out_root: Path, dataset_label_dir: Path,
            per_task: dict[int, Path]) -> None:
    merged: dict = {"config": str(DEFAULT_CONFIG), "detect_by_task": cfg["detect_by_task"],
                    "normalize": cfg["normalize"], "tasks": {}}
    rows: list[dict] = []

    for ti, out_dir in sorted(per_task.items()):
        slug = cfg["tasks"][str(ti)]["slug"]
        summ = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        merged["tasks"][str(ti)] = {"slug": slug, "cn": cfg["tasks"][str(ti)]["cn"],
                                    "params": summ["episodes"], "detect": summ["detect"]}
        gt_by_task = cfg["_gt_cycles"]["by_task"].get(str(ti))
        gt_by_ep = cfg["_gt_cycles"]["by_episode"]
        for ep_s, info in sorted(summ["episodes"].items(), key=lambda x: int(x[0])):
            ep = int(ep_s)
            n_grasp = len(info["grasp_lead_frames"])
            n_place = len([1 for _ in info["place_lead_frames"]])
            gt = gt_by_ep.get(ep_s, gt_by_task)
            rows.append({
                "task_index": ti, "slug": slug, "episode_index": ep,
                "frames": info["num_frames"],
                "n_grasp": n_grasp, "n_place": n_place,
                "n_cycles": n_grasp,
                "gt_cycles": gt,
                "match": "OK" if gt == n_grasp else "DIFF",
                "grasp_lead_p50": round(float(np.median(info["grasp_lead_frames"])), 1)
                                  if info["grasp_lead_frames"] else None,
                "place_lead_p50": round(float(np.median(info["place_lead_frames"])), 1)
                                  if info["place_lead_frames"] else None,
                "n_boundary_frames": info["n_boundary_stills"],
                "state_share_left": json.dumps(info["state_share_pct"]["left"],
                                               ensure_ascii=False),
                "state_share_right": json.dumps(info["state_share_pct"]["right"],
                                                ensure_ascii=False),
                "notes": " | ".join(info["notes"]),
            })

    (out_root / "all_summary.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(out_root / "detection_vs_gt.csv", index=False)

    # 合并逐任务写出的数据集标签表（主键 (episode, side, cycle) / (episode, frame)，无冲突）
    for stem, fname, sort_by in (
        ("gripper_anchors", "gripper_anchors.csv", ["episode_index", "side", "cycle"]),
        ("arm_states", "arm_states.csv", ["episode_index", "frame_index"]),
    ):
        parts = sorted(dataset_label_dir.glob(f"{stem}_task*.csv"))
        if not parts:
            continue
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        df = df.sort_values(sort_by).reset_index(drop=True)
        df.to_csv(dataset_label_dir / fname, index=False)
        print(f"合并 -> {dataset_label_dir / fname}  ({len(df)} 行, 来自 {len(parts)} 个任务)")

    print(f"\n{'=' * 78}\n逐集检出 vs 人工真值\n{'=' * 78}")
    print(f"{'task':>4} {'slug':26s} {'ep':>5} {'帧数':>6} {'检出':>5} {'真值':>5} {'':>5} "
          f"{'grasp lead':>11} {'place lead':>11}")
    for r in rows:
        flag = "  " if r["match"] == "OK" else " *"
        print(f"{r['task_index']:>4} {r['slug']:26s} {r['episode_index']:>5} "
              f"{r['frames']:>6} {r['n_cycles']:>5} {str(r['gt_cycles']):>5} {flag:>5} "
              f"{str(r['grasp_lead_p50']):>11} {str(r['place_lead_p50']):>11}")
    n_ok = sum(1 for r in rows if r["match"] == "OK")
    print(f"\n命中 {n_ok}/{len(rows)} 集一致" + ("  （* = 与真值不符）" if n_ok < len(rows) else ""))
    print(f"\n产物: {out_root}")
    print(f"  逐任务目录    t<ti>_<slug>/  segment_ep*.png + events_ep*.json + "
          f"labels_ep*.csv + boundary_sheet_ep*.png")
    print(f"  合并 summary  all_summary.json")
    print(f"  对照表        detection_vs_gt.csv")
    print(f"  数据集标签    {dataset_label_dir}/gripper_anchors.csv, arm_states.csv, meta_task*.json")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--tasks", default="all",
                    help="要跑的任务, 逗号分隔 (如 '3,5') 或 'all'")
    ap.add_argument("--episodes", default=None,
                    help="覆盖配置的 episode 白名单, 逗号分隔 (仅 --tasks 指定单个任务时有效)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-video", action="store_true", help="不导出边界帧图片")
    ap.add_argument("--no-plot", action="store_true", help="不出预览图")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    csv_path = resolve_csv(cfg["dataset"]["_csv"])
    out_root = Path(args.out)
    label_dir = out_root / "_dataset_labels"
    out_root.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    if args.tasks == "all":
        task_ids = sorted(int(k) for k in cfg["tasks"])
    else:
        task_ids = [int(x) for x in args.tasks.split(",") if x.strip()]

    print(f"csv      : {csv_path.relative_to(ROOT)}")
    print(f"tasks    : {task_ids}")
    print(f"normalize: 逐臂 p{cfg['normalize']['lo_pct']:g}~p{cfg['normalize']['hi_pct']:g}")
    print(f"out      : {out_root.relative_to(ROOT)}")

    install_real_hooks(cfg, csv_path)

    per_task: dict[int, Path] = {}
    for ti in task_ids:
        if args.episodes and len(task_ids) == 1:
            episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
        else:
            episodes = [int(e) for e in cfg["_episodes"][str(ti)]]
        per_task[ti] = run_one_task(cfg, csv_path, ti, episodes, out_root, label_dir,
                                    no_video=args.no_video, no_plot=args.no_plot)

    collect(cfg, out_root, label_dir, per_task)


if __name__ == "__main__":
    main()
