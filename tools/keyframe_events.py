#!/usr/bin/env python3
"""关键帧事件范围界定 + 训练权重 (配置驱动) —— 供 stage_label / frame_weight_trapezoid 共用。

本模块是"关键帧范围"与"权重参数"的唯一代码实现, 所有窗口/权重数值来自配置文件
(默认 configs/keyframe_weight_config.json), 不写死在代码里。流程分三步:

  1. 爪夹关键节点检测: tools/keyframe_detect.py 的 detect_gripper_keyframes,
     把左右爪夹曲线解析成若干"抓取周期" (close_start, hold_start, hold_end, open_full)。
  2. 事件范围界定: 按任务把每个周期的关键节点(hold_start/hold_end)归为具体事件
     (抓/放/接取/交接/插入...), 每个事件有参考帧 t0 和一组窗口参数 (见下)。角色判定见
     `_role_cycle_map` (fill 持筒臂/笔臂、plug 插入周期/非插入周期、stack 双臂)。
  3. 权重赋值: 对每个事件 t0 生成非对称梯形权重窗, 逐帧取各事件窗口的 max 得到
     frame_weight_loss; 由关键帧区间生成二值 is_key_frame 与常数 frame_weight_sampling;
     逐帧关键帧标签 = 命中窗口的事件 key 集合 ('|' 连接, 无则 'none')。

【两个训练字段】(doc v2 §1 开头 / §5.2, 严禁混用; 旧名 frame_weight 已废止)
    frame_weight_loss[t]     : 未来 action target 的逐帧 loss 权重 = 梯形事件权重取 max。
    frame_weight_sampling[t] : 当前 observation 的重采样权重 = 关键帧常数(默认 2) / 普通帧 1。
    is_key_frame[t]          : 二值关键帧标记, 按事件窗口闭区间直接生成。**不得**用
                               『frame_weight_loss > 1』反推 —— 梯形在窗口两端点恰好回到 1。

范围/权重定义 (方案见 docs/dual_arm_tasks_failure_and_keyframe_plan.md §1):
    参数 L (前窗口), R (后窗口), Pl (最大权重区间左端), Pr (最大权重区间右端), W (峰值);
    四边界 a=t0-L, b=t0+Pl, c=t0+Pr, d=t0+R (需 a < b <= c < d);
    权重 w(t): t<a → 1; [a,b) 线性 1→W; [b,c] 恒 W; (c,d] 线性 W→1; t>d → 1;
    关键帧标签范围 = [a, d] = [t0-L, t0+R] (含端点);
    帧同时落在多个事件窗内 → 标签并列, 权重取 max (不累加)。
    v2 统一窗口: 所有任务/事件 L=20, R=5, Pl=-5, Pr=0, 仅 W 因事件而异。

配置 schema (configs/keyframe_weight_config.json):
    {"sampling": {"key_value": 2, "normal_value": 1},
     "tasks": {"<task_index>": {"slug":..., "events": [
        {"key","cn","anchor","role","L","R","Pl","Pr","W","skip_incomplete"?}
    ]}}}
  anchor: "hold_start" | "hold_end" (事件参考帧取周期的哪个关键节点);
  role  : 事件绑定到哪些周期 (fill: holder/pen; plug: noninsert_hs/insert_hs/
          noninsert_he/insert_he; stack: all_hs/all_he), 语义见配置文件 _comment;
  skip_incomplete: 仅对 hold_end 事件有效, 默认 true —— 周期结尾未回开(open_full=-1)
          说明无真实释放, 该 hold_end 事件不产生; plug 的 insert 设 false。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.keyframe_detect import GRIP_L, GRIP_R, INCOMPLETE, detect_gripper_keyframes

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CSV = ROOT / "data" / "sim_lerobot_v30_ee" / "sim_lerobot_v30_ee.csv"
DEFAULT_CONFIG = ROOT / "configs" / "keyframe_weight_config.json"
DEFAULT_TASKS = ROOT / "data" / "sim_lerobot_v30_ee" / "meta" / "tasks.parquet"

WEIGHT_BASE = 1.0          # frame_weight_loss 的普通帧/窗口外取值
NONE_LABEL = "none"        # 非关键帧标签
MULTI_SEP = "|"            # 多标签连接符

# frame_weight_sampling 取值缺省 (配置 sampling 段可覆盖; doc §5.2: 关键帧 2, 普通帧 1)
SAMPLING_KEY_VALUE = 2.0
SAMPLING_NORMAL_VALUE = 1.0

# sim 数据集固定 3 任务 task_index -> slug (文件名用); 角色语义见配置 _comment
TASK_SLUGS: dict[int, str] = {
    0: "fill_pen_holder",
    1: "plug_in_charger",
    2: "stack_bowls",
}

# 事件调色板 (可视化共用; 缺省灰)
NONE_COLOR = "#eceff1"
EVENT_COLORS: dict[str, str] = {
    # fill
    "grasp_holder": "#2a78d6",
    "place_holder": "#b07bd8",
    "grasp_pen": "#eb6834",
    "place_pen": "#4c9f38",
    # plug
    "grasp_charger": "#2ca25f",
    "receive_charger": "#3aa0c9",
    "release_charger": "#9467bd",
    "insert": "#d62728",
    # stack
    "grasp_bowl": "#2ca25f",
    "place_bowl": "#d62728",
}


def event_color(key: str) -> str:
    return EVENT_COLORS.get(key, "#7f7f7f")

# fill: 持筒臂周期须跨段至少该比例才认定为"整段持筒"
HOLDER_SURE_FRAC = 0.5


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_config(config_path: Path | None = None) -> dict:
    """读取关键帧配置 (默认 configs/keyframe_weight_config.json)。"""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_file():
        raise SystemExit(f"未找到关键帧配置: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def task_events(config: dict, ti: int) -> list[dict]:
    """取某任务的事件定义列表 (保持配置中的先后顺序 —— 用于标签优先级/连接顺序)。"""
    key = str(ti)
    if key not in config.get("tasks", {}):
        raise SystemExit(f"task_index={ti} 未在关键帧配置中定义 (configs/"
                         f"keyframe_weight_config.json)")
    return config["tasks"][key]["events"]


def task_slug(config: dict, ti: int) -> str:
    ev = config.get("tasks", {}).get(str(ti), {})
    return ev.get("slug") or TASK_SLUGS.get(ti, f"task_{ti:03d}")


def sampling_values(config: dict) -> tuple[float, float]:
    """frame_weight_sampling 的 (关键帧常数, 普通帧常数), 配置 sampling 段可覆盖。"""
    s = config.get("sampling", {})
    return (float(s.get("key_value", SAMPLING_KEY_VALUE)),
            float(s.get("normal_value", SAMPLING_NORMAL_VALUE)))


# ---------------------------------------------------------------------------
# 数据加载 (供各工具复用)
# ---------------------------------------------------------------------------

def load_tasks(tasks_parquet: Path = DEFAULT_TASKS) -> dict[int, str]:
    """读取 tasks.parquet -> {task_index: 完整指令}。"""
    if not Path(tasks_parquet).is_file():
        return {}
    import pyarrow.parquet as pq

    t = pq.read_table(str(tasks_parquet)).to_pandas()
    return {int(v): str(k) for k, v in t["task_index"].items()}


def load_state_cols(csv_path: Path) -> pd.DataFrame:
    """读取 CSV 的 episode/task/frame/state 列。"""
    return pd.read_csv(
        csv_path,
        usecols=["episode_index", "task_index", "frame_index", "observation.state"],
    )


def parse_state_series(s: pd.Series) -> np.ndarray:
    """state 字符串序列 -> (T,16) 数组。"""
    return np.asarray(
        [np.fromstring(x.strip("[]"), sep=",", dtype=float) for x in s],
        dtype=float,
    )


def load_train_episodes(split_json: Path) -> set[int] | None:
    if not Path(split_json).is_file():
        return None
    d = json.loads(Path(split_json).read_text(encoding="utf-8"))
    train: set[int] = set()
    for cfg in d["tasks"].values():
        train.update(int(e) for e in cfg["train_episode_idx"])
    return train


def pick_episodes(eps: np.ndarray, n: int) -> np.ndarray:
    """在 ep 排序列表里均匀取 n 个 (首/中/尾), 确定性覆盖不同长度轨迹。"""
    if n >= len(eps):
        return eps
    if n <= 1:
        return np.asarray([eps[len(eps) // 2]], dtype=int)
    return np.asarray(
        [eps[int(round(i * (len(eps) - 1) / (n - 1)))] for i in range(n)],
        dtype=int,
    )


# ---------------------------------------------------------------------------
# 周期工具
# ---------------------------------------------------------------------------

def side_cycles(per_side: dict, side: str) -> list[tuple[int, int, int, int]]:
    """某侧抓取周期列表, 每项 = (close_start, hold_start, hold_end, open_full)。"""
    kf = per_side[side]
    return [(int(kf["close_start"][i]), int(kf["hold_start"][i]),
             int(kf["hold_end"][i]), int(kf["open_full"][i]))
            for i in range(len(kf["close_start"]))]


def detect_sides(state: np.ndarray, frame: np.ndarray,
                 detect_kwargs: dict) -> dict:
    """左右爪夹关键节点检测 -> {"left": {close_start,...}, "right": {...}}。"""
    per_side = {}
    for side, idx in (("left", GRIP_L), ("right", GRIP_R)):
        kf = detect_gripper_keyframes(state[:, idx], frame, **detect_kwargs)
        per_side[side] = {k: np.asarray(v, dtype=int) for k, v in kf.items()}
    return per_side


def window(t0: int, pre: int, post: int, T: int):
    """事件窗口 [t0-pre, t0+post] 裁剪到 [0, T-1]; 无效返回 None。"""
    a = max(0, t0 - pre)
    b = min(T - 1, t0 + post)
    return (a, b) if a <= b else None


def resolve_fill_roles(state: np.ndarray, per_side: dict) -> dict:
    """fill 持筒臂/笔臂判定。

    持筒臂 = 保持跨度最大的臂; 其"good 周期" = 保持跨度 >= episode 长度一半的周期
    (若都没有则退化为跨度最大的那个周期)。另一臂为笔臂, 其全部周期都算抓/放笔。
    返回 {"holder": side, "pen": side, "good_cycles": {side: cycles}}。
    """
    cycle_lists = {s: side_cycles(per_side, s) for s in ("left", "right")}

    def span(c):
        return (c[2] if c[2] != INCOMPLETE else int(1e9)) - c[1]

    span_max = {s: max((span(c) for c in cycle_lists[s]), default=-1)
                for s in ("left", "right")}
    if all(v <= 0 for v in span_max.values()):
        raise ValueError("两侧均无抓取周期, 无法判定持筒臂")
    holder = max(("left", "right"), key=lambda s: span_max[s])
    pen = "right" if holder == "left" else "left"
    ep_len = len(state)
    good = [c for c in cycle_lists[holder]
            if ((c[2] if c[2] != INCOMPLETE else ep_len - 1) - c[1])
            >= HOLDER_SURE_FRAC * ep_len]
    if not good:
        good = [max(cycle_lists[holder], key=span)]
    return {"holder": holder, "pen": pen,
            "good_cycles": {holder: good, pen: cycle_lists[pen]}}


def _role_cycle_map(ti: int, state: np.ndarray, per_side: dict):
    """按任务把事件 role 映射到具体抓取周期: {role: [(side, cycle), ...]}。

    返回 (role_cycles, meta)。meta 为可读角色说明, 供可视化/统计展示。
    """
    cycle_lists = {s: side_cycles(per_side, s) for s in ("left", "right")}
    all_c = [(s, c) for s in ("left", "right") for c in cycle_lists[s]]

    if ti == 0:                                     # fill: holder / pen
        roles = resolve_fill_roles(state, per_side)
        holder = roles["holder"]
        role_cycles = {
            "holder": [(holder, c) for c in roles["good_cycles"][holder]],
            "pen": [(roles["pen"], c) for c in cycle_lists[roles["pen"]]],
        }
        hc = roles["good_cycles"][holder][0]
        h_span = ((hc[2] if hc[2] != INCOMPLETE else len(state) - 1) - hc[1]) / len(state)
        meta = (f"holder={holder}(持筒臂 {h_span:.0%}) pen={roles['pen']}")
    elif ti == 1:                                   # plug: 抓取/接取/交接释放/插入
        insert_side, insert_cyc = max(all_c, key=lambda x: x[1][2])
        others = [(s, c) for (s, c) in all_c
                  if not (s == insert_side and c is insert_cyc)]
        # 插入周期: hold_start=接取插头(接取手接住递来的插头), hold_end=插入插头;
        # 其余周期: hold_start=抓取插头, hold_end=交接释放插头 (doc §3.2)
        role_cycles = {
            "noninsert_hs": others,
            "insert_hs": [(insert_side, insert_cyc)],
            "noninsert_he": others,
            "insert_he": [(insert_side, insert_cyc)],
        }
        n = {s: len(cycle_lists[s]) for s in ("left", "right")}
        meta = (f"insert={insert_side} (插入臂)  cycles L={n['left']} R={n['right']}")
    elif ti == 2:                                   # stack: 双臂各周期都算
        role_cycles = {"all_hs": all_c, "all_he": all_c}
        n = {s: len(cycle_lists[s]) for s in ("left", "right")}
        meta = f"cycles L={n['left']} R={n['right']}"
    else:
        raise SystemExit(f"task_index={ti} 暂无角色判定规则")
    return role_cycles, meta


# ---------------------------------------------------------------------------
# 事件实例界定
# ---------------------------------------------------------------------------

def episode_event_instances(ti: int, state: np.ndarray, frame: np.ndarray,
                            detect_kwargs: dict, config: dict) -> dict:
    """单 episode 的事件实例列表 (角色识别 + 范围界定)。

    返回 {"instances":[{key,cn,anchor,side,t0,a,b,L,R,Pl,Pr,W,skip_incomplete}],
          "meta":str, "per_side":..., "T":int}。
    a,b 为裁剪到 [0,T-1] 的关键帧范围 (位置空间, sim 中位置==frame_index)。
    """
    events = task_events(config, ti)
    T = len(state)
    per_side = detect_sides(state, frame, detect_kwargs)
    role_cycles, meta = _role_cycle_map(ti, state, per_side)

    instances: list[dict] = []
    for ev in events:
        for side, cyc in role_cycles.get(ev["role"], []):
            if ev["anchor"] == "hold_start":
                t0 = cyc[1]
            else:
                t0 = cyc[2]
                # 结尾未回开的周期无真实释放 -> 跳过该 hold_end 事件 (insert 例外)
                if ev.get("skip_incomplete", True) and cyc[3] == INCOMPLETE:
                    continue
            ab = window(t0, ev["L"], ev["R"], T)     # 关键帧范围 = [t0-L, t0+R]
            if ab is None:
                continue
            instances.append({**ev, "side": side, "t0": t0,
                              "a": ab[0], "b": ab[1]})
    instances.sort(key=lambda x: (x["a"], x["key"]))
    return {"instances": instances, "meta": meta, "per_side": per_side, "T": T}


# ---------------------------------------------------------------------------
# 梯形权重 + 逐帧标签
# ---------------------------------------------------------------------------

def trapezoid_weight(t0: int, frame: np.ndarray, ev: dict) -> np.ndarray:
    """单个事件 t0 的非对称梯形权重窗 (逐帧, 与 frame 对齐)。

    边界 a=t0-L, b=t0+Pl, c=t0+Pr, d=t0+R; 峰值 W (见文件头公式)。
    """
    L, R, Pl, Pr = ev["L"], ev["R"], ev["Pl"], ev["Pr"]
    W = float(ev["W"])
    a, b, c, d = t0 - L, t0 + Pl, t0 + Pr, t0 + R
    t = np.asarray(frame, dtype=float)
    w = np.full_like(t, WEIGHT_BASE, dtype=float)
    rise = (t > a) & (t < b)
    if b > a:
        w[rise] = WEIGHT_BASE + (W - WEIGHT_BASE) * (t[rise] - a) / (b - a)
    w[(t >= b) & (t <= c)] = W
    fall = (t > c) & (t < d)
    if d > c:
        w[fall] = W - (W - WEIGHT_BASE) * (t[fall] - c) / (d - c)
    return w


def episode_weight_and_labels(ti: int, state: np.ndarray, frame: np.ndarray,
                              detect_kwargs: dict, config: dict) -> dict:
    """单 episode -> 逐帧 frame_weight_loss / frame_weight_sampling / is_key_frame / 左右标签。

    返回 {"weight_loss":np.ndarray, "weight_sampling":np.ndarray,
          "is_key_frame":np.ndarray[bool],
          "is_key_frame_left":..., "is_key_frame_right":...,
          "labels_left":list[str], "labels_right":list[str],
          "instances":[...], "meta":str, "per_side":..., "T":int}。
    - weight_loss = 各事件梯形窗口取 max (普通帧 1);
    - is_key_frame = 落在任一事件窗口 [t0-L, t0+R] 的闭区间标记 (两臂并集, 独立二值,
      不由 weight 反推); is_key_frame_left/right 为分臂版本;
    - weight_sampling = 关键帧常数 / 普通帧常数 (取自配置 sampling 段);
    - 标签按配置事件顺序对命中窗口去重连接 (如 'grasp_pen|place_pen'), 无则 'none';
      事件实例锚定在具体抓取周期上, 故按该周期所属臂拆成 labels_left / labels_right。
    """
    info = episode_event_instances(ti, state, frame, detect_kwargs, config)
    instances = info["instances"]
    T = len(frame)

    # frame_weight_loss: 梯形事件权重逐帧取 max (doc §1.2/§5.3), 不累加
    weight_loss = np.full(T, WEIGHT_BASE, dtype=float)
    for inst in instances:
        weight_loss = np.maximum(weight_loss,
                                 trapezoid_weight(inst["t0"], frame, inst))

    # is_key_frame + 逐帧标签: 事件范围 [a,b] 内的帧打上该 key, 多标签按配置顺序 '|' 连接。
    # 事件实例带 side (其锚定抓取周期所属的臂), 故左/右臂各自出一列标签; 同一事件在两臂
    # 同时命中时两侧标签相同。is_key_frame 仍为两臂并集 (doc §1.3, 驱动 loss/sampling)。
    order = [ev["key"] for ev in task_events(config, ti)]
    cover = {s: {k: np.zeros(T, dtype=bool) for k in order}
             for s in ("left", "right")}
    is_key = np.zeros(T, dtype=bool)
    is_key_side = {s: np.zeros(T, dtype=bool) for s in ("left", "right")}
    for inst in instances:
        sl = slice(inst["a"], inst["b"] + 1)
        cover[inst["side"]][inst["key"]][sl] = True
        is_key[sl] = True
        is_key_side[inst["side"]][sl] = True
    labels_side: dict[str, list[str]] = {}
    for s in ("left", "right"):
        hit = (np.stack([cover[s][k] for k in order], axis=1) if order
               else np.zeros((T, 0), dtype=bool))
        labels_side[s] = [
            MULTI_SEP.join([order[j] for j in range(len(order)) if hit[i, j]])
            or NONE_LABEL for i in range(T)
        ]

    # frame_weight_sampling: 关键帧常数 / 普通帧常数 (doc §5.2/§6.1)
    key_v, normal_v = sampling_values(config)
    weight_sampling = np.where(is_key, key_v, normal_v).astype(float)

    return {"weight_loss": weight_loss, "weight_sampling": weight_sampling,
            "is_key_frame": is_key,
            "is_key_frame_left": is_key_side["left"],
            "is_key_frame_right": is_key_side["right"],
            "labels_left": labels_side["left"],
            "labels_right": labels_side["right"],
            "instances": instances,
            "meta": info["meta"], "per_side": info["per_side"], "T": T}
