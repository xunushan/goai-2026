"""profile_eval.py — 方法级计时钩子（fill_pen_holder 并行度基线复测用，默认不激活）。

用法：deploy.py 末尾已加 env-var 门控注入块；仅当运行环境设了
`PROFILE_EVAL=1` 时才安装，否则零副作用。安装后包装
eval_one_episode / eval_one_episode_batch，在 episode 结束时打印各
阶段统计（n/total/mean/min/max），并额外打印 episode 墙钟与 take_action
总次数（≈ policy step 数），便于折算"每 policy step 均耗"。

标签层级（注意嵌套标签为包含关系，解释时勿直接相加）：
- obs_render_capture : TASK_ENV.get_obs_batch（含 render/capture/读回/视频）
    ├ render          : TASK_ENV.render（env 无关，恒 ~40ms）
    ├ capture_step    : TASK_ENV.capture_manager.step（tile + GPU→CPU 读回）
    └ video_write     : VideoStreamWriter.append（3 相机逐帧写盘）
- take_action        : TASK_ENV.take_action_batch（IK+插值+10 子步 drain+reward+判终）
    ├ ik_solve_batch  : robot_manager.solve_ik_batch（批量 IK）
    ├ process_control_info : 插值 + 单 env 读回（80/20 插 10 子步）
    ├ step_drain      : TASK_ENV.step（1 子步 = ctrl pop + super.step + sim）
    │   ├ ctrl_pop    : control_manager.pop（每子步每 env 的 gripper 读回 sync）
    │   └ sim_physics : TASK_ENV.sim_step（物理子步）
    ├ reward_step     : reward_manager.step（逐 env python 结算）
    └ is_episode_end  : TASK_ENV.is_episode_end（reward get_reward 判终）
- policy_call        : model_client.call（WS 往返，get_action_batch 批推理）
"""

import functools
import os
import sys
import time

_ENABLED = os.environ.get("PROFILE_EVAL") == "1"
_TIMERS: dict[str, list[float]] = {}
_WRAPPED: dict[int, set[str]] = {}  # id(obj) -> set(已包装方法名)


def _accum(label: str, dt: float) -> None:
    _TIMERS.setdefault(label, []).append(dt)


def _wrap_method(obj, method_name: str, label: str) -> None:
    if not _ENABLED:
        return
    if not hasattr(obj, method_name):
        print(f"[profile] WARN: {type(obj).__name__} has no {method_name}, skip {label}", flush=True)
        return
    key = id(obj)
    if key not in _WRAPPED:
        _WRAPPED[key] = set()
    if method_name in _WRAPPED[key]:
        return
    _WRAPPED[key].add(method_name)
    orig = getattr(obj, method_name)

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            _accum(label, time.perf_counter() - t0)

    setattr(obj, method_name, wrapper)
    print(f"[profile] installed on {label}", flush=True)


def _wrap_class_method(cls, method_name: str, label: str) -> None:
    if not _ENABLED:
        return
    key = id(cls)
    if key in _WRAPPED and method_name in _WRAPPED[key]:
        return
    _WRAPPED.setdefault(key, set()).add(method_name)
    orig = getattr(cls, method_name)

    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            _accum(label, time.perf_counter() - t0)

    setattr(cls, method_name, wrapper)
    print(f"[profile] installed on {label}", flush=True)


def _snapshot():
    return {k: len(v) for k, v in _TIMERS.items()}


def _report(base=None) -> None:
    print("[profile] ---- per-stage timings ----", flush=True)
    for label, samples in _TIMERS.items():
        if not samples:
            continue
        n0 = (base or {}).get(label, 0)
        samples = samples[n0:]
        if not samples:
            continue
        n = len(samples)
        total = sum(samples)
        mean_ms = total / n * 1000.0
        mn = min(samples) * 1000.0
        mx = max(samples) * 1000.0
        print(
            f"[profile] {label}: n={n} total={total:.3f}s mean={mean_ms:8.2f}ms "
            f"min={mn:8.2f}ms max={mx:8.2f}ms",
            flush=True,
        )


def _install_env_timers(env) -> None:
    # --- obs 链（嵌套：obs ⊃ render/capture_step/video_write）---
    _wrap_method(env, "get_obs", "obs_render_capture")
    _wrap_method(env, "get_obs_batch", "obs_render_capture")
    _wrap_method(env, "render", "render")
    # --- 控制/物理链（嵌套：take_action ⊃ process_control_info/step_drain/...）---
    _wrap_method(env, "take_action", "take_action")
    _wrap_method(env, "take_action_batch", "take_action")
    _wrap_method(env, "process_control_info", "process_control_info")
    _wrap_method(env, "step", "step_drain")
    _wrap_method(env, "sim_step", "sim_physics")
    _wrap_method(env, "is_episode_end", "is_episode_end")
    if getattr(env, "robot_manager", None) is not None:
        _wrap_method(env.robot_manager, "solve_ik", "ik_solve")
        _wrap_method(env.robot_manager, "solve_ik_batch", "ik_solve_batch")
        if getattr(env.robot_manager, "control_manager", None) is not None:
            _wrap_method(env.robot_manager.control_manager, "pop", "ctrl_pop")
    if getattr(env, "reward_manager", None) is not None:
        _wrap_method(env.reward_manager, "step", "reward_step")
    if getattr(env, "capture_manager", None) is not None:
        _wrap_method(env.capture_manager, "step", "capture_step")
    try:
        from utils.save_file import VideoStreamWriter

        _wrap_class_method(VideoStreamWriter, "append", "video_write")
    except Exception as e:
        print(f"[profile] WARN: VideoStreamWriter patch failed: {e}", flush=True)


def _wrap_module_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # eval_env.py 以关键字传参调用：eval_one_episode_batch(TASK_ENV=..., model_client=...)
        env = args[0] if args else kwargs.get("TASK_ENV")
        client = args[1] if len(args) > 1 else kwargs.get("model_client")
        if env is not None:
            _install_env_timers(env)
        if client is not None:
            _wrap_method(client, "call", "policy_call")
        base = _snapshot()
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            wall = time.perf_counter() - t0
            n_take = _TIMERS.get("take_action", [])
            n_take_delta = max(0, len(n_take) - (base or {}).get("take_action", 0))
            print(f"[profile] ==== {func.__name__} episode wall = {wall:.1f}s, take_action(policy steps) = {n_take_delta} ====", flush=True)
            _report(base)

    return wrapper


def install() -> None:
    """包装调用方（deploy.py）模块里的 eval_one_episode(_batch)。"""
    if not _ENABLED:
        return
    frame = sys._getframe(1)
    module = sys.modules.get(frame.f_globals.get("__name__", ""))
    if module is None:
        print("[profile] WARN: cannot find caller module, install skipped", flush=True)
        return
    for name in ("eval_one_episode", "eval_one_episode_batch"):
        if hasattr(module, name):
            setattr(module, name, _wrap_module_func(getattr(module, name)))
            print(f"[profile] installed on {name}", flush=True)
