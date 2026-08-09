"""profile_eval.py — 方法级计时钩子（性能排查用，永久保留但默认不激活）。

用法：deploy.py 末尾已注入（默认注释）：
    # try:
    #     from . import profile_eval
    #     profile_eval.install()
    # except ImportError:
    #     pass

临时测量时取消注释注入块；测量结束再注释回去。install() 会包装
eval_one_episode / eval_one_episode_batch，在 episode 结束时打印各阶段
均耗（mean/min/max/n），覆盖 obs_render_capture、take_action（含
sim_physics / ik_solve / render / capture_step / video_write）、policy_call。

各计时点对应（见 docs/仿真评测性能排查_单episode耗时分析.md 第 4 节）：
- obs_render_capture : TASK_ENV.get_obs(_batch)
- take_action         : TASK_ENV.take_action(_batch)
- render              : TASK_ENV.render
- sim_physics         : TASK_ENV.sim_step（每控制步 collect_interval=10 子步）
- ik_solve            : TASK_ENV.robot_manager.solve_ik
- capture_step        : TASK_ENV.capture_manager.step（tile + GPU→CPU 读回）
- video_write         : VideoStreamWriter.append（3 相机逐帧写盘）
- policy_call         : model_client.call（WS 往返）
"""

from __future__ import annotations

import functools
import sys
import time

_TIMERS: dict[str, list[float]] = {}
_WRAPPED: dict[int, set[str]] = {}  # id(obj) -> set(已包装方法名)


def _accum(label: str, dt: float) -> None:
    _TIMERS.setdefault(label, []).append(dt)


def _wrap_method(obj, method_name: str, label: str) -> None:
    if not hasattr(obj, method_name):
        print(f"[profile] WARN: {type(obj).__name__} has no {method_name}, skip {label}")
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
    print(f"[profile] installed on {label}")


def _wrap_class_method(cls, method_name: str, label: str) -> None:
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
    print(f"[profile] installed on {label}")


def _report() -> None:
    print("[profile] ---- per-stage timings ----")
    for label, samples in _TIMERS.items():
        if not samples:
            continue
        n = len(samples)
        mean_ms = sum(samples) / n * 1000.0
        mn = min(samples) * 1000.0
        mx = max(samples) * 1000.0
        print(f"[profile] {label}: n={n} mean={mean_ms:8.2f}ms min={mn:8.2f}ms max={mx:8.2f}ms")


def _install_env_timers(env) -> None:
    _wrap_method(env, "get_obs", "obs_render_capture")
    _wrap_method(env, "get_obs_batch", "obs_render_capture")
    _wrap_method(env, "take_action", "take_action")
    _wrap_method(env, "take_action_batch", "take_action")
    _wrap_method(env, "render", "render")
    _wrap_method(env, "sim_step", "sim_physics")
    if getattr(env, "robot_manager", None) is not None:
        _wrap_method(env.robot_manager, "solve_ik", "ik_solve")
        _wrap_method(env.robot_manager, "solve_ik_batch", "ik_solve_batch")
    if getattr(env, "capture_manager", None) is not None:
        _wrap_method(env.capture_manager, "step", "capture_step")
    try:
        from utils.save_file import VideoStreamWriter

        _wrap_class_method(VideoStreamWriter, "append", "video_write")
    except Exception as e:
        print(f"[profile] WARN: VideoStreamWriter patch failed: {e}")


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
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            wall = time.perf_counter() - t0
            print(f"[profile] ==== {func.__name__} episode wall = {wall:.1f}s ====")
            _report()

    return wrapper


def install() -> None:
    """包装调用方（deploy.py）模块里的 eval_one_episode(_batch)。"""
    frame = sys._getframe(1)
    module = sys.modules.get(frame.f_globals.get("__name__", ""))
    if module is None:
        print("[profile] WARN: cannot find caller module, install skipped")
        return
    module.eval_one_episode = _wrap_module_func(module.eval_one_episode)
    module.eval_one_episode_batch = _wrap_module_func(module.eval_one_episode_batch)
    print("[profile] installed on eval_one_episode")
    print("[profile] installed on eval_one_episode_batch")
