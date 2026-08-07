"""环境端执行循环（xvla_2）：全 chunk 模式 + 可选每步 [xvla_2][sim] 日志。

与官方 X_VLA 的 deploy.py 一致：每次 get_action 拿到完整 chunk（默认 30 个动作），
全部执行完再请求下一次预测；**每个 take_action 之后都调用 get_obs()** —— 这保证
仿真自带的 _stream_vision 逐帧录制完整视频（只做预测帧的视频是不完整的根因）。

本模块只依赖 TASK_ENV 公开 API + WsModelClient + print，不改动任何 RoboDojo
仿真代码。sim_step_log 由同目录 deploy.yml 控制（默认 false）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _sim_step_log_enabled() -> bool:
    try:
        import yaml

        cfg = yaml.safe_load(Path(__file__).with_name("deploy.yml").read_text(encoding="utf-8"))
        return bool((cfg or {}).get("sim_step_log", False))
    except Exception:
        return False


def _image_summary(value) -> dict:
    if isinstance(value, dict):
        for key in ("color", "rgb"):
            if key in value:
                value = value[key]
                break
    arr = np.asarray(value)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _log_step(step: int, env_idx: int, obs: dict) -> None:
    state = obs.get("state", {})
    state16 = (
        list(state.get("left_ee_pose", []))
        + list(state.get("left_ee_joint_state", []))
        + list(state.get("right_ee_pose", []))
        + list(state.get("right_ee_joint_state", []))
    )
    images = {cam: _image_summary(value) for cam, value in obs.get("vision", {}).items()}
    print(
        "[xvla_2][sim] "
        + json.dumps(
            {"event": "step_observation", "step": step, "env_idx": env_idx,
             "state16": state16, "images": images},
            ensure_ascii=False,
        ),
        flush=True,
    )


def eval_one_episode(TASK_ENV, model_client):
    sim_step_log = _sim_step_log_enabled()

    model_client.call(func_name="reset")  # reset policy

    step = 0
    while not TASK_ENV.is_episode_end():  # Check whether the episode ends
        obs = TASK_ENV.get_obs()  # Get Observation
        if sim_step_log:
            _log_step(step, obs.get("env_idx", 0), obs)
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation
        actions = model_client.call(func_name="get_action")  # 全 chunk（默认 30 个动作）

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            step += 1

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = TASK_ENV.get_obs()  # 每步 get_obs → 仿真逐帧录制完整视频
            if sim_step_log:
                _log_step(step, obs.get("env_idx", 0), obs)
            model_client.call(func_name="update_obs", obs=obs)

    if sim_step_log:
        print(
            "[xvla_2][sim] "
            + json.dumps({"event": "episode_end", "step": step}, ensure_ascii=False),
            flush=True,
        )


def eval_one_episode_batch(TASK_ENV, model_client):
    sim_step_log = _sim_step_log_enabled()
    step_counts: dict[int, int] = {}

    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():  # Check whether the episode ends
        env_idx_list = TASK_ENV.get_running_env_idx_list()  # Get Running Environment Index List
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)  # Get Observation
        if sim_step_log:
            for env_idx, obs in zip(env_idx_list, obs_list, strict=True):
                _log_step(step_counts.get(int(env_idx), 0), int(env_idx), obs)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(
            func_name="get_action_batch", obs=env_idx_list
        )  # Get Action according to observation chunk

        chunk_size = len(actions[0])  # Get the chunk size
        for action_idx in range(chunk_size):  # Iterate over the action chunk
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)  # Take the action
            for env_idx in env_idx_list:
                key = int(env_idx)
                step_counts[key] = step_counts.get(key, 0) + 1

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:  # Check whether the episode ends
                break

            running = set(TASK_ENV.get_running_env_idx_list())  # Get the running environment index list
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            actions = [actions[i] for i in active_batch_idx]  # Get the active action list
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            next_obs = TASK_ENV.get_obs_batch(env_idx_list)  # 每步 get_obs → 逐帧视频
            if sim_step_log:
                for env_idx, obs in zip(env_idx_list, next_obs, strict=True):
                    _log_step(step_counts.get(int(env_idx), 0), int(env_idx), obs)
            model_client.call(func_name="update_obs_batch", obs=next_obs)  # Update the observation
