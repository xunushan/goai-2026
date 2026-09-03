import os

def _mid_step_mode():
    """读取 deploy.yml mid_step_obs → full|video|none。

    中间步（chunk 内非首帧）的 obs 在下一个 chunk 边界的重观察中整体覆盖，属死写；
    开关只影响中间步成本，真 obs（重规划，while 顶部）路径不变。不可读/非法一律回退 full。
    """
    mode = "full"
    try:
        from utils.load_file import load_yaml

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy.yml")
        mode = str(load_yaml(path).get("mid_step_obs", "full")).strip().lower()
    except Exception as e:
        print(f"[deploy] mid_step_obs unreadable, fallback full ({e!r})", flush=True)
        mode = "full"
    if mode not in ("full", "video", "none"):
        print(f"[deploy] invalid mid_step_obs={mode!r}, fallback full", flush=True)
        mode = "full"
    print(f"[deploy] mid_step_obs = {mode}", flush=True)
    return mode


def eval_one_episode(TASK_ENV, model_client):

    model_client.call(func_name="reset") # reset policy

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        obs = TASK_ENV.get_obs() # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)

def eval_one_episode_batch(TASK_ENV, model_client):

    mid_mode = _mid_step_mode()  # full | video | none（deploy.yml mid_step_obs）

    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        env_idx_list = TASK_ENV.get_running_env_idx_list() # Get Running Environment Index List
        obs_list = TASK_ENV.get_obs_batch(env_idx_list) # Get Observation
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)  # Get Action according to observation chunk

        chunk_size = len(actions[0]) # Get the chunk size
        for action_idx in range(chunk_size): # Iterate over the action chunk
            current_action_list = [env_actions[action_idx] for env_actions in actions] # Get the current action list
            TASK_ENV.take_action_batch(current_action_list, env_idx_list) # Take the action

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size: # Check whether the episode ends
                break

            running = set(TASK_ENV.get_running_env_idx_list()) # Get the running environment index list
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running] # Get the active batch index

            actions = [actions[i] for i in active_batch_idx] # Get the active action list
            env_idx_list = [env_idx_list[i] for i in active_batch_idx] # Get the active environment index list

            if mid_mode == "video":
                # 中间步只出帧+写 mp4：policy 只在中段后的下一次重规划（while 顶部）
                # 用 obs，这里组装的 obs 属死写。跳状态读回/组装/深拷贝/推服务端，
                # 录像帧不丢（_stream_vision 仍逐帧消费 color）。
                TASK_ENV.get_obs_batch(env_idx_list, vision_only=True)
            elif mid_mode == "none":
                # 中间步完全不取 obs（无 render/capture/录像）→ 视频只剩重规划帧；
                # 下个 chunk 边界（while 顶部）照常全量重新观察 + 推服务端 + 推理。
                pass
            else:  # full（默认）：与历史行为完全一致
                model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list)) # Update the observation


# ===== 性能剖析钩子（profile_eval.py，仅 PROFILE_EVAL=1 激活，否则零副作用）=====
import os as _os

if _os.environ.get("PROFILE_EVAL") == "1":
    try:
        from . import profile_eval
        profile_eval.install()
    except Exception as _e:
        print(f"[profile] install failed: {_e!r}", flush=True)