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

    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        env_idx_list = TASK_ENV.get_running_env_idx_list() # Get Running Environment Index List
        obs_list = TASK_ENV.get_obs_batch(env_idx_list) # Get Observation
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)  # Get Action according to observation chunk

        # 各 env 的 chunk 长度可能不同（PACE 逐 env 选执行视野 h），而 take_action_batch
        # 要求各 env 步数一致，故取批内最小长度对齐（同论文多臂取最早边界的规则）。
        # PACE 关闭时各 env 长度恒等于 actions_per_chunk，min 退化为 len(actions[0])，
        # 与旧行为完全一致。
        chunk_size = min(len(env_actions) for env_actions in actions) # Get the chunk size
        for action_idx in range(chunk_size): # Iterate over the action chunk
            current_action_list = [env_actions[action_idx] for env_actions in actions] # Get the current action list
            TASK_ENV.take_action_batch(current_action_list, env_idx_list) # Take the action

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size: # Check whether the episode ends
                break

            running = set(TASK_ENV.get_running_env_idx_list()) # Get the running environment index list
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running] # Get the active batch index

            actions = [actions[i] for i in active_batch_idx] # Get the active action list
            env_idx_list = [env_idx_list[i] for i in active_batch_idx] # Get the active environment index list
            model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list)) # Update the observation


# ===== 性能剖析钩子（profile_eval.py，仅 PROFILE_EVAL=1 激活，否则零副作用）=====
import os as _os

if _os.environ.get("PROFILE_EVAL") == "1":
    try:
        from . import profile_eval
        profile_eval.install()
    except Exception as _e:
        print(f"[profile] install failed: {_e!r}", flush=True)