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
    """批评估：每个 env 按自己的动作缓冲独立重规划。

    take_action_batch 要求各 env 同步走一步，但每个 env 的 chunk 一旦耗尽就该重新
    推理，不能按「批内统一 chunk 长度」推进。这里给每个 env 维护一个动作缓冲，缓冲
    耗尽的 env 才重新推理，其余 env 继续消费自己的剩余动作。

    当前服务端每次返回的动作数恒为 actions_per_chunk，所有缓冲同时耗尽，行为与
    「按批内统一长度推进」的旧实现逐调用一致（test_deploy_batch.py 逐条比对了动作
    序列与推理调用序列）。

    注意 get_obs_batch 承担 render / capture / 视频写盘，必须每步对全部 running env
    调用（跳过中间步的刷新会破坏精度，见 X_VLA mid_step_obs 三档的教训）。
    """
    model_client.call(func_name="reset")
    buffers: dict[int, list] = {} # env_idx -> 该 env 尚未执行完的动作

    while True: # Check whether the episode ends
        env_idx_list = TASK_ENV.get_running_env_idx_list() # Get Running Environment Index List
        if not env_idx_list:
            break
        obs_list = TASK_ENV.get_obs_batch(env_idx_list) # Get Observation（含 render/capture/视频）

        need_replan = [env_idx for env_idx in env_idx_list if not buffers.get(env_idx)] # 动作缓冲耗尽的 env
        if need_replan:
            obs_by_env = dict(zip(env_idx_list, obs_list))
            model_client.call(func_name="update_obs_batch", obs=[obs_by_env[i] for i in need_replan])
            replan_actions = model_client.call(func_name="get_action_batch", obs=need_replan) # Get Action according to observation chunk
            if len(replan_actions) != len(need_replan):
                raise AssertionError(f"get_action_batch returned {len(replan_actions)} chunks for {len(need_replan)} envs")
            for env_idx, env_actions in zip(need_replan, replan_actions):
                if not env_actions:
                    raise AssertionError(f"empty action chunk for env {env_idx}")
                buffers[env_idx] = list(env_actions)

        current_action_list = [buffers[env_idx].pop(0) for env_idx in env_idx_list] # Get the current action list
        TASK_ENV.take_action_batch(current_action_list, env_idx_list) # Take the action

        TASK_ENV.is_episode_end() # 与旧实现同位置：每步刷新 end_flag（判终/reward 结算）
        running = set(TASK_ENV.get_running_env_idx_list()) # Get the running environment index list
        for env_idx in [i for i in buffers if i not in running]: # 已结束的 env 丢弃其剩余缓冲
            del buffers[env_idx]


# ===== 性能剖析钩子（profile_eval.py，仅 PROFILE_EVAL=1 激活，否则零副作用）=====
import os as _os

if _os.environ.get("PROFILE_EVAL") == "1":
    try:
        from . import profile_eval
        profile_eval.install()
    except Exception as _e:
        print(f"[profile] install failed: {_e!r}", flush=True)