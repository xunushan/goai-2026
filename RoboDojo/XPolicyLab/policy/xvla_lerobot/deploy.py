"""Environment-side execution loops for xvla_lerobot action chunks."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        observation = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=observation)
        actions = model_client.call(func_name="get_action")
        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_indices = TASK_ENV.get_running_env_idx_list()
        if not env_indices:
            break
        observations = TASK_ENV.get_obs_batch(env_indices)
        model_client.call(func_name="update_obs_batch", obs=observations)
        action_chunks = model_client.call(func_name="get_action_batch", obs=env_indices)
        if len(action_chunks) != len(env_indices):
            raise ValueError(
                f"Policy returned {len(action_chunks)} chunks for {len(env_indices)} environments"
            )
        if not action_chunks:
            raise ValueError("Policy returned no action chunks for active environments")
        chunk_size = len(action_chunks[0])
        if chunk_size == 0 or any(len(chunk) != chunk_size for chunk in action_chunks):
            raise ValueError("Policy returned empty or inconsistent action chunks")
        for action_index in range(chunk_size):
            TASK_ENV.take_action_batch(
                [chunk[action_index] for chunk in action_chunks], env_indices
            )
            if TASK_ENV.is_episode_end() or action_index + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_positions = [
                index
                for index, env_index in enumerate(env_indices)
                if env_index in running
            ]
            action_chunks = [action_chunks[index] for index in active_positions]
            env_indices = [env_indices[index] for index in active_positions]
            if not env_indices:
                break
            model_client.call(
                func_name="update_obs_batch",
                obs=TASK_ENV.get_obs_batch(env_indices),
            )

__all__ = ["eval_one_episode", "eval_one_episode_batch"]
