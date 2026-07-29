"""Environment-side execution loop for SmolVLA joint-action chunks."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        observation = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=observation)
        actions = model_client.call(func_name="get_action")
        for action_index, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if (
                TASK_ENV.is_episode_end()
                or action_index + 1 == len(actions)
            ):
                break

            # EvalEnv records video frames when observations are captured, not
            # when actions are applied. Capture every intermediate chunk step
            # so a T-action chunk produces T video frames instead of one.
            observation = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=observation)


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_indices = TASK_ENV.get_running_env_idx_list()
        observations = TASK_ENV.get_obs_batch(env_indices)
        model_client.call(func_name="update_obs_batch", obs=observations)
        action_chunks = model_client.call(
            func_name="get_action_batch", obs=env_indices
        )
        chunk_size = len(action_chunks[0])
        for action_index in range(chunk_size):
            TASK_ENV.take_action_batch(
                [chunk[action_index] for chunk in action_chunks],
                env_indices,
            )
            if (
                TASK_ENV.is_episode_end()
                or action_index + 1 == chunk_size
            ):
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_indices = [
                index
                for index, env_idx in enumerate(env_indices)
                if env_idx in running
            ]
            action_chunks = [
                action_chunks[index] for index in active_batch_indices
            ]
            env_indices = [
                env_indices[index] for index in active_batch_indices
            ]
            observations = TASK_ENV.get_obs_batch(env_indices)
            model_client.call(
                func_name="update_obs_batch",
                obs=observations,
            )
