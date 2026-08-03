"""Execute complete 30-step X-VLA action chunks in RoboDojo."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        for action_index, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_index + 1 == len(actions):
                break

            # EvalEnv records/streams video inside get_obs(), not take_action().
            # Capture every intermediate execution step without asking the
            # policy to predict again. The final step is captured by the next
            # outer-loop observation before the following chunk is inferred.
            TASK_ENV.get_obs()


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_indices = TASK_ENV.get_running_env_idx_list()
        observations = TASK_ENV.get_obs_batch(env_indices)
        model_client.call(func_name="update_obs_batch", obs=observations)
        chunks = model_client.call(func_name="get_action_batch", obs=env_indices)
        for step in range(len(chunks[0])):
            TASK_ENV.take_action_batch([chunk[step] for chunk in chunks], env_indices)
            if TASK_ENV.is_episode_end() or step + 1 == len(chunks[0]):
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_positions = [
                position
                for position, env_idx in enumerate(env_indices)
                if env_idx in running
            ]
            chunks = [chunks[position] for position in active_positions]
            env_indices = [env_indices[position] for position in active_positions]
            if not env_indices:
                break

            # Capture video for every intermediate chunk step. No policy RPC
            # is made here; inference still happens once per action chunk.
            TASK_ENV.get_obs_batch(env_indices)
