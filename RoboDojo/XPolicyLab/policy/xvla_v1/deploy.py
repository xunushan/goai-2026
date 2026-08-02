"""Execute complete 30-step X-VLA action chunks in RoboDojo."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_indices = TASK_ENV.get_running_env_idx_list()
        observations = TASK_ENV.get_obs_batch(env_indices)
        model_client.call(func_name="update_obs_batch", obs=observations)
        chunks = model_client.call(func_name="get_action_batch", obs=env_indices)
        for step in range(len(chunks[0])):
            TASK_ENV.take_action_batch([chunk[step] for chunk in chunks], env_indices)
            if TASK_ENV.is_episode_end():
                break
