import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XPolicyLab.utils.policy_context import PolicyContextResolver  # noqa: E402


def main() -> None:
    context = PolicyContextResolver(None)

    task, episode = context.resolve(
        {"task_name": "stack_blocks_random", "episode_idx": 17}, 0
    )
    assert (task, episode) == ("stack_blocks_random", "17")

    context.reset()
    task, episode = context.resolve(
        {
            "task_name": None,
            "episode_idx": None,
            "instruction": "Stand the bottle upright.",
        },
        0,
    )
    assert task == "stand_up_bottles"
    assert episode.startswith("ep001_")
    assert context.resolve({}, 0) == (task, episode)

    context.reset()
    next_task, next_episode = context.resolve(
        {"instruction": "Stand the bottle upright."}, 0
    )
    assert next_task == task and next_episode != episode

    unknown_task, unknown_episode = context.resolve({"instruction": "unknown"}, 1)
    assert unknown_task == "unknown_task" and unknown_episode
    print("policy context tests passed")


if __name__ == "__main__":
    main()
