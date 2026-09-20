"""Resolve stable task and episode identities for policy logs and artifacts."""

from __future__ import annotations

from typing import Any, Mapping

from .episode_index import EpisodeIndexResolver
from .task_name_resolver import TaskNameResolver, load_real_task_name_map


class PolicyContextResolver:
    """Prefer client identity, with instruction and generated-ID fallbacks.

    Resolution is kept per environment and sticks until ``reset``. This makes
    simulator-provided values authoritative while guaranteeing that a real
    robot request containing only an instruction still has a task and episode
    in every observation/action log entry.
    """

    def __init__(
        self,
        configured_task_name: Any,
        instruction_map_path: str | None = None,
    ) -> None:
        self._configured_task_name = configured_task_name
        self._mapping = load_real_task_name_map(instruction_map_path)
        self._tasks: dict[int, TaskNameResolver] = {}
        self._episodes: dict[int, EpisodeIndexResolver] = {}

    def resolve(
        self, observation: Mapping[str, Any], env_idx: int
    ) -> tuple[str, str]:
        env_idx = int(env_idx)
        task = self._tasks.get(env_idx)
        if task is None:
            task = TaskNameResolver(self._configured_task_name, self._mapping)
            self._tasks[env_idx] = task
        episode = self._episodes.get(env_idx)
        if episode is None:
            episode = EpisodeIndexResolver()
            self._episodes[env_idx] = episode
        try:
            task_name = task.resolve(observation)
        except (KeyError, ValueError):
            # Logging must never stop policy inference. Unknown prose remains
            # visible in the instruction field and gets an explicit sentinel.
            task_name = str(self._configured_task_name or "unknown_task")
        return task_name, episode.resolve(observation.get("episode_idx"))

    def reset(self) -> None:
        self._tasks = {}
        self._episodes = {}
