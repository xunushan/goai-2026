"""Resolve a real-robot instruction to the task name used by policy task cards."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

DEFAULT_REAL_TASK_INSTRUCTIONS = (
    Path(__file__).resolve().parents[3] / "configs" / "real_task_instruction.json"
)
TASK_NAME_KEYS = ("task_name",)
INSTRUCTION_KEYS = ("prompt", "instruction", "task", "language_instruction")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    elif isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)):
        for item in value:
            result = _text(item)
            if result:
                return result
        return None
    value = str(value).strip()
    return value or None


def _key(value: Any) -> str:
    value = _text(value)
    return re.sub(r"\s+", " ", value).casefold() if value else ""


def load_real_task_name_map(path: str | Path | None = None) -> dict[str, str]:
    """Load only the platform contract: original_instruction -> task_name."""
    source = Path(path or DEFAULT_REAL_TASK_INSTRUCTIONS).expanduser().resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    entries = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{source} must contain a tasks array")
    mapping: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"{source} tasks entries must be objects")
        instruction = _key(entry.get("original_instruction"))
        task_name = str(entry.get("task_name") or "").strip()
        if not instruction or not task_name:
            raise ValueError(f"{source} task is missing original_instruction or task_name")
        if instruction in mapping and mapping[instruction] != task_name:
            raise ValueError(f"{source} maps one instruction to multiple tasks")
        mapping[instruction] = task_name
    return mapping


class TaskNameResolver:
    """Use a configured task, a task the client names, or a mapped instruction.

    Precedence is configured, then the observation's own ``task_name``, then the
    instruction mapping. A client that knows which task it is running -- the
    simulator does -- names the task directly, and that beats guessing from
    prose: two tasks can share one instruction template, and the mapping table
    only covers the platform contract anyway. The instruction path stays for
    prose-only clients.
    """

    def __init__(
        self,
        configured_task_name: Any,
        mapping: Mapping[str, str],
    ) -> None:
        configured = str(configured_task_name or "").strip()
        self._configured = configured or None
        self._mapping = dict(mapping)
        self._episode_task_name: str | None = self._configured

    def reset(self) -> None:
        self._episode_task_name = self._configured

    def resolve(self, observation: Mapping[str, Any]) -> str:
        if self._episode_task_name:
            return self._episode_task_name
        for field in TASK_NAME_KEYS:
            explicit = _text(observation.get(field))
            if explicit:
                self._episode_task_name = explicit
                return explicit
        for field in INSTRUCTION_KEYS:
            raw = observation.get(field)
            key = _key(raw)
            if not key:
                continue
            task_name = self._mapping.get(key)
            if task_name is None:
                raise KeyError(f"instruction does not map to a real task: {_text(raw)!r}")
            self._episode_task_name = task_name
            return task_name
        raise ValueError("task_name is null and the observation contains no instruction")
