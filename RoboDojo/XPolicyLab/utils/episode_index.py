"""Episode identity shared by simulator and real-robot policy adapters."""

from __future__ import annotations

import re
import uuid
from typing import Any


def _client_episode(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    return text or None


class EpisodeIndexResolver:
    """Use the client's episode_idx or synthesize one until the next reset."""

    def __init__(self) -> None:
        self._sequence = 0
        self._current: str | None = None

    def resolve(self, episode_idx: Any) -> str:
        client = _client_episode(episode_idx)
        if client is not None:
            self._current = client
            return client
        if self._current is None:
            self._sequence += 1
            self._current = f"ep{self._sequence:03d}_{uuid.uuid4().hex[:4]}"
        return self._current

    def reset(self) -> None:
        self._current = None
