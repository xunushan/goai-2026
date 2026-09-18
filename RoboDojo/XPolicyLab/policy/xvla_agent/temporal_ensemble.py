"""Online ACT temporal ensemble (sliding-window).

Mirrors LeRobot's ``ACTTemporalEnsembler`` and the XPolicyLab act_lerobot
adapter's ``_ServerTemporalEnsembler``, extended for chunked execution: instead
of re-planning every simulator step and consuming exactly one aligned action
per request, the policy server re-plans once per ``actions_per_chunk`` steps
and pops that many aligned ensembled actions from the window in one go.
"""
from __future__ import annotations

import numpy as np


class ServerTemporalEnsembler:
    """Online ACT temporal ensemble for one simulator environment.

    At simulator time t, the action for t combines the predictions aligned to
    t: chunk_t[0], chunk_{t-1}[1], ..., chunk_{t-k}[k], weighted by
    w_i = exp(-coefficient * i) with w_0 the oldest prediction (older actions
    are weighted more highly for a positive coefficient, matching LeRobot's
    ACTTemporalEnsembler / the original ACT implementation).

    Operates on the raw model action chunk [T, D] (numpy), before the
    rotate6d->quaternion conversion and gripper post-processing.

    Usage for chunked execution (``actions_per_chunk = K``): call
    ``update(chunk)`` once per re-plan (every K simulator steps), then ``pop()``
    K times to consume the K aligned ensembled actions. K must be smaller than
    ``chunk_size``; otherwise consecutive re-plans never overlap in time and
    the ensemble is a no-op (callers then skip this class entirely).
    """

    def __init__(self, coefficient: float, chunk_size: int):
        self.coefficient = float(coefficient)
        self.chunk_size = int(chunk_size)
        self.actions: np.ndarray | None = None
        self.counts: np.ndarray | None = None
        self.last_prediction_count = 0

    def reset(self) -> None:
        self.actions = None
        self.counts = None
        self.last_prediction_count = 0

    def update(self, chunk: np.ndarray) -> None:
        """Incorporate one fresh [chunk_size, D] prediction chunk.

        The chunk predicts actions for the current simulator step and the next
        ``chunk_size - 1`` steps. Rows still held in the window are averaged
        with their previous predictions (oldest prediction gets weight w_0);
        rows beyond the window are appended as fresh single-prediction entries.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] != self.chunk_size:
            raise ValueError(
                "Temporal ensemble expects one [T,D] chunk per environment, "
                f"got {tuple(chunk.shape)} (chunk_size={self.chunk_size})"
            )

        weights = np.exp(-self.coefficient * np.arange(self.chunk_size))
        cumulative_weights = np.cumsum(weights)

        if self.actions is None:
            self.actions = chunk.copy()
            self.counts = np.ones((self.chunk_size, 1), dtype=np.int64)
            return

        assert self.counts is not None
        previous_count = self.counts
        if previous_count.shape[0] == 0:
            # Window fully consumed since the last re-plan: start fresh.
            self.actions = chunk.copy()
            self.counts = np.ones((self.chunk_size, 1), dtype=np.int64)
            return

        # The window already holds aligned weighted averages for the next L
        # steps (it shrank by the actions consumed since the previous re-plan).
        # The new chunk's first L rows align with those steps; the remaining
        # rows predict steps the window no longer covers and are appended.
        L = self.actions.shape[0]
        self.actions *= cumulative_weights[previous_count - 1]
        self.actions += chunk[:L] * weights[previous_count]
        self.actions /= cumulative_weights[previous_count]
        self.counts = np.clip(previous_count + 1, 1, self.chunk_size)
        if L < self.chunk_size:
            self.actions = np.concatenate((self.actions, chunk[L:]), axis=0)
            self.counts = np.concatenate(
                (self.counts, np.ones((self.chunk_size - L, 1), dtype=np.int64)),
                axis=0,
            )

    def pop(self) -> np.ndarray:
        """Return the ensembled action for the current simulator step.

        Shifts the window forward by one step. The returned action is the
        weighted average of every prediction aligned to this step. Raises if
        the window is empty (an ``update`` must precede pops).
        """
        if self.actions is None or self.actions.shape[0] == 0:
            raise RuntimeError(
                "Temporal ensemble window is empty: call update() before pop()"
            )
        assert self.counts is not None
        self.last_prediction_count = int(self.counts[0, 0])
        action = self.actions[0]
        self.actions = self.actions[1:]
        self.counts = self.counts[1:]
        return action
