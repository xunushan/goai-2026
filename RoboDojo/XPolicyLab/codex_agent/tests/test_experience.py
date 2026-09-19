#!/usr/bin/env python3
"""Experience selection, rendering, and image injection checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from codex_agent.bridge.experience import (  # noqa: E402
    CAMERA_KEYS,
    ExperienceError,
    ExperienceLibrary,
    render_demo_text,
)
from codex_agent.bridge.bridge import BridgeState  # noqa: E402


LIBRARY = PACKAGE / "experience_library"


def test_exact_task_selection_and_compact_camera_policy() -> None:
    library = ExperienceLibrary(LIBRARY)
    assert library.items("unknown_task") == []
    items = library.items("stack_bowls")
    assert "stack_bowls" in library._cache
    assert library.items("stack_bowls") == items
    text = "\n".join(item["text"] for item in items if item["type"] == "text")
    images = [item for item in items if item["type"] == "image"]
    assert "HISTORICAL SUCCESSFUL DEMONSTRATION" in text
    assert "[EXAMPLE 1/5 | approach | frame 24]" in text
    assert "[EXAMPLE 5/5 | home | frame 366]" in text
    assert '"action"' not in text
    # approach/transport/home use cam_high; grasp/place retain all three views.
    assert len(images) == 9
    assert all(item["url"].startswith("data:image/jpeg;base64,") for item in images)
    labels = [item["text"] for item in items if item["type"] == "text" and item["text"].startswith("Historical image")]
    assert labels == [
        "Historical image 1: cam_high",
        "Historical image 2: cam_high",
        "Historical image 2: cam_left_wrist",
        "Historical image 2: cam_right_wrist",
        "Historical image 3: cam_high",
        "Historical image 4: cam_high",
        "Historical image 4: cam_left_wrist",
        "Historical image 4: cam_right_wrist",
        "Historical image 5: cam_high",
    ]
    assert library.index["stack_bowls"]["views"] == {
        "default": ("cam_high",),
        "grasp": ("cam_high", "cam_left_wrist", "cam_right_wrist"),
        "place": ("cam_high", "cam_left_wrist", "cam_right_wrist"),
    }

    demo = json.loads((LIBRARY / "stack_bowls" / "demo.json").read_text(encoding="utf-8"))
    assert len(demo["keyframes"]) == 5
    assert all("decision" in frame and "action" not in frame for frame in demo["keyframes"])
    assert sum(len(frame["images"]) for frame in demo["keyframes"]) == 15

    malformed = json.loads(json.dumps(demo))
    del malformed["keyframes"][0]["images"][CAMERA_KEYS[1]]
    try:
        render_demo_text(malformed, "stack_bowls")
    except ExperienceError:
        pass
    else:
        raise AssertionError("a missing experience camera was accepted")


def test_experience_is_added_only_when_a_thread_opens() -> None:
    state = object.__new__(BridgeState)
    state.experience_library = ExperienceLibrary(LIBRARY)
    state.workspace = PACKAGE / "workspace"
    state.episode_id = "ep-test"
    state.use_experience = True
    state.initial_context = []
    state.initial_context_loaded = False
    state.history = []
    fresh = state.thread_prefix("stack_bowls", fresh_thread=True)
    assert fresh and fresh[0]["text"].startswith("HISTORICAL SUCCESSFUL DEMONSTRATION")
    assert state.initial_context == fresh
    assert state.initial_context_loaded
    assert state.thread_prefix("stack_bowls", fresh_thread=False) == []

    state.history = [{
        "observation_text": "prior observation",
        "decision": {"phase": "transport", "note": "move bowl"},
        "image_paths": ["observations/cam_head/000001.jpg"],
        "images": [],
    }]
    rotated = state.thread_prefix("stack_bowls", fresh_thread=True)
    text = "\n".join(item.get("text", "") for item in rotated)
    assert rotated[:len(state.initial_context)] == state.initial_context
    assert "HISTORICAL SUCCESSFUL DEMONSTRATION" in text
    assert "prior observation" in text

    disabled = object.__new__(BridgeState)
    disabled.experience_library = ExperienceLibrary(LIBRARY)
    disabled.workspace = PACKAGE / "workspace"
    disabled.episode_id = "ep-no-demo"
    disabled.use_experience = False
    disabled.initial_context = []
    disabled.initial_context_loaded = False
    disabled.history = []
    assert disabled.thread_prefix("stack_bowls", fresh_thread=True) == []
    assert disabled.initial_context_loaded


def main() -> int:
    test_exact_task_selection_and_compact_camera_policy()
    test_experience_is_added_only_when_a_thread_opens()
    print("experience tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
