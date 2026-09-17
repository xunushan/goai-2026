#!/usr/bin/env python3
"""App Server context, tool, skill, image, and three-turn rotation checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from codex_agent.bridge.app_server import (  # noqa: E402
    AppServerError,
    REQUIRED_SKILLS,
    CodexAppServer,
)
from codex_agent.bridge.bridge import (  # noqa: E402
    BridgeState,
    ROLLOVER_TURNS,
    _data_url,
    _render_turn,
)
from codex_agent.bridge.schema import ArmObservation, ImageInput, Observation  # noqa: E402

WORKSPACE = PACKAGE / "workspace"


class RecordingAppServer(CodexAppServer):
    """Exercise turn construction and rotation without calling a model."""

    def __init__(self) -> None:
        super().__init__(workspace=WORKSPACE, max_live_image_turns=3)
        self.turn_requests: list[dict] = []
        self.thread_count = 0

    def start_thread(self) -> str:
        if self.thread_id is None:
            self.thread_count += 1
            self.thread_id = f"thread-{self.thread_count}"
        return self.thread_id

    def _request(self, method: str, params: dict) -> dict:
        assert method == "turn/start"
        self.turn_requests.append(params)
        return {"turn": {"id": f"turn-{len(self.turn_requests)}"}}

    def _wait_for_final(self, thread_id: str, turn_id: str, timeout_s: float):
        return "{}", {}


class FailingWaitAppServer(RecordingAppServer):
    def _wait_for_final(self, thread_id: str, turn_id: str, timeout_s: float):
        raise AppServerError("policy_timeout", "test timeout")


def observation(turn: int) -> Observation:
    left = ArmObservation((float(turn), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 1.0)
    right = ArmObservation((0.0, float(turn), 0.0), (1.0, 0.0, 0.0, 0.0), 0.5)
    return Observation(
        episode_id="ep-test",
        request_id=f"request-{turn}",
        step_id=turn,
        turn_index=turn,
        task={"instruction": "test instruction"},
        budget={
            "max_decisions": 10,
            "max_sim_steps": 100,
            "remaining_decisions": 9 - turn,
            "remaining_steps": 99 - turn,
        },
        arms={"left": left, "right": right},
        feedback=(f"feedback-{turn}",),
        images=(ImageInput("cam_head", "image/jpeg", b"jpeg"),),
    )


def test_each_turn_contains_the_fresh_observation() -> None:
    first = _render_turn(observation(1))
    second = _render_turn(observation(2))
    assert "left: position [1.0000, 0.0000, 0.0000]" in first
    assert "left: position [2.0000, 0.0000, 0.0000]" in second
    assert "feedback-1" in first and "feedback-2" in second
    assert "Scene:" not in first and "Success when:" not in first


def test_images_and_three_turn_rotation() -> None:
    server = RecordingAppServer()
    encoded_images = [
        (name, _data_url(ImageInput(name, "image/jpeg", b"\xff\xd8\xff\xd9")))
        for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
    ]
    for turn in range(4):
        server.decide(
            text=f"observation-{turn}",
            images=encoded_images,
            output_schema={"type": "object"},
            timeout_s=1,
            rollover_context="three-step-summary" if turn == 3 else None,
        )

    assert [request["threadId"] for request in server.turn_requests] == [
        "thread-1", "thread-1", "thread-1", "thread-2"
    ]
    for turn, request in enumerate(server.turn_requests):
        inputs = request["input"]
        assert f"observation-{turn}" in inputs[0]["text"]
        assert [item.get("text") for item in inputs if item["type"] == "text"][1:] == [
            "Camera image: cam_head",
            "Camera image: cam_left_wrist",
            "Camera image: cam_right_wrist",
        ]
        image_items = [item for item in inputs if item["type"] == "image"]
        assert len(image_items) == 3
        assert all(item["url"] == "data:image/jpeg;base64,/9j/2Q==" for item in image_items)
    assert server.turn_requests[3]["input"][0]["text"].startswith("three-step-summary\n\n")


def test_failed_image_turns_still_trigger_rotation() -> None:
    server = FailingWaitAppServer()
    for turn in range(4):
        try:
            server.decide(
                text=f"observation-{turn}",
                images=[("cam_head", "data:image/jpeg;base64,/9j/2Q==")],
                output_schema={"type": "object"},
                timeout_s=1,
            )
        except AppServerError as error:
            assert error.kind == "policy_timeout"
        else:
            raise AssertionError("the failing test server unexpectedly completed")
    assert [request["threadId"] for request in server.turn_requests] == [
        "thread-1", "thread-1", "thread-1", "thread-2"
    ]


def test_rollover_replays_exactly_three_earlier_decisions() -> None:
    assert ROLLOVER_TURNS == 3
    state = object.__new__(BridgeState)
    state.history = [
        {"turn_index": index, "phase": f"phase-{index}", "note": f"note-{index}"}
        for index in range(8)
    ]
    text = state.rollover_context()
    assert text is not None
    assert "turn 6 [phase-5] note-5" in text
    assert "turn 7 [phase-6] note-6" in text
    assert "turn 8 [phase-7] note-7" in text
    assert "turn 5 " not in text


def test_live_app_server_tool_and_skill_isolation() -> None:
    codex_bin = Path(os.environ.get("CODEX_BIN", "/Applications/ChatGPT.app/Contents/Resources/codex"))
    assert codex_bin.is_file(), f"live App Server check requires {codex_bin}"

    server = CodexAppServer(workspace=WORKSPACE, codex_bin=str(codex_bin))
    server.bind("context-smoke-test")
    args = server._permission_args()
    assert args[args.index("shell_tool") - 1] == "--enable"
    assert args[args.index("view_image") - 1] == "--enable"
    assert args[args.index("plugins") - 1] == "--disable"
    filesystem = args[args.index("permissions.rollout_agent.extends=\":workspace\"") + 2]
    assert f'{json.dumps(str(WORKSPACE / "output"))} = "none"' in filesystem
    observations = WORKSPACE / "output" / "context-smoke-test" / "observations"
    assert f'{json.dumps(str(observations))} = "read"' in filesystem
    try:
        server.start()
        response = server._request(
            "skills/list", {"cwds": [str(WORKSPACE)], "forceReload": True}
        )
        enabled = {
            skill["name"]
            for entry in response["data"]
            for skill in entry["skills"]
            if skill["enabled"]
        }
        assert enabled == set(REQUIRED_SKILLS)

        runtime_home = Path(server._runtime_home.name)
        environment = os.environ.copy()
        environment["HOME"] = str(runtime_home)
        environment["CODEX_HOME"] = str(runtime_home / ".codex")
        rendered = subprocess.run(
            [str(codex_bin), "debug", "prompt-input", *server._permission_args(), "probe"],
            cwd=WORKSPACE,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        prompt_items = json.loads(rendered.stdout)
        prompt_text = json.dumps(prompt_items, ensure_ascii=False)
        assert "codex_agent" in prompt_text
        for unrelated in (
            "openai-docs", "plugin-creator", "skill-creator", "skill-installer",
            "agent-browser", "recommended_plugins", "Apple Music",
        ):
            assert unrelated not in prompt_text
    finally:
        server.close()


def main() -> int:
    test_each_turn_contains_the_fresh_observation()
    test_images_and_three_turn_rotation()
    test_failed_image_turns_still_trigger_rotation()
    test_rollover_replays_exactly_three_earlier_decisions()
    test_live_app_server_tool_and_skill_isolation()
    print("app-server context tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
