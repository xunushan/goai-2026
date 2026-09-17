#!/usr/bin/env python3
"""App Server context, tool, skill, image, and three-turn rotation checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from codex_agent.bridge.app_server import (  # noqa: E402
    AppServerError,
    CodexAppServer,
    discover_workspace_skills,
)
from codex_agent.bridge.bridge import (  # noqa: E402
    BridgeState,
    _data_url,
    _render_turn,
)
from codex_agent.bridge.schema import ArmObservation, ImageInput, Observation  # noqa: E402
from codex_agent.bridge.record import relative_paths, store_images  # noqa: E402

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


def test_required_skills_are_discovered_from_workspace_paths() -> None:
    with tempfile.TemporaryDirectory(prefix="policy-skills-test-") as directory:
        workspace = Path(directory)
        for name in ("agent_policy", "geometric_grounding"):
            skill = workspace / ".agents" / "skills" / name / "SKILL.md"
            skill.parent.mkdir(parents=True, exist_ok=True)
            skill.write_text(f"---\nname: {name}\ndescription: test\n---\n", encoding="utf-8")
        assert discover_workspace_skills(workspace) == {
            "agent_policy": ".agents/skills/agent_policy/SKILL.md",
            "geometric_grounding": ".agents/skills/geometric_grounding/SKILL.md",
        }


def test_observations_are_grouped_by_camera_then_step() -> None:
    with tempfile.TemporaryDirectory(prefix="policy-observations-test-") as directory:
        record_dir = Path(directory)
        images = tuple(
            ImageInput(name, "image/jpeg", b"\xff\xd8\xff\xd9")
            for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        )
        paths = store_images(
            images, record_dir=record_dir, step_id=12, request_id="ep-test-000003"
        )
        assert relative_paths(paths, record_dir) == [
            "observations/cam_head/000012_ep-test-000003.jpg",
            "observations/cam_left_wrist/000012_ep-test-000003.jpg",
            "observations/cam_right_wrist/000012_ep-test-000003.jpg",
        ]


def test_images_and_three_turn_rotation() -> None:
    server = RecordingAppServer()
    encoded_images = [
        (name, _data_url(ImageInput(name, "image/jpeg", b"\xff\xd8\xff\xd9")))
        for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
    ]
    for turn in range(4):
        replay = None
        if turn == 3:
            replay = [
                {"type": "text", "text": "historical observation"},
                {"type": "image", "url": "data:image/jpeg;base64,/9j/2Q=="},
                {"type": "text", "text": "historical structured decision"},
            ]
        server.decide(
            text=f"observation-{turn}",
            images=encoded_images,
            output_schema={"type": "object"},
            timeout_s=1,
            replay=replay,
        )

    assert [request["threadId"] for request in server.turn_requests] == [
        "thread-1", "thread-1", "thread-1", "thread-2"
    ]
    for turn, request in enumerate(server.turn_requests):
        inputs = request["input"]
        assert any(f"observation-{turn}" in item.get("text", "") for item in inputs)
        camera_labels = [
            item.get("text") for item in inputs
            if item["type"] == "text" and item.get("text", "").startswith("Camera image:")
        ]
        assert camera_labels == [
            "Camera image: cam_head",
            "Camera image: cam_left_wrist",
            "Camera image: cam_right_wrist",
        ]
        image_items = [item for item in inputs if item["type"] == "image"]
        assert len(image_items) == (4 if turn == 3 else 3)
        assert all(item["url"] == "data:image/jpeg;base64,/9j/2Q==" for item in image_items)
    assert server.turn_requests[3]["input"][0]["text"] == "historical observation"


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


def test_rollover_replays_all_text_and_only_latest_images() -> None:
    state = object.__new__(BridgeState)
    state.history = [
        {
            "observation_text": f"observation-{index}",
            "decision": {
                "left": {"position": "keep", "orientation": "keep", "gripper": "keep"},
                "right": {"position": "keep", "orientation": "keep", "gripper": "keep"},
                "note": f"evidence-{index}",
                "phase": f"phase-{index}",
            },
            "image_paths": [f"observations/cam_head/{index:06d}.jpg"],
            "images": [] if index < 7 else [
                ("cam_head", "data:image/jpeg;base64,/9j/2Q==")
            ],
        }
        for index in range(8)
    ]
    items = state.replay()
    text = json.dumps(items, ensure_ascii=False)
    for index in range(8):
        assert f"observation-{index}" in text
        assert f"evidence-{index}" in text
    assert sum(item["type"] == "image" for item in items) == 1


def test_live_app_server_tool_and_skill_isolation() -> None:
    codex_bin = Path(os.environ.get("CODEX_BIN", "/Applications/ChatGPT.app/Contents/Resources/codex"))
    assert codex_bin.is_file(), f"live App Server check requires {codex_bin}"

    server = CodexAppServer(workspace=WORKSPACE, codex_bin=str(codex_bin))
    server.bind("context-smoke-test")
    args = server._permission_args()
    assert args[args.index("shell_tool") - 1] == "--enable"
    assert args[args.index("view_image") - 1] == "--enable"
    assert args[args.index("plugins") - 1] == "--disable"
    assert 'permissions.rollout_agent.extends=":workspace"' not in args
    filesystem = next(value for value in args if value.startswith("permissions.rollout_agent.filesystem="))
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
        assert enabled == set(server.required_skills)

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


def test_timeout_interrupts_turn_and_discards_only_thread() -> None:
    server = RecordingAppServer()
    server.thread_id = "thread-1"
    server._live_image_turns = 2
    messages: list[dict] = []

    def send(message: dict) -> None:
        messages.append(message)

    server._send = send  # type: ignore[method-assign]
    try:
        CodexAppServer._wait_for_final(server, "thread-1", "turn-1", 0)
    except AppServerError as error:
        assert error.kind == "policy_timeout"
    else:
        raise AssertionError("timeout unexpectedly completed")
    assert messages == [{
        "jsonrpc": "2.0",
        "id": 1,
        "method": "turn/interrupt",
        "params": {"threadId": "thread-1", "turnId": "turn-1"},
    }]
    assert server.thread_id is None
    assert server._live_image_turns == 0


def main() -> int:
    test_each_turn_contains_the_fresh_observation()
    test_required_skills_are_discovered_from_workspace_paths()
    test_observations_are_grouped_by_camera_then_step()
    test_images_and_three_turn_rotation()
    test_failed_image_turns_still_trigger_rotation()
    test_rollover_replays_all_text_and_only_latest_images()
    test_live_app_server_tool_and_skill_isolation()
    test_timeout_interrupts_turn_and_discards_only_thread()
    print("app-server context tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
