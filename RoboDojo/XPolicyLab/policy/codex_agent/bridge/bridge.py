"""The policy service: one HTTP endpoint in front of one persistent Codex session.

Runs on the operator's machine, where the Codex CLI and its credentials live, and
is reached by the RoboDojo policy server through an SSH reverse tunnel::

    Mac                                    issac-server
    codex_agent/bridge :8765  <-- ssh -N -R 8765:localhost:8765 --  policy server
       +-- codex app-server --stdio (one process, cwd = workspace/)

The division of labour is the point of this whole layout. The GPU-side adapter
owns the simulator-side work: it reads the observation, interpolates the model's
target into an action chunk and accounts for the step budget. It
sends a structured observation -- arm state, task, budget, feedback, images -- and
this service turns that into one turn of text, hands the images through unchanged,
and writes down what happened. No policy text lives here; the embodiment contract
is in ``workspace/AGENTS.md`` and the decision procedure in
``workspace/.agents/skills/codex_agent/SKILL.md``, both read by Codex itself.

Three constraints shape everything below:

* **Always answer.** RoboDojo has a hard timeout and discards a whole episode when
  it expires, so every path here returns an HTTP response -- including the ones
  where Codex hung, died, or said something that was not JSON.
* **Bound to the loopback interface.** The tunnel is the only way in; a shared
  token is optional extra.
* **Everything is written down.** ``workspace/output/<episode_id>/`` keeps the
  images at the resolution they arrived in and a line per turn, so a run can be
  audited afterwards rather than argued about.

Usage::

    python -m bridge.bridge --port 8765 --model gpt-6-astra
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

from . import app_server as app_server_module
from . import record
from .app_server import AppServerError, CodexAppServer
from .schema import Observation, PolicyValidationError, output_schema, validate_response

MAX_BODY_BYTES = 32 * 1024 * 1024

# Where the workspace sits relative to this package, i.e. ``codex_agent/workspace``.
DEFAULT_WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"

# How many earlier decisions are replayed as words when a thread is rotated.
# The rotation itself is the App Server's business; this is how much of the story
# survives it.
ROLLOVER_TURNS = 3
CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _vec(values: Sequence[float], digits: int = 4) -> str:
    limit = 0.5 * 10.0**-digits
    return "[" + ", ".join(
        f"{0.0 if abs(float(value)) < limit else float(value):.{digits}f}"
        for value in values
    ) + "]"


def _render_turn(observation: Observation) -> str:
    task = observation.task
    budget = observation.budget
    task_lines = [
        "TASK",
        f"Instruction: {task.get('instruction', '')}",
        f"Scene: {task.get('scene', '')}",
        f"Success when: {task.get('success_rule', '')}",
    ]
    task_lines.extend(f"- {hint}" for hint in task.get("hints") or [])
    state_lines = ["OBSERVED ARM STATE (absolute quaternion wxyz)"]
    for side in ("left", "right"):
        arm = observation.arms[side]
        state_lines.append(
            f"  {side}: position {_vec(arm.position)}; "
            f"orientation {_vec(arm.orientation)}; gripper {arm.gripper:.2f}"
        )
    feedback = observation.feedback or ["(first decision; nothing has executed yet)"]
    views = ", ".join(
        f"{index}. {image.name}" for index, image in enumerate(observation.images, 1)
    )
    return "\n\n".join([
        f"TURN {observation.turn_index + 1}\n"
        f"ATTACHED VIEWS: {views}\n"
        f"remaining decisions after this call: {budget['remaining_decisions']}\n"
        f"remaining simulator steps: {budget['remaining_steps']}",
        "\n".join(task_lines),
        "\n".join(state_lines),
        "OUTCOME OF PREVIOUS DECISION\n  " + "\n  ".join(feedback),
        "Choose the single next motion. Reply with the JSON object only.",
    ])


class BridgeError(Exception):
    """A request we will not act on, with ``kind`` naming why."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class BridgeState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.workspace = Path(args.workspace).expanduser().resolve()
        if not (self.workspace / "AGENTS.md").is_file():
            raise BridgeError("bad_request", f"{self.workspace} has no AGENTS.md, so it is not a Codex workspace")
        self.expected_cameras = CAMERA_NAMES
        self.server = CodexAppServer(
            workspace=self.workspace,
            codex_bin=args.codex_bin,
            model=args.model,
            effort=args.reasoning_effort,
            max_live_image_turns=args.max_live_image_turns,
        )
        # One active rollout and one active turn: the App Server answers one turn
        # at a time, and two overlapping requests would interleave their events.
        self.lock = threading.Lock()
        self.episode_id: str | None = None
        self.history: list[dict[str, Any]] = []

    def begin_episode(self, episode_id: str) -> None:
        """Forget the previous episode's history when a new one starts.

        Decisions are only meaningful within their own episode -- the arm state
        they refer to is gone -- and replaying them into the next thread would
        describe a scene that no longer exists.
        """
        if episode_id == self.episode_id:
            return
        self.episode_id = episode_id
        self.history = []

    def rollover_context(self) -> str | None:
        """The earlier decisions, as words, for a thread that is about to rotate.

        The most recent decision is left out: the turn text it is part of already
        carries its outcome in full, and repeating it would read as two different
        accounts of the same event.
        """
        earlier = self.history[:-1][-ROLLOVER_TURNS:]
        if not earlier:
            return None
        lines = ["EARLIER DECISIONS (their images are no longer attached)"]
        lines.extend(f"  turn {entry['turn_index'] + 1} [{entry['phase']}] {entry['note']}" for entry in earlier)
        return "\n".join(lines)

    def close(self) -> None:
        self.server.close()


def _data_url(image: Any) -> str:
    return f"data:{image.mime};base64,{base64.b64encode(image.data).decode('ascii')}"


def _check_images(observation: Observation, expected: tuple[str, ...]) -> None:
    """Require the fixed camera set and order used by the observation protocol."""
    names = tuple(image.name for image in observation.images)
    if names != expected:
        raise PolicyValidationError(
            f"the request attached views {list(names)}; expected {list(expected)}"
        )


def _parse_reply(text: str) -> dict[str, Any]:
    """The model's answer, as an object, or a Codex-side failure.

    A reply that does not parse is not the caller's fault -- the request was fine
    and the App Server was asked to constrain the answer -- so it is reported the
    same way a broken turn is, not as a bad request.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AppServerError("policy_invalid_response", f"the reply is not JSON: {exc}") from exc
    try:
        return validate_response(value)
    except PolicyValidationError as exc:
        raise AppServerError("policy_invalid_response", str(exc)) from exc


def _decide(state: BridgeState, payload: Any, started: float) -> dict[str, Any]:
    """Validate, record, ask, validate again, record.

    Raises ``PolicyValidationError`` for a request we will not act on, and
    ``AppServerError`` for anything that went wrong behind us -- with the failure
    written to the log before it is raised, so a failed turn leaves the same
    evidence a successful one does.
    """
    observation = Observation.parse(payload)
    _check_images(observation, state.expected_cameras)

    with state.lock:
        state.begin_episode(observation.episode_id)
        record_dir = record.prepare(state.workspace, observation.episode_id)
        image_paths = record.relative_paths(
            record.store_images(
                observation.images,
                record_dir=record_dir,
                step_id=observation.step_id,
                request_id=observation.request_id,
            ),
            record_dir,
        )

        def log(**fields: Any) -> None:
            record.append(
                record_dir,
                record.turn_record(
                    episode_id=observation.episode_id,
                    request_id=observation.request_id,
                    step_id=observation.step_id,
                    turn_index=observation.turn_index,
                    image_paths=image_paths,
                    **fields,
                ),
            )

        state.server.bind(observation.episode_id)
        rotating = state.server.rotation_due
        # A decision that opens a thread also has to read the workspace and the
        # skill, which is why it gets its own, longer, wall clock.
        fresh_thread = state.server.thread_id is None or rotating
        timeout_s = state.args.timeout_first_turn_s if fresh_thread else state.args.timeout_s

        try:
            text, usage = state.server.decide(
                text=_render_turn(observation),
                images=[(image.name, _data_url(image)) for image in observation.images],
                output_schema=output_schema(),
                timeout_s=timeout_s,
                rollover_context=state.rollover_context() if rotating else None,
            )
            decision = _parse_reply(text)
        except AppServerError as exc:
            log(ok=False, error_kind=exc.kind, error=str(exc), latency_ms=int((time.monotonic() - started) * 1000))
            raise

        latency_ms = int((time.monotonic() - started) * 1000)
        log(
            ok=True,
            decision=decision,
            note=decision["note"],
            phase=decision["phase"],
            usage=usage,
            latency_ms=latency_ms,
        )
        state.history.append(
            {"turn_index": observation.turn_index, "note": decision["note"], "phase": decision["phase"]}
        )
        return {"ok": True, "decision": decision}


# HTTP status per failure kind. Anything Codex-side that is not a timeout is a bad
# gateway: the request was fine, the thing behind us was not.
_STATUS = {
    "policy_timeout": 504,
    "policy_overloaded": 502,
    "policy_usage_exhausted": 502,
    "policy_turn_failed": 502,
    "policy_invalid_response": 502,
    "app_server_closed": 502,
    "app_server_start_failed": 502,
    "app_server_not_bound": 500,
    "app_server_protocol": 502,
    "app_server_rpc": 502,
}


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = "codex-bridge/2.0"
    state: BridgeState  # set by main()

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        if self.state.args.quiet:
            return
        print(f"[bridge] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorised(self) -> bool:
        token = self.state.args.token
        if not token:
            return True
        return self.headers.get("X-Bridge-Token") == token

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        if self.path.split("?")[0] not in ("/healthz", "/"):
            self._send(404, {"ok": False, "error": "not found"})
            return
        if not self._authorised():
            self._send(401, {"ok": False, "error": "bad token"})
            return
        state = self.state
        self._send(
            200,
            {
                "ok": True,
                "model": state.server.model,
                # Reported so a run can be audited after the fact: the effort level
                # changes per-call latency and is set by a startup flag, not by
                # anything the policy server sends.
                "reasoning_effort": state.server.effort,
                "codex_bin": state.server.codex_bin,
                "workspace": str(state.workspace),
                "episode_id": state.episode_id,
                "cameras": list(state.expected_cameras),
                "max_live_image_turns": state.server.max_live_image_turns,
                "timeout_s": state.args.timeout_s,
                "timeout_first_turn_s": state.args.timeout_first_turn_s,
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        if self.path.split("?")[0] != "/v1/decide":
            self._send(404, {"ok": False, "error": "not found"})
            return
        if not self._authorised():
            self._send(401, {"ok": False, "error": "bad token"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send(400, {"ok": False, "error": "bad Content-Length"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(413, {"ok": False, "error": f"body must be 1..{MAX_BODY_BYTES} bytes"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send(400, {"ok": False, "error": f"invalid JSON body: {exc}"})
            return
        if not isinstance(payload, dict):
            self._send(400, {"ok": False, "error": "body must be a JSON object"})
            return

        started = time.monotonic()
        try:
            response = _decide(self.state, payload, started)
        except (PolicyValidationError, BridgeError) as exc:
            self._send(400, {"ok": False, "error_kind": getattr(exc, "kind", "bad_request"), "error": str(exc)})
            return
        except AppServerError as exc:
            self._send(
                _STATUS.get(exc.kind, 502),
                {
                    "ok": False,
                    "error_kind": exc.kind,
                    "error": str(exc),
                },
            )
            return
        except BaseException as exc:  # noqa: BLE001 - the episode must not die on our bug
            self._send(
                500,
                {
                    "ok": False,
                    "error_kind": "bridge_internal_error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return

        self._send(200, response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local policy service for the codex_agent policy.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default=None, help="Model id, e.g. gpt-6-astra")
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--timeout-s", type=float, default=75.0, help="wall clock for one decision")
    parser.add_argument(
        "--timeout-first-turn-s",
        type=float,
        default=90.0,
        help="wall clock for a decision that opens a thread, which also has to read the workspace",
    )
    parser.add_argument(
        "--max-live-image-turns",
        type=int,
        default=app_server_module.DEFAULT_MAX_LIVE_IMAGE_TURNS,
        help="image-bearing turns a thread keeps before it is rotated; 0 disables rotation",
    )
    parser.add_argument("--token", default=os.environ.get("CODEX_BRIDGE_TOKEN"))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        state = BridgeState(args)
    except BridgeError as exc:
        print(f"[bridge] {exc}", flush=True)
        return 2
    BridgeHandler.state = state

    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    server.daemon_threads = True
    print(
        f"[bridge] listening on http://{args.host}:{args.port}  "
        f"codex={state.server.codex_bin}  model={state.server.model}  "
        f"effort={state.server.effort}  workspace={state.workspace}  "
        f"cameras={','.join(state.expected_cameras)}  "
        f"timeout={args.timeout_s}s  rotate_every={args.max_live_image_turns}",
        flush=True,
    )
    print(
        "[bridge] expose it to the remote host with:\n"
        f"           ssh -N -R {args.port}:localhost:{args.port} <remote-host>",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] shutting down", flush=True)
    finally:
        server.server_close()
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
