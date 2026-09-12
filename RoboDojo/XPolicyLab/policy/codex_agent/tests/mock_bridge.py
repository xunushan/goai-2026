"""A stand-in for the Codex bridge, so the adapter can be tested with no Codex.

``tests/fake_codex`` tests the *real* bridge (``bridge/server.py``) by faking the
CLI underneath it. This module goes one level up: it replaces the bridge itself,
so ``model.py`` can be exercised end to end over real HTTP without any bridge,
CLI or network dependency at all. That distinction matters because the operator
has asked that the Codex allowance be spent only on the official run.

Every mode answers on the same ``/v1/decide`` and ``/healthz`` endpoints and
records what it received, so a test can assert on the requests as well as the
responses.

    with MockBridge("legal") as bridge:
        model = make_model(bridge.url)
        chunk = model.get_action()
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# Modes that answer with a well-formed decision.
DECISION_MODES = ("legal", "all_keep", "out_of_box", "far_away", "absurd_quat")

# Modes that fail, with the error_kind the adapter should end up reporting.
FAILURE_MODES = {
    "unparseable": "codex_unparseable",
    "http_500": "bridge_http_error",
    "garbage_body": "bridge_bad_response",
    "empty": "codex_error",
    "hung": "bridge_unreachable",
    "timeout_status": "codex_timeout",
}

DEFAULT_TARGET = [-0.15, -0.25, 1.05]
DEFAULT_THREAD_ID = "thr_mock_0001"


def legal_reply(position=None) -> dict[str, Any]:
    return {
        "left": {"position": list(position or DEFAULT_TARGET), "orientation": "keep", "gripper": "keep"},
        "right": {"position": "keep", "orientation": "keep", "gripper": "keep"},
        "note": "moving the left arm towards the socket",
        "phase": "approach",
    }


class _QuietServer(ThreadingHTTPServer):
    """A server that does not shout when a client gives up on a slow handler.

    The ``hung`` mode deliberately answers after the client has already timed
    out, which leaves the handler writing to a closed socket. That is the point
    of the test, so the traceback is noise.
    """

    def handle_error(self, request, client_address) -> None:
        return


class MockBridge:
    """A scriptable bridge. ``mode`` may be reassigned between calls."""

    def __init__(self, mode: str = "legal", *, delay_s: float = 0.0) -> None:
        self.mode = mode
        self.delay_s = float(delay_s)
        self.healthz_count = 0
        self.decide_payloads: list[dict[str, Any]] = []
        self.thread_counter = 0
        self._lock = threading.Lock()

        handler = type("Handler", (self._handler_base(),), {"mock": self})
        self.server = _QuietServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def decide_count(self) -> int:
        with self._lock:
            return len(self.decide_payloads)

    def last_payload(self) -> dict[str, Any]:
        with self._lock:
            return self.decide_payloads[-1]

    # ------------------------------------------------------------------ #
    def _next_thread_id(self) -> str:
        self.thread_counter += 1
        return f"{DEFAULT_THREAD_ID}_{self.thread_counter}"

    def _decide_response(self, payload: dict[str, Any]) -> tuple[int, Any, bool]:
        """Return ``(status, body, is_raw_text)``.

        ``hung`` answers correctly but far too late -- the client gives up first,
        which is exactly how a real Codex overrun looks from the policy server.
        ``timeout_status`` answers the way the real bridge does once it has
        killed the Codex subprocess.
        """
        thread_id = payload.get("thread_id") or self._next_thread_id()
        mode = self.mode

        if self.delay_s > 0:
            time.sleep(self.delay_s)

        if mode == "hung":
            return 200, {"ok": True, "thread_id": thread_id, "parsed": legal_reply()}, False
        if mode == "timeout_status":
            return (
                502,
                {
                    "ok": False,
                    "error_kind": "codex_timeout",
                    "error": "codex did not finish within 75s",
                    "timed_out": True,
                    "thread_id": thread_id,
                    "latency_ms": 75000,
                },
                False,
            )
        if mode == "http_500":
            return 500, {"ok": False, "error": "the bridge exploded"}, False
        if mode == "garbage_body":
            return 200, "<html>not json at all</html>", True
        if mode == "empty":
            return 200, {"ok": False, "thread_id": thread_id}, False
        if mode == "unparseable":
            return (
                502,
                {
                    "ok": False,
                    "error_kind": "codex_unparseable",
                    "error": "codex replied with text that is not a JSON object",
                    "text": "I would rather describe the scene in prose.",
                    "thread_id": thread_id,
                    "latency_ms": 24000,
                },
                False,
            )

        if mode == "all_keep":
            parsed = {
                "left": "keep",
                "right": "keep",
                "note": "waiting and watching",
                "phase": "observe",
            }
        elif mode == "out_of_box":
            # 0.30 m from HOME, so the per-decision distance limit does not fire:
            # this exercises the workspace box specifically (x lo is -0.50).
            parsed = legal_reply([-0.58, -0.30, 1.00])
        elif mode == "far_away":
            # 0.98 m from HOME: exercises the per-decision distance limit.
            parsed = legal_reply([-0.10, 0.60, 1.05])
        elif mode == "absurd_quat":
            # A real target with an unusable orientation: the parse reports the
            # problem and falls back to keep, but the motion still happens, so
            # the adapter logs the problem rather than swallowing it in a no-op.
            parsed = {
                "left": {
                    "position": list(DEFAULT_TARGET),
                    "orientation": [0.0, 0.0, 0.0, 0.0],
                    "gripper": "keep",
                },
                "right": "keep",
                "note": "rotating",
                "phase": "orient",
            }
        else:
            parsed = legal_reply()

        return (
            200,
            {
                "ok": True,
                "thread_id": thread_id,
                "parsed": parsed,
                "text": json.dumps(parsed),
                "usage": {"output_tokens": 120, "cached_input_tokens": 900},
                "latency_ms": 24000,
                "runs_dir": "/tmp/mock_bridge_runs",
                "error": "",
                "error_kind": "",
                "timed_out": False,
            },
            False,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _handler_base():
        class Handler(BaseHTTPRequestHandler):
            mock: MockBridge
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
                return

            def _send(self, status: int, body: Any, raw: bool = False) -> None:
                data = body.encode("utf-8") if raw else json.dumps(body).encode("utf-8")
                if raw:
                    self.send_response(status)
                    self.send_header("Content-Type", "text/plain")
                else:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802 - stdlib signature
                if self.path.split("?")[0] not in ("/healthz", "/"):
                    self._send(404, {"ok": False, "error": "not found"})
                    return
                with self.mock._lock:
                    self.mock.healthz_count += 1
                self._send(200, {"ok": True, "model": "gpt-6-astra", "stats": {}})

            def do_POST(self):  # noqa: N802 - stdlib signature
                if self.path.split("?")[0] != "/v1/decide":
                    self._send(404, {"ok": False, "error": "not found"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    self._send(400, {"ok": False, "error": "bad body"})
                    return
                if not isinstance(payload, dict):
                    self._send(400, {"ok": False, "error": "body must be an object"})
                    return
                with self.mock._lock:
                    self.mock.decide_payloads.append(payload)
                status, body, raw = self.mock._decide_response(payload)
                self._send(status, body, raw=raw)

        return Handler

    # ------------------------------------------------------------------ #
    def __enter__(self) -> MockBridge:
        self.thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def wait_until_responsive(self, timeout_s: float = 5.0) -> None:
        """healthz-poll so a test never races the server thread's startup."""
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{self.url}/healthz", timeout=1.0):
                    # The readiness poll is this test harness talking to itself,
                    # not the adapter; do not let it show up in the counts.
                    self.healthz_count = 0
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(0.02)
        raise RuntimeError("the mock bridge never came up")


def unused_port() -> int:
    """A port that was open a moment ago and now is not: connection refused."""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
