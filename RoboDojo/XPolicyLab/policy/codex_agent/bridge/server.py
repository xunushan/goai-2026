"""Local Codex bridge.

Runs on the operator's macOS machine (where the Codex CLI and its credentials
live) and exposes one HTTP endpoint that the remote RoboDojo policy server
reaches through an SSH reverse tunnel::

    Mac                                   issac-server
    codex_bridge :8765  <-- ssh -R 8765:localhost:8765 --  codex_agent policy server
       +-- codex exec / codex exec resume

Design constraints:

* **Always answer.** Codex may hang; the policy server has a 120 s hard timeout
  after which RoboDojo silently discards the whole episode. Every path here
  returns an HTTP response, including the timeout path.
* **Bound 127.0.0.1.** The tunnel is the only way in; a token is optional extra.
* **Everything is written down.** ``runs/<episode>/turn_NNN/`` keeps the prompt,
  the images and the raw event stream for audit and replay.

Usage::

    python bridge/server.py --config bridge/config.json
    python bridge/server.py --port 8765 --model gpt-6-astra
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from codex_runner import CodexRunner, extract_json_object  # type: ignore[import-not-found]

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 12 * 1024 * 1024
MAX_BODY_BYTES = 32 * 1024 * 1024
DEFAULT_RUNS_DIR = "~/codex_bridge/runs"
DEFAULT_WORKDIR = "~/codex_bridge/work"

_MAGIC = ((b"\xff\xd8\xff", "jpeg"), (b"\x89PNG\r\n\x1a\n", "png"))


def detect_magic(payload: bytes) -> str | None:
    """Sniff image magic bytes so a truncated or mislabelled upload is rejected."""
    for magic, name in _MAGIC:
        if payload.startswith(magic):
            return name
    return None


class BridgeError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _decode_images(images: Any, target_dir: Path) -> list[Path]:
    """Validate and persist the attached views, returning their on-disk paths."""
    if not images:
        return []
    if not isinstance(images, list):
        raise BridgeError("bad_request", "images must be a list")
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    total = 0
    for index, entry in enumerate(images):
        if not isinstance(entry, dict):
            raise BridgeError("bad_request", f"images[{index}] must be an object")
        name = str(entry.get("name") or f"view_{index}")
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:64]
        try:
            payload = base64.b64decode(str(entry["b64"]), validate=True)
        except (KeyError, binascii.Error, TypeError) as exc:
            raise BridgeError("bad_request", f"images[{index}] is not valid base64: {exc}") from exc
        if len(payload) > MAX_IMAGE_BYTES:
            raise BridgeError(
                "bad_request",
                f"images[{index}] is {len(payload)} bytes, over the {MAX_IMAGE_BYTES} limit",
            )
        total += len(payload)
        if total > MAX_TOTAL_IMAGE_BYTES:
            raise BridgeError("bad_request", f"images total more than {MAX_TOTAL_IMAGE_BYTES} bytes")
        kind = detect_magic(payload)
        if kind is None:
            raise BridgeError(
                "bad_request", f"images[{index}] is neither JPEG nor PNG (truncated payload?)"
            )
        path = target_dir / f"{index:02d}_{safe}.{'jpg' if kind == 'jpeg' else 'png'}"
        path.write_bytes(payload)
        paths.append(path)
    return paths


class BridgeState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runs_dir = Path(args.runs_dir).expanduser()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.runner = CodexRunner(
            binary=args.codex_bin,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            workdir=Path(args.workdir).expanduser(),
            timeout_s=args.timeout_s,
            timeout_first_turn_s=args.timeout_first_turn_s,
            sandbox=args.sandbox,
        )
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "ok": 0, "timeouts": 0, "errors": 0, "turns": 0}

    def bump(self, key: str) -> None:
        with self.lock:
            self.stats[key] = self.stats.get(key, 0) + 1


def _decide(state: BridgeState, payload: dict[str, Any]) -> dict[str, Any]:
    episode_id = str(payload.get("episode_id") or "episode").strip() or "episode"
    safe_episode = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in episode_id)[:80]
    turn_index = int(payload.get("turn_index") or 0)
    thread_id = payload.get("thread_id") or None
    prompt = str(payload.get("prompt") or "")
    if not prompt.strip():
        raise BridgeError("bad_request", "prompt is empty")
    schema = payload.get("schema")

    turn_dir = state.runs_dir / safe_episode / f"turn_{turn_index:03d}"
    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    schema_path: Path | None = None
    if isinstance(schema, dict):
        schema_path = turn_dir / "schema.json"
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    image_paths = _decode_images(payload.get("images"), turn_dir / "images")
    last_message_path = turn_dir / "last_message.txt"
    if last_message_path.exists():
        last_message_path.unlink()

    started = time.monotonic()
    result = state.runner.run(
        prompt=prompt,
        image_paths=image_paths,
        thread_id=thread_id,
        schema_path=schema_path,
        last_message_path=last_message_path,
    )

    # A thread that vanished (bridge restarted, Codex pruned the rollout) is
    # recoverable: replay this same turn as the first turn of a fresh thread.
    if result.thread_lost:
        thread_id = None
        result = state.runner.run(
            prompt=prompt,
            image_paths=image_paths,
            thread_id=None,
            schema_path=schema_path,
            last_message_path=last_message_path,
        )

    (turn_dir / "stdout.jsonl").write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        (turn_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")

    parsed = extract_json_object(result.text) if result.ok else None
    response: dict[str, Any] = {
        "ok": bool(result.ok and parsed is not None),
        "thread_id": result.thread_id,
        "parsed": parsed,
        "text": result.text,
        "usage": result.usage,
        "latency_ms": int(result.duration_s * 1000),
        "runs_dir": str(turn_dir),
        "error": result.error or "",
        "error_kind": result.error_kind or "",
        "timed_out": result.timed_out,
        "events": result.events,
    }
    if result.ok and parsed is None:
        response["error_kind"] = "codex_unparseable"
        response["error"] = "codex replied with text that is not a JSON object"

    (turn_dir / "result.json").write_text(
        json.dumps({k: v for k, v in response.items() if k != "events"}, indent=2),
        encoding="utf-8",
    )
    response["duration_ms"] = int((time.monotonic() - started) * 1000)
    return response


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = "codex-bridge/1.0"
    state: BridgeState  # set by serve()

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
        self._send(
            200,
            {
                "ok": True,
                "model": self.state.runner.model,
                "codex_bin": self.state.runner.binary,
                "workdir": str(self.state.runner.workdir),
                "runs_dir": str(self.state.runs_dir),
                "timeout_s": self.state.runner.timeout_s,
                "timeout_first_turn_s": self.state.runner.timeout_first_turn_s,
                "stats": self.state.stats,
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

        self.state.bump("requests")
        self.state.bump("turns")
        try:
            response = _decide(self.state, payload)
        except BridgeError as exc:
            self.state.bump("errors")
            self._send(400, {"ok": False, "error_kind": exc.kind, "error": str(exc)})
            return
        except BaseException as exc:  # noqa: BLE001 - the episode must not die on our bug
            self.state.bump("errors")
            self._send(
                500,
                {
                    "ok": False,
                    "error_kind": "bridge_internal_error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return

        if response["ok"]:
            self.state.bump("ok")
        elif response.get("timed_out"):
            self.state.bump("timeouts")
        else:
            self.state.bump("errors")
        self._send(200 if response["ok"] else 502, response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Codex bridge for the codex_agent policy.")
    parser.add_argument("--config", type=str, default=None, help="JSON config file (optional)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--codex-bin", default="/Applications/ChatGPT.app/Contents/Resources/codex")
    parser.add_argument("--model", default=None, help="Model id, e.g. gpt-6-astra")
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--sandbox", default="read-only")
    parser.add_argument("--timeout-s", type=float, default=75.0)
    parser.add_argument("--timeout-first-turn-s", type=float, default=90.0)
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--token", default=os.environ.get("CODEX_BRIDGE_TOKEN"))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.config:
        with Path(args.config).expanduser().open("r", encoding="utf-8") as handle:
            for key, value in json.load(handle).items():
                setattr(args, key.replace("-", "_"), value)

    state = BridgeState(args)
    BridgeHandler.state = state

    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    server.daemon_threads = True
    print(
        f"[bridge] listening on http://{args.host}:{args.port}  "
        f"codex={state.runner.binary}  model={state.runner.model}  "
        f"runs={state.runs_dir}  timeout={state.runner.timeout_s}s",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
