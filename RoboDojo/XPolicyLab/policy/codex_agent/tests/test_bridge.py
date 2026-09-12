"""Tests for the local Codex bridge, driven by the ``fake_codex`` stub.

Covers :mod:`bridge.codex_runner` directly and :mod:`bridge.server` over real
HTTP. **The real Codex is never invoked**, so this burns no quota -- a hard
requirement from the operator, who wants the whole allowance left for the
official run.

The stub is steered through environment variables. Because the runner copies
``os.environ`` when it launches a child, the sandbox writes those variables into
the *live* environment and puts them back on ``close()``; setting them any other
way would be a snapshot the test could no longer steer.

    python tests/test_bridge.py
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
_BRIDGE = _PKG / "bridge"
for _entry in (str(_PKG.parent), str(_BRIDGE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import server as bridge_server  # noqa: E402
from codex_runner import CodexRunner, extract_json_object, parse_events  # noqa: E402

FAKE_CODEX = Path(__file__).resolve().parent / "fake_codex"

# Only the magic bytes matter to the bridge; everything after them is filler, so
# there is no reason to embed a real (and transcription-error-prone) image.
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

REPLY = {
    "left": {"position": [-0.2, -0.15, 0.95], "orientation": "keep", "gripper": "close"},
    "right": {"position": "keep", "orientation": "keep", "gripper": "keep"},
    "note": "reaching for the plug",
    "phase": "approach",
}

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL  {label}")


class Sandbox:
    """A temp dir plus ownership of the environment variables that steer the stub."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="codex_bridge_test_"))
        self.workdir = self.root / "work"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.runs = self.root / "runs"
        self._saved: dict[str, str | None] = {}

    def set(self, **variables: str) -> None:
        for key, value in variables.items():
            self._saved.setdefault(key, os.environ.get(key))
            os.environ[key] = str(value)

    def unset(self, *keys: str) -> None:
        for key in keys:
            self._saved.setdefault(key, os.environ.get(key))
            os.environ.pop(key, None)

    def runner(self, **kwargs) -> CodexRunner:
        options = dict(
            binary=str(FAKE_CODEX),
            model="gpt-6-astra",
            reasoning_effort="low",
            workdir=self.workdir,
            timeout_s=10.0,
            timeout_first_turn_s=10.0,
        )
        options.update(kwargs)
        return CodexRunner(**options)

    def close(self) -> None:
        for key, previous in self._saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        self._saved.clear()
        shutil.rmtree(self.root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #


def test_extract_json_object() -> None:
    print("extract_json_object")
    check(extract_json_object('{"a": 1}') == {"a": 1}, "a bare object parses")
    check(extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}, "a fenced object parses")
    check(extract_json_object('```\n{"a": 1}\n```') == {"a": 1}, "an unlabelled fence parses")
    check(
        extract_json_object('Sure, here you go:\n{"a": 1}\nHope that helps!') == {"a": 1},
        "prose around the object is tolerated",
    )
    check(
        extract_json_object('{"a": {"b": 2}} trailing') == {"a": {"b": 2}},
        "the outermost span is taken",
    )
    check(
        extract_json_object('prose\n```json\n{"a": 1}\n```\nmore prose') == {"a": 1},
        "a fenced object inside prose is recovered",
    )
    for bad in ("", "   ", "no json here", "[1, 2, 3]", '{"a": ', "null", "just a sentence"):
        check(extract_json_object(bad) is None, f"{bad!r} yields None")


def test_parse_events() -> None:
    print("parse_events")
    stream = "\n".join(
        [
            "not json at all",
            json.dumps({"type": "thread.started", "thread_id": "thr_1"}),
            "",
            json.dumps({"type": "item.completed", "item": {"type": "reasoning"}}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "first"}}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "second"}}),
            json.dumps({"type": "turn.completed", "usage": {"output_tokens": 7}}),
        ]
    )
    thread_id, message, usage, events = parse_events(stream)
    check(thread_id == "thr_1", "the thread id is read")
    check(message == "second", "the last agent message wins")
    check(usage == {"output_tokens": 7}, "usage is captured")
    check(len(events) == 5, f"each JSON event is summarised, got {len(events)}")
    check(all("type" in event for event in events), "every summary has a type")
    check(events[0] == {"type": "thread.started", "thread_id": "thr_1"}, "the first event is the thread")
    check(events[-1]["type"] == "turn.completed", "the last event is the turn")

    thread_id, message, usage, events = parse_events("")
    check(
        thread_id is None and message == "" and usage == {} and events == [],
        "an empty stream is safe",
    )


def test_build_command() -> None:
    print("build_command")
    sandbox = Sandbox()
    try:
        runner = sandbox.runner()
        cmd = runner.build_command(prompt="hello", image_paths=["/tmp/a.jpg", "/tmp/b.jpg"])
        check(cmd[:2] == [str(FAKE_CODEX), "exec"], "a first turn runs `exec`")
        check(cmd[-2:] == ["--", "hello"], "the prompt is terminated by `--`")
        check(cmd.count("-i") == 2, "each image gets its own -i")
        check("-s" not in cmd, "`-s` is never used (resume does not accept it)")
        check(
            "-c" in cmd and 'sandbox_mode="read-only"' in cmd,
            "the sandbox is set through -c, which both subcommands accept",
        )
        check('model_reasoning_effort="low"' in cmd, "the reasoning effort is passed")
        check(cmd[cmd.index("-m") + 1] == "gpt-6-astra", "the model is passed")
        check(cmd.index("--") > cmd.index("-i"), "the separator comes after the images")

        resumed = runner.build_command(prompt="hi", thread_id="thr_9", image_paths=["/tmp/a.jpg"])
        check(resumed[1:4] == ["exec", "resume", "thr_9"], "a follow-up turn resumes the thread")
        check(resumed[-2:] == ["--", "hi"], "the resumed prompt is still terminated by `--`")
        check("-i" in resumed, "resume still receives its image")

        no_model = CodexRunner(binary=str(FAKE_CODEX), model=None, workdir=sandbox.workdir)
        check("-m" not in no_model.build_command(prompt="x"), "no -m when no model is configured")

        with_paths = runner.build_command(
            prompt="x", schema_path="/tmp/s.json", last_message_path="/tmp/o.txt"
        )
        check("--output-schema" in with_paths, "the schema path is passed")
        check(
            with_paths[with_paths.index("--output-schema") + 1] == "/tmp/s.json",
            "the schema path is right",
        )
        check("-o" in with_paths, "the last-message path is passed")
        check(with_paths[with_paths.index("-o") + 1] == "/tmp/o.txt", "the output path is right")
    finally:
        sandbox.close()


def test_binary_resolution() -> None:
    print("binary resolution")
    sandbox = Sandbox()
    try:
        try:
            CodexRunner(binary="/definitely/not/here/codex", workdir=sandbox.workdir)
        except FileNotFoundError as exc:
            check("codex binary not found" in str(exc), "a missing binary raises with a hint")
        else:
            check(False, "a missing binary must raise")
    finally:
        sandbox.close()


def test_build_parser_defaults() -> None:
    print("the bridge CLI defaults")
    parsed = bridge_server.build_parser().parse_args([])
    check(parsed.host == "127.0.0.1", "the bridge binds loopback only")
    check(parsed.port == 8765, "the default port matches the documented tunnel")
    check(parsed.sandbox == "read-only", "codex runs read-only by default")
    check(parsed.timeout_s < 120.0, "the per-turn timeout leaves room under the client's 120 s")
    check(parsed.timeout_first_turn_s < 120.0, "so does the first-turn timeout")
    check(parsed.token is None, "no token unless one is configured")
    check(parsed.reasoning_effort is None, "the reasoning effort is opt-in")


def test_detect_magic() -> None:
    print("detect_magic")
    check(bridge_server.detect_magic(JPEG_BYTES) == "jpeg", "JPEG magic bytes are detected")
    check(bridge_server.detect_magic(PNG_BYTES) == "png", "PNG magic bytes are detected")
    for bad in (b"", b"GIF89a", b"\xff\xd8", JPEG_BYTES[:2], b'{"not": "an image"}'):
        check(bridge_server.detect_magic(bad) is None, f"{bad[:8]!r} is rejected")


# --------------------------------------------------------------------------- #
# running the stub
# --------------------------------------------------------------------------- #


def test_run_happy_path() -> None:
    print("run() against the stub")
    sandbox = Sandbox()
    try:
        sandbox.set(FAKE_CODEX_THREAD_ID="thr_fake_0001", FAKE_CODEX_REPLY=json.dumps(REPLY))
        runner = sandbox.runner()

        last_message = sandbox.root / "last_message.txt"
        image = sandbox.root / "view.jpg"
        image.write_bytes(JPEG_BYTES)

        result = runner.run(prompt="decide", image_paths=[image], last_message_path=last_message)
        check(result.ok, f"the run succeeded: {result.error}")
        check(result.thread_id == "thr_fake_0001", "the thread id came from the event stream")
        check(result.usage.get("output_tokens") == 56, "usage was captured")
        check(result.error_kind == "", "no error kind on success")
        check(result.error == "", "no error text on success")
        check(json.loads(result.text) == REPLY, "the -o file contents are the reply")
        check(last_message.is_file(), "the -o file was written")
        check(result.returncode == 0, "the exit code is recorded")
        check(result.duration_s > 0.0, "the duration is recorded")
        check(len(result.cmd) > 0, "the command is retained for the audit log")

        # a follow-up turn resumes the same thread
        result2 = runner.run(prompt="again", thread_id=result.thread_id)
        check(result2.ok, f"the resumed run succeeded: {result2.error}")
        check(result2.thread_id == "thr_fake_0001", "the resumed run keeps the thread id")

        # and the stub recorded the command it received
        args_path = sandbox.root / "args.json"
        sandbox.set(FAKE_CODEX_ARGS_OUT=str(args_path))
        runner.run(prompt="recorded", thread_id=result.thread_id, image_paths=[image])
        recorded = json.loads(args_path.read_text(encoding="utf-8"))
        check(recorded["subcommand"] == "resume", "the second turn resumed")
        check(recorded["thread_id"] == "thr_fake_0001", "the right thread was resumed")
        check(recorded["prompt"] == "recorded", "the prompt arrived intact")
        check(recorded["images_exist"] == [True], "the image file existed when codex read it")
        check(recorded["model"] == "gpt-6-astra", "the model reached the CLI")
        check(recorded["json"] is True, "the --json stream was requested")
        check(recorded["skip_git_repo_check"] is True, "the git check was skipped")
        check('sandbox_mode="read-only"' in recorded["config"], "the sandbox reached the CLI")
        check(
            'model_reasoning_effort="low"' in recorded["config"],
            "the reasoning effort reached the CLI",
        )
        # realpath, because macOS resolves the /var -> /private/var symlink in the child
        check(
            os.path.realpath(recorded["cwd"]) == os.path.realpath(sandbox.workdir),
            f"the working directory is the configured workdir, got {recorded['cwd']}",
        )
    finally:
        sandbox.close()


def test_run_jsonl_fallback_and_fences() -> None:
    print("run() fallbacks: -o missing, code fences, prose")
    sandbox = Sandbox()
    try:
        sandbox.set(
            FAKE_CODEX_REPLY=json.dumps(REPLY),
            FAKE_CODEX_NO_OUTPUT_FILE="1",
            FAKE_CODEX_FENCE="1",
            FAKE_CODEX_PROSE="Here is my decision:",
        )
        runner = sandbox.runner()

        last_message = sandbox.root / "absent.txt"
        result = runner.run(prompt="decide", last_message_path=last_message)
        check(result.ok, f"the run succeeded: {result.error}")
        check(not last_message.exists(), "the stub honoured FAKE_CODEX_NO_OUTPUT_FILE")
        check(result.text.startswith("Here is my decision:"), "the text fell back to the JSONL message")
        check(extract_json_object(result.text) == REPLY, "the fenced reply is recoverable")

        # An empty -o file must not shadow a perfectly good JSONL message.
        empty = sandbox.root / "empty.txt"
        empty.write_text("", encoding="utf-8")
        sandbox.set(FAKE_CODEX_NO_OUTPUT_FILE="1")
        result = runner.run(prompt="decide", last_message_path=empty)
        check(result.ok, "an empty -o file still succeeds via the JSONL message")
        check(extract_json_object(result.text) == REPLY, "the JSONL message was used")
    finally:
        sandbox.close()


def test_run_timeout_kills_the_process() -> None:
    print("run() timeout")
    sandbox = Sandbox()
    try:
        sandbox.set(FAKE_CODEX_SLEEP_S="30")
        runner = sandbox.runner(timeout_s=0.6, timeout_first_turn_s=0.6)
        started = time.monotonic()
        result = runner.run(prompt="decide")
        elapsed = time.monotonic() - started

        check(not result.ok, "a hung codex is not ok")
        check(result.timed_out, "the result is flagged as timed out")
        check(result.error_kind == "codex_timeout", f"error_kind={result.error_kind!r}")
        check(not result.thread_lost, "a timeout is not mistaken for a lost thread")
        check(elapsed < 6.0, f"the call returned promptly, took {elapsed:.1f}s")
        check("did not finish within" in result.error, "the error explains the timeout")

        # The process must actually be gone, not merely abandoned.
        if shutil.which("pgrep"):
            deadline = time.monotonic() + 3.0
            survivors = "?"
            while time.monotonic() < deadline:
                probe = subprocess.run(
                    ["pgrep", "-f", str(FAKE_CODEX)], capture_output=True, text=True
                )
                survivors = probe.stdout.strip()
                if probe.returncode != 0:
                    break
                time.sleep(0.1)
            check(survivors == "", f"the killed process left nothing behind, saw {survivors!r}")
        else:  # pragma: no cover - pgrep ships with macOS
            check(True, "pgrep unavailable; the orphan check was skipped")
    finally:
        sandbox.close()


def test_run_failure_modes() -> None:
    print("run() failure modes")
    sandbox = Sandbox()
    try:
        runner = sandbox.runner()

        sandbox.set(FAKE_CODEX_EXIT="3", FAKE_CODEX_STDERR="something exploded")
        result = runner.run(prompt="decide")
        check(not result.ok and result.error_kind == "codex_failed", "a non-zero exit is codex_failed")
        check("exited with 3" in result.error, "the error reports the exit code")
        check("something exploded" in result.stderr, "stderr is kept for the audit log")
        check("something exploded" in result.transcript_tail(), "the transcript tail includes stderr")
        sandbox.unset("FAKE_CODEX_EXIT", "FAKE_CODEX_STDERR")

        # resuming an unknown thread is a distinct, recoverable failure
        result = runner.run(prompt="decide", thread_id="thr_missing")
        check(not result.ok, "resuming an unknown thread fails")
        check(result.error_kind == "thread_not_found", f"error_kind={result.error_kind!r}")
        check(result.thread_lost, "the result is flagged as a lost thread")
        check("thr_missing" in result.error, "the error names the thread")

        # the lost-thread marker without a thread id is an ordinary failure
        sandbox.set(FAKE_CODEX_EXIT="9", FAKE_CODEX_STDERR="session not found")
        result = runner.run(prompt="decide")
        check(result.error_kind == "codex_failed", "the lost-thread marker needs a thread id")
        check(not result.thread_lost, "so it is not reported as a lost thread")
        sandbox.unset("FAKE_CODEX_EXIT", "FAKE_CODEX_STDERR")

        # no message at all
        sandbox.set(FAKE_CODEX_NO_MESSAGE="1", FAKE_CODEX_NO_OUTPUT_FILE="1")
        last_message = sandbox.root / "none.txt"
        result = runner.run(prompt="decide", last_message_path=last_message)
        check(not result.ok and result.error_kind == "codex_empty_reply", "an empty reply is reported")
        sandbox.unset("FAKE_CODEX_NO_MESSAGE", "FAKE_CODEX_NO_OUTPUT_FILE")

        # a spawn failure is a result, not an exception
        broken = CodexRunner(binary=str(FAKE_CODEX), workdir=sandbox.workdir)
        broken.binary = str(sandbox.root / "not_a_binary")
        result = broken.run(prompt="decide")
        check(
            not result.ok and result.error_kind == "codex_spawn_failed",
            f"a spawn failure is reported, got {result.error_kind!r}",
        )
        check(result.cmd, "the attempted command is retained for the audit log")
    finally:
        sandbox.close()


def test_three_views_and_a_fresh_thread() -> None:
    print("three views, and a fresh thread on the first turn")
    sandbox = Sandbox()
    try:
        runner = sandbox.runner()
        images = []
        for index, payload in enumerate([JPEG_BYTES, PNG_BYTES, JPEG_BYTES]):
            path = sandbox.root / f"view_{index}"
            path.write_bytes(payload)
            images.append(path)
        args_path = sandbox.root / "args3.json"
        sandbox.set(FAKE_CODEX_ARGS_OUT=str(args_path))

        result = runner.run(prompt="decide", image_paths=images)
        check(result.ok, f"the run succeeded: {result.error}")
        recorded = json.loads(args_path.read_text(encoding="utf-8"))
        check(len(recorded["images"]) == 3, "all three views were passed")
        check(all(recorded["images_exist"]), "all three view files existed")
        check(recorded["subcommand"] == "exec", "a turn with no thread id runs `exec`")
        check(recorded["thread_id"] is None, "and it resumes nothing")
        check(result.thread_id == "thr_fake_0001", "the fresh thread id is reported back")
    finally:
        sandbox.close()


# --------------------------------------------------------------------------- #
# the HTTP server
# --------------------------------------------------------------------------- #


class BridgeUnderTest:
    """Runs bridge/server.py in-process on an ephemeral port."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.args = bridge_server.build_parser().parse_args(
            ["--runs-dir", str(sandbox.runs), "--workdir", str(sandbox.workdir), "--quiet"]
        )
        self.args.codex_bin = str(FAKE_CODEX)
        self.args.model = "gpt-6-astra"
        self.args.reasoning_effort = "low"
        self.args.timeout_s = 5.0
        self.args.timeout_first_turn_s = 5.0
        self.args.port = 0

        self.state = bridge_server.BridgeState(self.args)
        handler = type("Handler", (bridge_server.BridgeHandler,), {"state": self.state})
        self.server = bridge_server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def post(self, path: str, body, timeout: float = 30.0):
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def get(self, path: str, token: str | None = None):
        headers = {"X-Bridge-Token": token} if token else {}
        request = urllib.request.Request(self.url + path, headers=headers)
        with urllib.request.urlopen(request, timeout=10.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_server_end_to_end() -> None:
    print("bridge server over HTTP")
    sandbox = Sandbox()
    bridge = None
    try:
        sandbox.set(FAKE_CODEX_REPLY=json.dumps(REPLY))
        bridge = BridgeUnderTest(sandbox)

        status, health = bridge.get("/healthz")
        check(status == 200 and health["ok"], "healthz answers")
        check(health["model"] == "gpt-6-astra", "healthz reports the model")
        check(health["codex_bin"] == str(FAKE_CODEX), "healthz reports the codex binary")

        payload = {
            "episode_id": "ep_0001",
            "thread_id": None,
            "turn_index": 0,
            "prompt": "decide",
            "schema": {"type": "object"},
            "images": [
                {"name": "cam_head", "b64": base64.b64encode(JPEG_BYTES).decode()},
                {"name": "cam_left_wrist", "b64": base64.b64encode(PNG_BYTES).decode()},
            ],
        }
        status, body = bridge.post("/v1/decide", payload)
        check(status == 200, f"decide answered 200, got {status} {body.get('error')}")
        check(body["ok"], f"the decision succeeded: {body.get('error')}")
        check(body["parsed"] == REPLY, "the parsed reply came back")
        check(body["thread_id"] == "thr_fake_0001", "a thread id came back for the adapter to keep")
        check(body["usage"].get("output_tokens") == 56, "usage is surfaced to the adapter")
        check(body["latency_ms"] >= 0, "latency is reported")
        check(body["error"] == "" and body["error_kind"] == "", "no error on a good turn")

        turn_dir = Path(body["runs_dir"])
        check(turn_dir.is_dir(), "the audit directory exists")
        check(turn_dir.name == "turn_000", f"the turn directory is numbered, got {turn_dir.name}")
        check(turn_dir.parent.name == "ep_0001", "the audit directory is keyed by episode")
        for name in ("prompt.txt", "schema.json", "stdout.jsonl", "last_message.txt", "result.json"):
            check((turn_dir / name).is_file(), f"the audit trail contains {name}")
        check((turn_dir / "prompt.txt").read_text(encoding="utf-8") == "decide", "the prompt was saved")
        check(
            json.loads((turn_dir / "result.json").read_text(encoding="utf-8"))["ok"] is True,
            "result.json records the outcome",
        )
        images = sorted((turn_dir / "images").iterdir())
        check(len(images) == 2, f"both views were written, got {len(images)}")
        check(
            [path.suffix for path in images] == [".jpg", ".png"],
            f"the magic bytes chose the extensions, got {[p.name for p in images]}",
        )
        check(images[0].name.startswith("00_cam_head"), f"the view name is kept, got {images[0].name}")

        # the second turn resumes
        payload.update(thread_id=body["thread_id"], turn_index=1)
        status, body2 = bridge.post("/v1/decide", payload)
        check(status == 200 and body2["ok"], "the resumed turn succeeded")
        check(body2["thread_id"] == "thr_fake_0001", "the thread id is stable across turns")
        check(Path(body2["runs_dir"]).name == "turn_001", "the second turn gets its own directory")

        # a vanished thread is rebuilt once, transparently
        payload.update(thread_id="thr_gone", turn_index=2)
        status, body3 = bridge.post("/v1/decide", payload)
        check(status == 200 and body3["ok"], "a lost thread is rebuilt and the turn still succeeds")
        check(body3["thread_id"] == "thr_fake_0001", "the rebuilt turn reports the new thread")

        check(bridge.state.stats["requests"] == 3, "the request counter tracked all three")
        check(bridge.state.stats["ok"] == 3, "the success counter tracked all three")

        # a reply that is not JSON is reported, not silently accepted
        sandbox.set(
            FAKE_CODEX_REPLY=json.dumps(["not", "an", "object"]), FAKE_CODEX_PROSE="sorry"
        )
        status, body4 = bridge.post("/v1/decide", {"episode_id": "ep_text", "prompt": "x"})
        check(status == 502, f"an unparseable reply is a 502, got {status}")
        check(body4["error_kind"] == "codex_unparseable", f"got {body4['error_kind']!r}")
        check(body4["parsed"] is None, "nothing was parsed")
        check(body4["text"], "the raw text is returned so the adapter can decide what to do")
    finally:
        if bridge is not None:
            bridge.close()
        sandbox.close()


def test_server_rejections() -> None:
    print("bridge server input validation")
    sandbox = Sandbox()
    bridge = None
    try:
        bridge = BridgeUnderTest(sandbox)

        status, body = bridge.post("/v1/decide", {"episode_id": "e", "prompt": "   "})
        check(status == 400 and body["error_kind"] == "bad_request", "an empty prompt is rejected")

        status, body = bridge.post("/v1/decide", {"episode_id": "e", "prompt": "x", "images": "nope"})
        check(status == 400, "a non-list images field is rejected")
        check("must be a list" in body["error"], f"and says so: {body['error']}")

        status, body = bridge.post(
            "/v1/decide",
            {"episode_id": "e", "prompt": "x", "images": [{"name": "a", "b64": "!!!not base64!!!"}]},
        )
        check(
            status == 400 and "base64" in body["error"],
            f"invalid base64 is rejected: {body['error']}",
        )

        status, body = bridge.post(
            "/v1/decide",
            {
                "episode_id": "e",
                "prompt": "x",
                "images": [{"name": "a", "b64": base64.b64encode(b"plain text").decode()}],
            },
        )
        check(
            status == 400 and "JPEG nor PNG" in body["error"],
            f"a non-image is rejected: {body['error']}",
        )

        oversize = base64.b64encode(b"\xff\xd8\xff" + b"x" * (5 * 1024 * 1024)).decode()
        status, body = bridge.post(
            "/v1/decide",
            {"episode_id": "e", "prompt": "x", "images": [{"name": "a", "b64": oversize}]},
        )
        check(
            status == 400 and "over the" in body["error"],
            f"an oversize image is rejected: {body['error']}",
        )

        status, body = bridge.post(
            "/v1/decide", {"episode_id": "e", "prompt": "x", "images": [{"b64": ""}]}
        )
        check(status == 400, "an empty image payload is rejected")

        status, body = bridge.post("/v1/nope", {"prompt": "x"})
        check(status == 404, "an unknown POST path is 404")

        # a failure inside codex still returns a well-formed HTTP response
        sandbox.set(FAKE_CODEX_EXIT="7", FAKE_CODEX_STDERR="boom")
        status, body = bridge.post("/v1/decide", {"episode_id": "e", "prompt": "x", "turn_index": 1})
        check(status == 502, f"a codex failure is a 502, got {status}")
        check(body["ok"] is False and body["error_kind"] == "codex_failed", "and it is explained")
        check(bool(body["error"]), "the error text is present")
        check(bridge.state.stats["errors"] >= 1, "the error counter ticked")
    finally:
        if bridge is not None:
            bridge.close()
        sandbox.close()


def test_server_timeout_returns_http() -> None:
    """The 120 s client timeout is the one failure that silently kills an episode."""
    print("bridge server timeout still answers")
    sandbox = Sandbox()
    bridge = None
    try:
        sandbox.set(FAKE_CODEX_SLEEP_S="30")
        bridge = BridgeUnderTest(sandbox)
        bridge.state.runner.timeout_s = 0.6
        bridge.state.runner.timeout_first_turn_s = 0.6

        started = time.monotonic()
        status, body = bridge.post(
            "/v1/decide", {"episode_id": "ep_timeout", "prompt": "decide"}, timeout=25.0
        )
        elapsed = time.monotonic() - started
        check(status == 502, f"a timed-out codex is a 502, got {status}")
        check(body["ok"] is False, "the timeout is not reported as ok")
        check(body["timed_out"] is True, "the result is flagged as timed out")
        check(body["error_kind"] == "codex_timeout", f"error_kind={body['error_kind']!r}")
        check(elapsed < 10.0, f"the bridge answered in {elapsed:.1f}s, well under the client's 120 s")
        check(bridge.state.stats["timeouts"] == 1, "the timeout counter ticked")
        check(
            Path(body["runs_dir"], "result.json").is_file(),
            "even a timeout leaves an audit record",
        )
    finally:
        if bridge is not None:
            bridge.close()
        sandbox.close()


def test_server_token_auth() -> None:
    print("bridge server token")
    sandbox = Sandbox()
    bridge = None
    try:
        bridge = BridgeUnderTest(sandbox)
        bridge.state.args.token = "s3cret"

        try:
            bridge.get("/healthz")
        except urllib.error.HTTPError as exc:
            check(exc.code == 401, "a missing token is 401")
        else:
            check(False, "a missing token must be rejected")

        status, health = bridge.get("/healthz", token="s3cret")
        check(status == 200 and health["ok"], "the right token is accepted")

        try:
            bridge.get("/healthz", token="wrong")
        except urllib.error.HTTPError as exc:
            check(exc.code == 401, "a wrong token is 401")
        else:
            check(False, "a wrong token must be rejected")

        status, _ = bridge.post("/v1/decide", {"prompt": "x"})
        check(status == 401, f"decide enforces the token too, got {status}")
    finally:
        if bridge is not None:
            bridge.close()
        sandbox.close()


def main() -> int:
    if not FAKE_CODEX.is_file():
        print(f"missing stub at {FAKE_CODEX}")
        return 1
    FAKE_CODEX.chmod(0o755)

    for test in (
        test_extract_json_object,
        test_parse_events,
        test_build_command,
        test_binary_resolution,
        test_build_parser_defaults,
        test_detect_magic,
        test_run_happy_path,
        test_run_jsonl_fallback_and_fences,
        test_run_timeout_kills_the_process,
        test_run_failure_modes,
        test_three_views_and_a_fresh_thread,
        test_server_end_to_end,
        test_server_rejections,
        test_server_timeout_returns_http,
        test_server_token_auth,
    ):
        test()
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} of {CHECKS} checks:")
        for label in FAILURES[:40]:
            print(f"  - {label}")
        return 1
    print(f"ok: {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
