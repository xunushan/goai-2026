"""Wrap the local Codex CLI as a stateful, one-decision-per-call black box.

Verified against ``codex-cli 0.153.4``. Two things about that CLI shape this
module and are easy to get wrong:

* ``codex exec`` takes ``-i`` **variadically**, so the prompt must be separated
  from the image list with ``--`` or it is swallowed as another image path.
* ``codex exec resume`` supports neither ``-s`` nor ``-C``, and its ``-i`` takes
  a single value. The working directory therefore has to come from the
  subprocess ``cwd``, which also matters because ``resume`` filters recorded
  sessions by working directory.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

DEFAULT_BIN = "/Applications/ChatGPT.app/Contents/Resources/codex"

# Substrings that mean "this thread id is gone", so the caller can start a fresh
# thread instead of failing the episode.
_THREAD_LOST_MARKERS = (
    "session not found",
    "no such session",
    "thread not found",
    "no rollout found",
    "could not find",
    "unknown session",
)


@dataclass
class CodexRunResult:
    ok: bool
    thread_id: str | None = None
    text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    error_kind: str = ""
    error: str = ""
    cmd: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    output_last_message: str = ""

    @property
    def thread_lost(self) -> bool:
        if self.timed_out or not self.error_kind:
            return False
        return self.error_kind == "thread_not_found"

    def transcript_tail(self, limit: int = 2000) -> str:
        body = (self.stderr or "") + "\n" + (self.stdout or "")
        return body.strip()[-limit:]


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model reply, tolerating prose and code fences.

    Cascade: parse as-is, then strip a ``` fence, then take the outermost
    ``{...}`` span. Returns ``None`` when nothing parses -- the caller records
    the raw text and moves on rather than guessing.
    """
    if not text:
        return None
    candidates = [text.strip()]
    stripped = text.strip()
    if stripped.startswith("```"):
        body = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
        if body.rstrip().endswith("```"):
            body = body.rstrip()[: -len("```")]
        candidates.append(body.strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_events(stdout: str) -> tuple[str | None, str, dict[str, Any], list[dict[str, Any]]]:
    """Scan the ``--json`` event stream.

    Returns ``(thread_id, last_agent_message, usage, events)``. Unknown or
    non-JSON lines are kept in ``events`` for the audit log but ignored here.
    """
    thread_id: str | None = None
    last_message = ""
    usage: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = str(event.get("type") or "")
        if kind == "thread.started":
            thread_id = str(event.get("thread_id") or thread_id or "") or thread_id
            summary: dict[str, Any] = {"type": kind, "thread_id": thread_id}
        elif kind == "item.completed":
            item = event.get("item")
            item_type = str(item.get("type")) if isinstance(item, dict) else "?"
            summary = {"type": kind, "item_type": item_type}
            if item_type == "agent_message" and isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    last_message = text
        elif kind == "turn.completed":
            if isinstance(event.get("usage"), dict):
                usage = dict(event["usage"])
            summary = {"type": kind, "usage": usage}
        elif kind in ("error", "turn.failed"):
            summary = {"type": kind, "detail": str(event)[:400]}
        else:
            summary = {"type": kind or "unknown"}
        if len(events) < 200:
            events.append(summary)
    return thread_id, last_message, usage, events


class CodexRunner:
    """Runs ``codex exec`` / ``codex exec resume`` with a hard timeout."""

    def __init__(
        self,
        *,
        binary: str = DEFAULT_BIN,
        model: str | None = None,
        reasoning_effort: str | None = None,
        workdir: str | Path,
        timeout_s: float = 75.0,
        timeout_first_turn_s: float | None = None,
        sandbox: str = "read-only",
        codex_home: str | None = None,
        env_extra: dict[str, str] | None = None,
    ) -> None:
        self.binary = self._resolve_binary(binary)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.workdir = Path(workdir).expanduser()
        self.timeout_s = float(timeout_s)
        self.timeout_first_turn_s = float(
            timeout_first_turn_s if timeout_first_turn_s is not None else timeout_s
        )
        self.sandbox = sandbox
        self.codex_home = codex_home
        self.env_extra = dict(env_extra or {})
        self.workdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_binary(binary: str) -> str:
        path = Path(binary).expanduser()
        if path.is_file():
            return str(path)
        found = shutil.which(binary)
        if found:
            return found
        raise FileNotFoundError(
            f"codex binary not found at {binary!r} and not on PATH. The Codex CLI "
            "bundled with the ChatGPT desktop app lives at "
            f"{DEFAULT_BIN}."
        )

    # ------------------------------------------------------------------ #
    def build_command(
        self,
        *,
        prompt: str,
        image_paths: Sequence[str | Path] = (),
        thread_id: str | None = None,
        schema_path: str | Path | None = None,
        last_message_path: str | Path | None = None,
    ) -> list[str]:
        cmd = [self.binary, "exec"]
        if thread_id:
            cmd += ["resume", str(thread_id)]
        cmd.append("--skip-git-repo-check")
        cmd += ["--json"]
        if last_message_path is not None:
            cmd += ["-o", str(last_message_path)]
        if schema_path is not None:
            cmd += ["--output-schema", str(schema_path)]
        if self.model:
            cmd += ["-m", str(self.model)]
        # `-s` exists on `exec` but not on `resume`; `-c` works on both.
        cmd += ["-c", f'sandbox_mode="{self.sandbox}"']
        if self.reasoning_effort:
            cmd += ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        for path in image_paths:
            cmd += ["-i", str(path)]
        cmd += ["--", prompt]
        return cmd

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("NO_COLOR", "1")
        if self.codex_home:
            env["CODEX_HOME"] = str(self.codex_home)
        env.update(self.env_extra)
        return env

    def run(
        self,
        *,
        prompt: str,
        image_paths: Sequence[str | Path] = (),
        thread_id: str | None = None,
        schema_path: str | Path | None = None,
        last_message_path: str | Path | None = None,
        timeout_s: float | None = None,
    ) -> CodexRunResult:
        """Run one Codex turn. Never raises: every failure becomes a result."""
        cmd = self.build_command(
            prompt=prompt,
            image_paths=image_paths,
            thread_id=thread_id,
            schema_path=schema_path,
            last_message_path=last_message_path,
        )
        budget = float(
            timeout_s
            if timeout_s is not None
            else (self.timeout_first_turn_s if thread_id is None else self.timeout_s)
        )
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self.workdir),
                env=self._env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,  # own process group, so we can kill the tree
            )
        except OSError as exc:
            return CodexRunResult(
                ok=False,
                error_kind="codex_spawn_failed",
                error=f"{type(exc).__name__}: {exc}",
                cmd=cmd,
                duration_s=time.monotonic() - started,
            )

        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=budget)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate(process)
            stdout, stderr = process.communicate()
        duration = time.monotonic() - started

        result = CodexRunResult(
            ok=False,
            stdout=stdout or "",
            stderr=stderr or "",
            returncode=process.returncode,
            timed_out=timed_out,
            cmd=cmd,
            duration_s=duration,
        )

        parsed_thread, message, usage, events = parse_events(result.stdout)
        result.thread_id = parsed_thread or thread_id
        result.usage = usage
        result.events = events

        output_file_text = ""
        if last_message_path is not None:
            try:
                output_file_text = Path(last_message_path).read_text(encoding="utf-8")
            except OSError:
                output_file_text = ""
        result.output_last_message = output_file_text
        result.text = (output_file_text.strip() or message or "").strip()

        if timed_out:
            result.error_kind = "codex_timeout"
            result.error = f"codex did not finish within {budget:.0f}s"
            return result
        if process.returncode != 0:
            combined = (result.stderr + result.stdout).lower()
            if thread_id and any(marker in combined for marker in _THREAD_LOST_MARKERS):
                result.error_kind = "thread_not_found"
                result.error = f"thread {thread_id} is no longer available"
            else:
                result.error_kind = "codex_failed"
                result.error = f"codex exited with {process.returncode}"
            return result
        if not result.text:
            result.error_kind = "codex_empty_reply"
            result.error = "codex returned no message"
            return result

        result.ok = True
        return result

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        """SIGTERM the whole process group, then SIGKILL if it will not go."""
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            process.terminate()
        try:
            process.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
