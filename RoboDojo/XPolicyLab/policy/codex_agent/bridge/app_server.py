"""A persistent Codex App Server, spoken to over stdio JSON-RPC.

One process for the whole episode instead of one per decision. The reason is not
startup cost, it is context: the model is meant to remember what it just did, and
a thread that is re-created every turn cannot. Images accumulate in that thread's
context and there is no way to delete them, so the thread is rotated after
``max_live_image_turns`` image-bearing turns, carrying forward a text summary the
caller builds (see ``bridge.decide``). Dropping the thread and replaying words is
the only mechanism available for forgetting images.

The thread belongs to this class and never crosses the network: the adapter sends
an episode id, and this decides whether that id is the one the current thread was
opened for. A caller that has to round-trip a thread id is a caller that can send
back the wrong one.

Failure classification is the other half of the job. ``serverOverloaded`` and
``usageLimitExceeded`` are retryable conditions and must not look like a broken
turn, because the adapter treats them differently -- so they leave here as
distinct ``kind``s rather than as one generic failure.

Transport framing is inherited from the reference implementation this mirrors
(``agent_policy/src/agent_policy/app_server.py``), including the two details that
are easy to get wrong: messages that are not the response we are waiting for go
to a backlog rather than being discarded, and the child is killed by process
group so that a wrapper script does not survive its own worker.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

# Enough turns of memory to keep a decision in view, few enough that the image
# context does not grow without bound. The same number as the reference
# controller's ``live_image_window``.
DEFAULT_MAX_LIVE_IMAGE_TURNS = 3

SKILL_PATH = ".agents/skills/codex_agent/SKILL.md"

# The DeepSeek provider cannot come from the workspace's `.codex/config.toml`.
# Codex loads that file, but refuses these two keys from a project-local source
# and says so on startup:
#
#   Ignored unsupported project-local config keys in <workspace>/.codex/config.toml:
#   model_provider, model_providers. If you want these settings to apply, manually
#   set them in your user-level config.toml.
#
# Setting them in the user-level config would change the model provider for every
# Codex session on the machine, the desktop app included. `-c` reaches only the
# child process started below, which is why the override lives here.
#
# The key is named, never carried: `env_key` makes Codex read DEEPSEEK_API_KEY
# from the environment, so no secret is written to a file in this tree.
DEEPSEEK_MODEL_ARGS = (
    "-c", 'model="deepseek-flash"',
    "-c", 'model_provider="deepseek"',
    "-c", 'model_providers.deepseek.name="deepseek"',
    "-c", 'model_providers.deepseek.base_url="https://api.deepseek.com/"',
    "-c", 'model_providers.deepseek.wire_api="responses"',
    "-c", 'model_providers.deepseek.env_key="DEEPSEEK_API_KEY"',
)


class AppServerError(RuntimeError):
    """A turn did not produce an answer, with ``kind`` saying why.

    The kinds are load-bearing: the adapter retries ``policy_overloaded`` and
    gives up on ``policy_usage_exhausted``, so collapsing them into one error
    would silently change the retry policy.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class CodexAppServer:
    def __init__(
        self,
        *,
        workspace: Path,
        codex_bin: str = "codex",
        model: str | None = None,
        effort: str | None = None,
        max_live_image_turns: int = DEFAULT_MAX_LIVE_IMAGE_TURNS,
    ) -> None:
        self.workspace = Path(workspace)
        self.codex_bin = codex_bin
        self.model = model
        self.effort = effort
        self.max_live_image_turns = max_live_image_turns
        self.process: subprocess.Popen[str] | None = None
        self.thread_id: str | None = None
        self._episode_id: str | None = None
        self._live_image_turns = 0
        self._stdin: Any = None
        self._messages: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self._backlog: list[dict[str, Any]] = []
        self._next_id = 0

    # ----------------------------------------------------------------- #
    # lifecycle
    # ----------------------------------------------------------------- #

    def bind(self, episode_id: str) -> None:
        """Make sure the running server is the one for this episode.

        The model's write permission is scoped to ``output/<episode>/scratch`` and
        permissions are fixed when the process starts, so a different episode
        means a different process. Episodes are serial, so this happens once per
        episode rather than once per decision.
        """
        if self._episode_id == episode_id:
            return
        self.close()
        self._episode_id = episode_id

    def start(self) -> None:
        if self.process is not None:
            return
        if self._episode_id is None:
            raise AppServerError("app_server_not_bound", "bind() an episode before starting the server")
        # A fresh queue and backlog per process. The previous reader thread left an
        # end-of-stream marker in the old queue as it exited, and a restarted server
        # that reads that marker first believes it is already dead.
        messages = self._messages = queue.Queue()
        self._backlog = []
        self.process = subprocess.Popen(
            [
                self.codex_bin, "app-server", "--stdio", "--strict-config",
                *self._permission_args(),
                *self._model_args(),
            ],
            cwd=self.workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise AppServerError("app_server_start_failed", "Codex App Server stdio is unavailable")
        self._stdin = self.process.stdin
        threading.Thread(target=self._read_stdout, args=(self.process.stdout, messages), daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(self.process.stderr,), daemon=True).start()
        self._request("initialize", {"clientInfo": {"name": "codex-agent", "version": "0.1.0"}})
        self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

    def _permission_args(self) -> list[str]:
        """The sandbox, spelled out as flags rather than left to config files.

        A project-local ``.codex/config.toml`` is only honoured once the workspace
        has been marked trusted, and a permission that quietly does not apply is
        exactly the failure this project exists to avoid. The workspace file says
        the same thing for a human running ``codex`` by hand; this is what the
        bridge actually relies on.
        """
        scratch = self.workspace / "output" / str(self._episode_id) / "scratch"
        filesystem = "{" + ",".join(
            f"{json.dumps(str(path))} = {json.dumps(access)}"
            for path, access in (
                (self.workspace, "read"),
                (self.workspace / "output", "read"),
                (scratch, "write"),
            )
        ) + "}"
        return [
            "-c", 'default_permissions="rollout_agent"',
            "-c", 'permissions.rollout_agent.extends=":workspace"',
            "-c", "permissions.rollout_agent.filesystem=" + filesystem,
            "--enable", "shell_tool",
            "--enable", "view_image",
            "--disable", "web_search",
            "--disable", "computer_use",
            "--disable", "multi_agent",
            "--disable", "multi_agent_v2",
        ]

    def _model_args(self) -> list[str]:
        """Which model answers, and over which provider.

        The catalog is what tells Codex that ``deepseek-flash`` accepts images.
        The policy attaches three camera views per turn, and the only other
        DeepSeek entry in that catalog is text-only, so the pairing is not
        interchangeable. It is read at startup, which means a missing or
        malformed file fails the server rather than a turn.
        """
        catalog = self.workspace / ".codex" / "models.json"
        return [*DEEPSEEK_MODEL_ARGS, "-c", f"model_catalog_json={json.dumps(str(catalog))}"]

    def close(self) -> None:
        process, self.process = self.process, None
        self._stdin = None
        self.thread_id = None
        self._live_image_turns = 0
        if process is None or process.poll() is not None:
            return
        # By process group: the child may be a wrapper that does not forward a
        # signal to the real server, and an orphan would hold the only Codex
        # session this machine has.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass

    # ----------------------------------------------------------------- #
    # threads and turns
    # ----------------------------------------------------------------- #

    def start_thread(self) -> str:
        """The current thread, opening one if there is none."""
        self.start()
        if self.thread_id is not None:
            return self.thread_id
        params: dict[str, Any] = {
            "cwd": str(self.workspace),
            "approvalPolicy": "never",
            "sandbox": "workspace-write",
            "ephemeral": False,
            "baseInstructions": (
                "For every robot decision, follow the workspace skill codex_agent at "
                f"{SKILL_PATH} and the embodiment contract in AGENTS.md. "
                "Return only the required JSON object."
            ),
        }
        if self.model:
            params["model"] = self.model
        if self.effort:
            params["config"] = {"model_reasoning_effort": self.effort}
        result = self._request("thread/start", params)
        try:
            self.thread_id = str(result["thread"]["id"])
        except (KeyError, TypeError) as error:
            raise AppServerError("app_server_protocol", "thread/start did not return a thread id") from error
        return self.thread_id

    @property
    def rotation_due(self) -> bool:
        """The next image turn will open a fresh thread and lose the images.

        Exposed because the caller is the one that knows what is worth carrying
        across: the bridge has the earlier decisions written down and can hand
        back the words, and this class has only the thread.
        """
        return bool(self.max_live_image_turns) and self._live_image_turns >= self.max_live_image_turns

    def decide(
        self,
        *,
        text: str,
        images: list[tuple[str, str]],
        output_schema: dict[str, Any],
        timeout_s: float,
        rollover_context: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """One turn, returning ``(final text, usage)``.

        ``images`` is ``(camera name, data URL)`` in attachment order. Each one is
        preceded by its own text label, because three unlabelled pictures of the
        same table are not something a model can tell apart -- and the label is
        the same name the turn text just listed.
        """
        thread_id = self._thread_for_next_turn()
        inputs: list[dict[str, Any]] = [{
            "type": "text",
            "text": (rollover_context + "\n\n" if rollover_context else "") + text,
        }]
        for name, url in images:
            inputs.append({"type": "text", "text": f"Camera image: {name}"})
            inputs.append({"type": "image", "url": url})

        result = self._request("turn/start", {
            "threadId": thread_id,
            "input": inputs,
            "outputSchema": output_schema,
            **({"effort": self.effort} if self.effort else {}),
        })
        try:
            turn_id = str(result["turn"]["id"])
        except (KeyError, TypeError) as error:
            raise AppServerError("app_server_protocol", "turn/start did not return a turn id") from error
        answer, usage = self._wait_for_final(thread_id, turn_id, timeout_s)
        if images:
            self._live_image_turns += 1
        return answer, usage

    def _thread_for_next_turn(self) -> str:
        """Rotate before the next image turn when the window is full."""
        if self.rotation_due:
            self.thread_id = None
            self._live_image_turns = 0
        return self.start_thread()

    def _wait_for_final(self, thread_id: str, turn_id: str, timeout_s: float) -> tuple[str, dict[str, Any]]:
        answer: str | None = None
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AppServerError("policy_timeout", "Timed out waiting for Codex App Server")
            event = self._next_message(remaining)
            params = event.get("params", {})
            # A rotated-away thread still has a turn in flight, and its events keep
            # arriving. Anything not from the thread and turn we asked about is not
            # ours to interpret.
            if params.get("threadId") != thread_id:
                continue
            if params.get("turnId", turn_id) != turn_id:
                continue
            if event.get("method") == "item/completed":
                item = params.get("item", {})
                if item.get("type") == "agentMessage" and item.get("phase") in (None, "final_answer"):
                    answer = str(item.get("text", ""))
            if event.get("method") != "turn/completed":
                continue
            turn = params.get("turn", {})
            if turn.get("status") != "completed":
                error = turn.get("error") or {}
                info = error.get("codexErrorInfo") if isinstance(error, dict) else None
                if info == "serverOverloaded":
                    raise AppServerError("policy_overloaded", "Codex service is overloaded")
                if info == "usageLimitExceeded":
                    raise AppServerError("policy_usage_exhausted", "Codex usage limit is exhausted")
                raise AppServerError("policy_turn_failed", f"Codex turn failed: {error}")
            if not answer:
                raise AppServerError("policy_invalid_response", "Codex completed without a final JSON response")
            usage = params.get("usage") or turn.get("usage") or {}
            return answer, usage if isinstance(usage, dict) else {}

    # ----------------------------------------------------------------- #
    # JSON-RPC plumbing
    # ----------------------------------------------------------------- #

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            # Deliberately not through _next_message: a notification parked in the
            # backlog would be handed straight back and the loop would spin.
            try:
                message = self._messages.get()
            except queue.Empty as error:  # get() without a timeout cannot normally raise
                raise AppServerError("app_server_closed", "Codex App Server response queue closed") from error
            if isinstance(message, BaseException):
                raise AppServerError("app_server_closed", str(message))
            if message.get("id") != request_id:
                if "id" in message:
                    # A response to a request we never sent. Nothing to do with
                    # this call, and keeping it would only grow the list.
                    continue
                self._backlog.append(message)
                continue
            if "error" in message:
                raise AppServerError("app_server_rpc", f"{method} failed: {message['error']}")
            result = message.get("result")
            if not isinstance(result, dict):
                raise AppServerError("app_server_protocol", f"{method} returned a non-object result")
            return result

    def _next_message(self, timeout_s: float | None) -> dict[str, Any]:
        if self._backlog:
            return self._backlog.pop(0)
        try:
            message = self._messages.get(timeout=timeout_s)
        except queue.Empty as error:
            raise AppServerError("policy_timeout", "Timed out waiting for Codex App Server") from error
        if isinstance(message, BaseException):
            raise AppServerError("app_server_closed", str(message))
        return message

    def _send(self, message: dict[str, Any]) -> None:
        if self.process is None or self._stdin is None or self.process.poll() is not None:
            raise AppServerError("app_server_closed", "Codex App Server is not running")
        self._stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self._stdin.flush()

    def _read_stdout(self, stream: Any, messages: queue.Queue) -> None:
        """Read one process's stdout into the queue that was created for it.

        The queue is an argument rather than ``self._messages`` on purpose. A
        restarted server gets a fresh queue so that it does not read the previous
        process's end-of-stream marker first, but this thread outlives that swap:
        it wakes up when the old process exits, which can be after ``start()`` has
        already installed the new queue. Resolving the attribute here would make
        the dying thread post its marker -- and any last line still in the pipe --
        into the new process's queue, where the next turn reads it as "the server
        closed" instead of the answer that was actually sent.
        """
        try:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.put(json.loads(line))
                except json.JSONDecodeError as error:
                    messages.put(AppServerError("app_server_protocol", f"Invalid App Server JSON: {error}"))
        finally:
            messages.put(EOFError("Codex App Server stdout closed"))

    @staticmethod
    def _drain_stderr(stream: Any) -> None:
        # Drained rather than left unread: a full pipe would block the child.
        if stream is not None:
            for _ in stream:
                pass
