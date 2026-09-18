# codex_bridge — the policy service for `codex_agent`

`codex_agent` is a **stateful Codex session**: one thread that sees a picture of
the robot, says where to move next, sees the result, and decides again. The policy
server runs on the GPU machine, but the Codex CLI and its credentials live on the
operator's Mac, so this small HTTP server sits in between.

```
Mac                                        GPU machine
┌────────────────────────────────────┐     ┌──────────────────────────────────┐
│ codex_bridge :8765                 │     │ codex_agent policy server        │
│   └ codex app-server --stdio       │◄─R──┤   └ dials localhost:8765         │
│     cwd = codex_agent/workspace    │     │       sim client (ws://…)        │
└────────────────────────────────────┘     └──────────────────────────────────┘
```

The division of labour is the point of the layout. **The GPU side owns strategy**:
it reads the simulator, enforces the guardrail, turns an accepted target into an
action chunk, and accounts for the step budget. **This service owns transport**:
it turns a structured observation into one turn of text, hands the images through
unchanged, and writes down what happened. It holds no policy text — the embodiment
contract is `workspace/AGENTS.md` and the decision procedure is
`workspace/.agents/skills/codex_agent/SKILL.md`, both read by Codex itself.

Everything below runs **on the Mac**, inside `codex_agent/`. Two processes: the
bridge, and a tunnel.

## 1. Start the bridge

```bash
cd RoboDojo/XPolicyLab/policy/codex_agent
/Applications/ChatGPT.app/Contents/Resources/codex --version   # see the version note below
python3 -m bridge.bridge --quiet
```

Run it **from `codex_agent/`**: the package is imported as `bridge.*`, and
`--workspace` defaults to `codex_agent/workspace`, resolved relative to this
package rather than to the current directory. Pass `--workspace` only if the
workspace lives somewhere else.

`--codex-bin` defaults to `codex` on `PATH`. On a Mac with the ChatGPT app the CLI
is inside the bundle and there is no `codex` on `PATH`, so either pass
`--codex-bin /Applications/ChatGPT.app/Contents/Resources/codex` or put a symlink
on `PATH`.

Defaults: binds `127.0.0.1:8765`, 75 s per decision (90 s for the one that opens a
thread, which also has to read the workspace), and rotation every 8 image-bearing
turns. Useful flags: `--model`, `--reasoning-effort`, `--port`, `--token`,
`--workspace`, `--experience-library`, `--max-live-image-turns`, `--timeout-s`,
`--timeout-first-turn-s`.

`python3 -m bridge.bridge --help` is the authority on the flag list; there is no
config file.

**Version note.** This bridge speaks `codex app-server --stdio`, the JSON-RPC
protocol behind the app server. `app-server` is marked `[experimental]` in
`codex --help`, so the protocol can move between CLI versions. Verified against
`codex-cli 0.154.0-alpha.6.2`. Re-check after upgrading the CLI; the failure mode
of a changed protocol is a bridge that starts and then fails every turn.

## 2. Open the tunnel

From the Mac, in a second terminal:

```bash
tools/tunnel_mac.sh issac-server
```

That is `ssh -N -R 8765:localhost:8765 issac-server`. The direction matters — the
policy server dials `localhost:8765` *on the GPU machine*, so the remote sshd has
to listen there and forward back here. `-L` would point the wrong way and would
also expose the bridge to the remote network.

Check it from the GPU machine:

```bash
curl -s http://localhost:8765/healthz | python -m json.tool
```

`{"ok": true, ...}` means the whole chain is up. Do this **before** starting the
episode: an episode that begins with an unreachable bridge burns its decision
budget on holds and scores zero.

## 3. Point the policy server at it

Nothing to do when the port is 8765. To use a different host or port, start the
policy server with `CODEX_BRIDGE_URL` set:

```bash
export CODEX_BRIDGE_URL=http://localhost:9000
```

Precedence: `CODEX_BRIDGE_URL` → flat `bridge_url` in `deploy.yml` →
`bridge.base_url` in `deploy.yml`.

## HTTP contract

`GET /healthz` → `{ok, model, reasoning_effort, codex_bin, workspace, episode_id,
cameras, max_live_image_turns, timeout_s, timeout_first_turn_s, stats}`.

`POST /v1/decide` — one observation in, one decision out:

```jsonc
{
  "episode_id": "ep-1",
  "request_id": "ep-1-000042",        // identifies this turn in the log; retries reuse it
  "step_id": 42,                      // simulator step the observation was taken at
  "turn_index": 41,                   // 0-based decision counter
  "task": {"name": "stack_bowls", "instruction": "…"},
  "budget": {"max_decisions": 100, "max_sim_steps": 550,
             "remaining_decisions": 59, "remaining_steps": 320},
  "observation": {
    "left":  {"position": [x, y, z], "orientation": [w, x, y, z], "gripper": 1.0},
    "right": {"position": [x, y, z], "orientation": [w, x, y, z], "gripper": 1.0}
  },
  "feedback": ["…"],                  // prose about the previous decision, written on the GPU side
  "images": [{"name": "cam_head", "mime": "image/jpeg", "b64": "…"}]
}
```

`orientation` is the measured absolute quaternion in `wxyz` order. The bridge
does no frame conversion.

Two checks on the way in. Images are validated by **magic bytes**, not by the
declared `mime`, so a truncated payload is rejected rather than reaching Codex as
a corrupt attachment; and the `images[].name` list must equal the camera list
`workspace/AGENTS.md` names, in order, because the turn text lists those views by
name and a mismatch would hand the model three pictures under labels that
contradict them.

→ `200 {ok, decision}`

```jsonc
"decision": {
  "left":  {"position": [x, y, z], "orientation": [w, x, y, z], "gripper": "keep"},
  "right": {"position": "keep", "orientation": "keep", "gripper": "open"},
  "note": "moving the left arm towards the socket",
  "phase": "reach"
}
```

`decision` is the JSON object the model produced in exactly the shape the
adapter's `protocol.parse_decision` understands.

Failures come back as `{ok: false, error_kind, error, latency_ms, record_dir}` with
`400` for a request we will not act on, `504` for a timeout, `502` for anything
that went wrong behind us, and `500` for a bridge bug. **Every path returns an
HTTP response** — the episode degrades to a hold rather than hanging, because a
120 s websocket timeout on the eval client discards the whole episode.

| `error_kind` | meaning |
| --- | --- |
| `bad_request` | the packet is malformed; the reason is spelled out |
| `bridge_internal_error` | a bug in the bridge |
| `policy_timeout` | no final answer inside the wall clock |
| `policy_overloaded` | `serverOverloaded`; retryable |
| `policy_usage_exhausted` | `usageLimitExceeded`; not retryable |
| `policy_turn_failed` | the turn ended in a failure status |
| `policy_invalid_response` | the turn completed with no parseable JSON object |
| `app_server_closed` | the subprocess died or its stdout ended |
| `app_server_start_failed` | the subprocess could not be launched |
| `app_server_not_bound` | no episode bound, so no workspace to run in |
| `app_server_protocol` | a response that does not fit the protocol |
| `app_server_rpc` | the server answered a request with an error |

`policy_overloaded` and `policy_usage_exhausted` are kept distinct on purpose: the
adapter retries the first and gives up on the second. The client adds two kinds of
its own for trouble that happens before a reply exists: `bridge_unreachable` and
`bridge_bad_response`.

## The workspace and the audit trail

`workspace/` is the Codex workspace: `AGENTS.md`, the skill, and `.codex/config.toml`
for a human running `codex` by hand. The bridge does not rely on that config file —
a project-local `.codex/config.toml` is only honoured once the workspace is trusted,
so the bridge passes the same permissions with `-c` at launch:

- read the workspace and the current episode's `observations/`, but not
  `rollout.jsonl` or other episodes;
- write the current episode's `scratch/`, and nothing else;
- `shell_tool` and `view_image` enabled; plugins, apps, recommendations, web
  search, computer use and multi-agent tools disabled;
- only the workspace `codex_agent` skill exposed to the model.

Because permissions are fixed when the process starts, **a new episode means a new
process** — the write scope names the episode directory. Episodes run serially, so
that happens once per episode rather than once per decision.

Everything the bridge receives is written down under `workspace/output/<episode_id>/`:

| path | contents |
| --- | --- |
| `rollout.jsonl` | one line per turn: request ids, the decision, usage, latency, errors |
| `observations/<camera>/NNNNNN_<request_id>.<jpg\|png>` | the images, **byte for byte as received** |
| `scratch/` | the one place the model may write |

Images are stored at the resolution they arrived in: nothing is cropped, rescaled
or re-encoded, so a recorded episode is evidence of exactly what the model was
shown rather than a lossy copy of it. The suffix comes from the sniffed magic
bytes. Re-sending a `request_id` with identical bytes is an idempotent retry;
re-sending it with *different* bytes is refused rather than overwriting evidence.
A failed turn still gets a line, and it carries no `decision` — an invented action
in the log would be indistinguishable from one the model actually chose.

Reading a failed episode back is the point of all this: `rollout.jsonl` shows what
the model decided and why, without spending another call.

## Thread rotation

The model remembers its own thread, and images accumulate in it with no way to
delete them. After `--max-live-image-turns` image-bearing turns (8 by default, the
same window the reference controller uses) the thread is dropped and a new one is
opened. The complete earlier observation text and structured decisions are replayed;
only the most recent turn retains live image encodings. Dropping the thread and
replaying old images as text is the mechanism for forgetting image payloads without
discarding the episode history. `0` disables rotation.

The thread itself never crosses the network. The adapter sends an episode id; the
bridge decides whether that is the id the current thread was opened for. A caller
that has to round-trip a thread id is a caller that can send back the wrong one.

## Experience library

`experience_library/` is host-owned and deliberately outside `workspace/`, so the
agent cannot browse or repeatedly load demonstrations. `index.json` maps an exact
task name to one successful `demo.json` and declares the camera views for each
stage; an unmapped task simply receives no demo.
The bridge injects the selected demonstration before the current observation only
when a Codex thread opens. Thread rotation injects it again before replaying the
episode history, so rotation removes neither the prior history nor the demonstration.

Each task directory keeps the full JSON and every source image. The shipped
`stack_bowls` configuration uses only `cam_high` by default and overrides `grasp`
and `place` to use all three cameras. This relation lives in `index.json`, not in
Python. Images are passed to App Server as data URLs; filesystem paths never enter
the model context. Rendered App Server items are cached after their first load;
thread rotation reuses that in-memory result instead of reading and rendering the
demo again. Use
`tools/render_experience.py experience_library/<task>/demo.json` to inspect the exact
generated text without calling Codex.

## Operational notes

- **Timeout doubles as the kill switch.** The App Server runs in its own session
  (`start_new_session`); on timeout the whole group gets SIGTERM, then SIGKILL.
  The 120 s websocket timeout on the eval client is the real ceiling, so bridge
  timeouts (75 s / 90 s) stay well inside it.
- **Quota.** Each decision is one real Codex call, and a full episode at 100 calls
  is 45-50 minutes. Nothing in `tests/` calls Codex: `tests/fake_app_server` stands
  in for the CLI and `tests/mock_bridge.py` stands in for this server, so the
  allowance is spent only on real runs.
- **Token.** With `--token` (or `CODEX_BRIDGE_TOKEN`) every request must carry
  `X-Bridge-Token`; the policy server reads the same variable. Optional on a
  reverse tunnel, since the bridge is loopback-only on both ends.
