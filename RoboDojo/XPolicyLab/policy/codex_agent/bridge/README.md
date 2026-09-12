# codex_bridge — the local half of `codex_agent`

The `codex_agent` policy is a **stateful Codex thread**. The policy server runs
on the GPU machine, but the Codex CLI and its credentials live on the operator's
Mac, so this small HTTP server sits in between: it owns the Codex subprocess,
remembers the thread id across turns, and answers the policy server over the
tunnel.

```
Mac                                        GPU machine
┌──────────────────────────┐               ┌──────────────────────────────┐
│ codex_bridge :8765       │               │ codex_agent policy server    │
│   └ codex exec / resume  │◄── ssh -R ────┤   └ dials localhost:8765     │
│     (gpt-6-astra)        │               │       sim client (ws://…)    │
└──────────────────────────┘               └──────────────────────────────┘
```

Everything below runs **on the Mac**. Two processes: the bridge, and a tunnel.

## 1. Start the bridge

```bash
cd RoboDojo/XPolicyLab/policy/codex_agent/bridge
python3 server.py --quiet
```

Defaults: binds `127.0.0.1:8765`, Codex at
`/Applications/ChatGPT.app/Contents/Resources/codex`, model read from
`~/.codex/config.toml`, read-only sandbox, 75 s per turn (90 s for the first,
which pays the CLI's cold start). Useful flags: `--model`, `--reasoning-effort`,
`--port`, `--token`, `--runs-dir`, `--workdir`, `--timeout-s`.

Nothing but a tunnel can reach it: the bind address is loopback, and the sandbox
is read-only, so a confused model cannot edit files on the Mac.

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
policy server with `CODEX_BRIDGE_URL` set (it is forwarded as a `bridge_url`
override):

```bash
export CODEX_BRIDGE_URL=http://localhost:9000
```

Precedence: `CODEX_BRIDGE_URL` → flat `bridge_url` in `deploy.yml` →
`bridge.base_url` in `deploy.yml`.

## HTTP contract

`GET /healthz` → `{ok, model, codex_bin, workdir, runs_dir, timeout_s, stats}`.

`POST /v1/decide`

```jsonc
{
  "episode_id": "ep-1",
  "thread_id": null,            // null on the first turn; the bridge creates and returns one
  "turn_index": 0,
  "prompt": "…",                // the turn prompt; the system prompt rides in it too
  "schema": { … },              // optional JSON schema for --output-schema
  "images": [{"name": "cam_head", "format": "jpeg", "b64": "…"}]
}
```

Images are validated by **magic bytes**, not by the declared `format`, so a
truncated payload is rejected with `bad_request` instead of reaching Codex as a
corrupt attachment. Per-image and total size caps apply.

→ `200 {ok, thread_id, parsed, text, usage, latency_ms, runs_dir, error_kind, timed_out}`

`parsed` is the decision object the model produced, already stripped of fences
and prose. `thread_id` must be echoed back on the next turn — that is what makes
the agent stateful across processes. Failures come back as `{ok: false,
error_kind, error, …}` with `502` for Codex-side problems and `500` for bridge
bugs; either way the policy server degrades to a hold rather than raising.

`error_kind` values, as the bridge reports them: `codex_timeout`,
`codex_unparseable` (a reply that is prose, not a JSON object),
`codex_empty_reply`, `codex_failed`, `codex_spawn_failed`, `thread_not_found`,
`bad_request`, `bridge_internal_error`. The client adds two of its own for
trouble that happens before a reply exists: `bridge_unreachable` and
`bridge_bad_response`. `bad_request` arrives as `400` with the reason spelled
out; everything else from Codex arrives as `502`. A stale `thread_id`
(`thread_not_found`) is rebuilt once from scratch.

## Audit trail

Every turn is written to `~/codex_bridge/runs/<episode_id>/turn_NNN/`:

| file | contents |
| --- | --- |
| `prompt.txt` | exactly what the model was asked |
| `images/` | the JPEGs that were attached |
| `stdout.jsonl` | the raw `--json` event stream |
| `last_message.txt` | Codex's final message |
| `result.json` | parsed decision, usage, latency, errors |

These are the primary debugging artefact for a failed episode: read the
`result.json` chain to see what the model decided and why, without spending
another call.

## Operational notes

- **Timeout doubles as the kill switch.** Codex runs in its own session
  (`start_new_session`); on timeout the whole group gets SIGTERM, then SIGKILL
  after 2 s. A turn that overruns still produces an HTTP response — the episode
  degrades to a hold, it never hangs. The 120 s websocket timeout on the eval
  client is the real ceiling, so bridge timeouts (75 s / 90 s) stay well inside
  it.
- **One thread per episode.** The policy server sends `thread_id: null` when the
  simulator reports a new `episode_idx`; the bridge creates a fresh thread and
  the old thread is simply dropped.
- **Quota.** Each decision is one real Codex call and they are expensive (~25-30 s,
  substantial tokens). The policy allows at most `episode.max_codex_calls`
  (default 10) per episode. Nothing in `tests/` calls Codex — `tests/fake_codex`
  stands in for the CLI and `tests/mock_bridge.py` stands in for this server — so
  the allowance is spent only on real runs.
- **Token.** With `--token` (or `CODEX_BRIDGE_TOKEN`) every request must carry
  `X-Bridge-Token`; the policy server reads the same variable. Optional on a
  reverse tunnel, since the bridge is loopback-only on both ends.
