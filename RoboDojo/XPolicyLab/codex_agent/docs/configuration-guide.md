# Codex robot harness configuration

Configuration is split by ownership. Policy adapters configure task-facing
behavior and transport. The standalone Bridge configures Codex, context lifetime,
permissions and motion synthesis.

## Experience example switch

Both `policy/agent_policy/deploy.yml` and `policy/xvla_agent/deploy.yml` use:

```yaml
experience:
  enabled: true
```

- `true`: on the first Codex thread of an episode, the Bridge looks up the exact
  `task_name` in the active experience library's `index.json` and injects that
  demonstration. If the task has no index entry, nothing is injected.
- `false`: no demonstration is read or injected. Normal observation and decision
  history is still retained and replayed when the thread rotates.

The value is fixed for an episode. Changing it during an episode is rejected so
one rollout cannot silently change its context halfway through. Restart/reset the
episode after changing the policy configuration.

Simulation and real-robot demonstrations are isolated libraries:

```text
codex_agent/experience_library/
├── sim_experience_library/
│   ├── index.json
│   ├── stack_bowls/
│   └── plug_in_charger/
└── real_experience_library/
    ├── index.json
    ├── fill_pen_holder/
    └── ...
```

`index.json` is the only routing table. Merely placing another directory in the
library does not cause it to be loaded.

When the first Codex thread opens, Bridge stdout prints one audit line:

```text
[bridge][experience] episode=... task=stack_bowls status=loaded demo=stack_bowls/demo.json keyframes=5 images=7
```

`status` is one of `disabled`, `no_match`, `loaded`, or `error`. The same object
is stored as `experience` in every `workspace/output/<episode>/rollout.jsonl`
record. Before a Codex turn actually opens it may be `pending`; this commonly
occurs when VLA review routing accepts a chunk without calling Codex.

## Bridge startup parameters

Run from `RoboDojo/XPolicyLab/codex_agent`:

```bash
python3 -m bridge.bridge \
  --codex-bin codex \
  --model gpt-6-astra \
  --reasoning-effort medium \
  --experience-library experience_library/sim_experience_library \
  --max-live-image-turns 8
```

Use `--experience-library experience_library/real_experience_library` for the
real robot. The selected path is fixed for the Bridge process; restart the
Bridge to switch environments. `GET /healthz` reports the resolved active path.

Important parameters:

| Parameter | Default | Purpose |
|---|---:|---|
| `--model` | Codex default | Model used by App Server |
| `--reasoning-effort` | `medium` | Reasoning strength: `low`, `medium`, `high`, or `max` |
| `--timeout-s` | `75` | Normal turn wall-clock timeout in seconds |
| `--timeout-first-turn-s` | `90` | New-thread timeout, including workspace/skill loading |
| `--max-live-image-turns` | `8` | Image-bearing turns before thread rotation; `0` disables rotation |
| `--workspace` | `codex_agent/workspace` | AGENTS.md, skills, output and scratch workspace |
| `--experience-library` | `codex_agent/experience_library/sim_experience_library` | Active simulation or real demonstration library root |
| `--host` / `--port` | `127.0.0.1` / `8765` | Bridge listen address |
| `--token` | environment token | Optional HTTP authentication token |

The Bridge passes `--reasoning-effort` into every App Server turn. This is the
effective runtime setting and takes precedence over the workspace default
`model_reasoning_effort` in `workspace/.codex/config.toml`. Use the Bridge flag
when running the robot policy; do not edit the workspace file for each run.

## Motion parameters

Motion interpolation is owned by the Bridge, not the policy service:

| Parameter | Default |
|---|---:|
| `--delta-p-max-m` | `0.005` |
| `--delta-theta-max-rad` | `0.035` |
| `--max-target-translation-m` | `0.05` |
| `--max-target-rotation-rad` | `0.35` |
| `--settle-steps` | `3` |
| `--gripper-open` | `1.0` |
| `--gripper-close` | `0.0` |

`GET /healthz` reports the active model, reasoning effort, image window, timeouts,
workspace, experience-library path and motion configuration. Check it before a
rollout:

```bash
curl -s http://127.0.0.1:8765/healthz | python3 -m json.tool
```

## Policy-side parameters

`agent_policy/deploy.yml` owns the Bridge URL/request timeout, JPEG transport
quality, experience switch and IO logging. `xvla_agent/deploy.yml` additionally
owns X-VLA inference, chunk, gripper and sparse-review routing parameters.

The image `jpeg_quality` is transport compression performed before HTTP upload.
Image storage paths and conversation image-window handling remain Bridge-owned.
