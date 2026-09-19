# Codex robot harness

`codex_agent` is the policy-independent robot harness. It exposes an HTTP bridge,
owns the Codex App Server process and conversation history, and provides the
workspace, skills and experience library used by the robot agent.

It does not import a RoboDojo policy or a VLA model. Policy services send a
structured observation packet to `POST /v1/decide` and consume the structured
decision returned by the Bridge.

## Layout

```text
codex_agent/
├── bridge/              HTTP bridge, schema, App Server client and recording
├── workspace/           AGENTS.md and robot policy skill
├── experience_library/  task-indexed successful examples
├── docs/                protocol and design documents
├── tests/               harness-only offline tests
└── tools/               harness utilities
```

## Run

From `RoboDojo/XPolicyLab/codex_agent`:

```bash
python3 -m bridge.bridge --quiet --codex-bin codex
```

The default endpoint is `http://127.0.0.1:8765`. The Bridge starts
`codex app-server --stdio` as a child process. Run `bash tests/run_all.sh` for
offline verification.

See [`docs/configuration-guide.md`](docs/configuration-guide.md) for the policy
experience switch, model/reasoning settings, timeouts, image window and motion
parameters.

The Bridge may run beside a policy server or on another host reached through an
SSH tunnel. The HTTP protocol remains the same in both deployments.
