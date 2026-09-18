# agent_policy

`agent_policy` is the thin RoboDojo policy adapter for the Codex robot harness.
It owns task configuration, episode/call/step budgets, observation extraction,
the Bridge HTTP client and deterministic EEF interpolation.

It does not start Codex, load skills, manage App Server threads or access the
experience library. Those responsibilities belong to the sibling
`XPolicyLab/codex_agent` harness.

Runtime flow:

```text
RoboDojo observation
→ build structured request
→ POST codex_agent Bridge
→ parse absolute EEF target
→ deterministic LERP/SLERP action chunk
→ simulator
```

Run the policy-only tests with `bash tests/run_all.sh`.
