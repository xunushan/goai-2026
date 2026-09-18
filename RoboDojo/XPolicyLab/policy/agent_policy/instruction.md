# agent_policy

This directory is the RoboDojo-side adapter for the independent
`XPolicyLab/codex_agent` robot harness.

The policy server owns task descriptions and budgets, converts simulator
observations into the Bridge request, and interpolates the returned absolute EEF
target into an action chunk. The Bridge URL is configured by `CODEX_BRIDGE_URL`
or `deploy.yml`.

```bash
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/agent_policy \
  --task stack_blocks \
  --ckpt none \
  --policy-env XVLA \
  --eval-num 1
```

The Bridge can run on the same server at `127.0.0.1:8765` or on another host
through a tunnel. No Codex runtime, workspace, skill or experience data is
loaded by this policy package.
