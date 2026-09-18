# xvla_agent deployment

Use the same model checkpoint and Python environment as `X_VLA`. Start the
standalone `XPolicyLab/codex_agent` Bridge first, then launch this policy:

```bash
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/xvla_agent \
  --task stack_blocks \
  --ckpt <checkpoint> \
  --policy-env XVLA \
  --eval-num 1
```

This policy supports one rollout at a time because its Codex review history is a
single stateful thread. Task descriptions and episode budgets come from
`tasks/<task>.json`; the Bridge does not define them.
