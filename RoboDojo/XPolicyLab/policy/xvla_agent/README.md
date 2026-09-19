# xvla_agent

`xvla_agent` is a copy of the validated `X_VLA_OPT` policy adapter with one bounded
integration: after X-VLA produces its EEF action chunk, the policy sends the
current observation and proposal to the standalone `XPolicyLab/codex_agent`
Bridge and executes the final chunk returned by that service.

The original `policy/X_VLA_OPT` directory is unchanged. Model loading, observation
encoding, inference and existing X-VLA post-processing remain in this copy.

```text
observation → X-VLA H30 → Codex Bridge → final action chunk → simulator
```

The Bridge usually passes a proposal through without invoking Codex. It invokes
Codex for a gripper-change proposal or when the previous response requested
fresh-image verification.

Set `CODEX_BRIDGE_URL` to override `bridge_url` in `deploy.yml`.
