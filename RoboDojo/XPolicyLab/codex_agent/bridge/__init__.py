"""Local policy service: a persistent Codex App Server behind one HTTP endpoint.

This package is transport, validation and recording, and nothing else. It owns no
policy text of its own: the embodiment contract lives in ``workspace/AGENTS.md``
and the decision procedure in ``workspace/.agents/skills/codex_agent/SKILL.md``,
both read by Codex itself from the workspace. The GPU-side adapter owns the
strategy -- the guardrail, the interpolation, the step budget -- and sends a
structured observation; this package turns that into one turn of text, hands the
images through unchanged, and writes down what happened.

Stdlib only, deliberately: it runs on the operator's machine, not in the policy
server's conda environment, so it must not need numpy or PIL.
"""
