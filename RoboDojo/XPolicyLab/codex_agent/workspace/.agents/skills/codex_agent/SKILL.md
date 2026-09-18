---
name: codex_agent
description: Choose or review the next end-effector action for a dual-arm robot from current RGB views, measured arm state, task context and remaining budget.
---

# Robot policy decision

Use this skill for every decision. Read `AGENTS.md` first; it is the embodiment
contract for coordinates, arms, EEF pose and gripper semantics.

Base the decision on the current images and measured state. A previous command is
not evidence that motion, grasp or release succeeded. Keep the note concise and
make `phase` describe the action being returned, not the scene before it.

## Route by request

- If the turn has no `VLA REVIEW` section, read
  [references/codex-only.md](references/codex-only.md) and return an absolute EEF
  target.
- If the turn has a `VLA REVIEW` section, read
  [references/vla-review.md](references/vla-review.md) and either select a VLA
  prefix or return an absolute EEF correction.

Read exactly one reference for the current mode.
