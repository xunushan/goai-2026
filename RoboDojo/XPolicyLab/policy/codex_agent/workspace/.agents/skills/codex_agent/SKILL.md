---
name: codex_agent
description: Choose the single next end-effector motion for the dual-arm robot from the current camera views and observed arm state, and reply with one JSON object.
---

# Deciding the next motion

Use this skill for every decision. It is the whole procedure; do not improvise a
different one.

## 1. Read the embodiment contract first

`AGENTS.md` at the workspace root states what the robot is, what `position` and
`orientation` mean, what the gripper number means, and which arm is which. Read it
before your first decision. Everything you emit is interpreted through it.

## 2. Look before you move

Start from the images and the observed arm state, not from your plan. Say in `note`
what you actually see in the current views. If the scene does not match what you
expected from your last decision, trust the images and the numbers, and re-plan --
the previous decision is not evidence about the present.

## 3. Move in small deliberate steps

Ask for one motion that advances the task, then look again.

- Plan the path against both the remaining simulator-step budget and the remaining
  decision budget. Preserve enough budget for final alignment and verification.
- The controller executes the returned action chunk. It stops short of the target
  only if the remaining simulator-step budget caps that chunk; re-observe before
  deciding what to do next.
- Use `"keep"` for every component that does not need to change. Keep is free.
- Do not reason about speed, duration or interpolation. A deterministic controller
  owns execution of the requested target.
- Stay inside the per-decision limit stated in `AGENTS.md`. A target beyond it is
  refused outright, the arm does not move at all, and the decision still counts
  against your budget -- so an over-large request costs you two decisions, not one.

## 4. Read the feedback from your last decision

The turn text states the outcome of your previous motion: whether it was carried
out, whether it was refused or clamped, and how far the arms actually got. Read it
before repeating a request. If something was refused, the reason is stated; fix the
request rather than sending it again.

## 5. Watch the budget

The turn text states how many decisions and simulator steps remain. Travel and
reposition early while the budget is large; reserve the last decisions for precise
alignment and for the ending the success condition requires.

## 6. Reply with exactly one JSON object

No prose before or after it, no Markdown fence, no explanation, and never a second
action:

```json
{"left":  {"position": "keep" | [x, y, z],
           "orientation": "keep" | [yaw, pitch, roll],
           "gripper": "keep" | "open" | "close"},
  "right": {"position": "keep" | [x, y, z],
           "orientation": "keep" | [yaw, pitch, roll],
           "gripper": "keep" | "open" | "close"},
  "note": "one or two sentences describing what you see and why you chose this motion",
  "phase": "short label"}
```

If you need an exact absolute orientation you may give `"orientation"` as
`{"quat": [w, x, y, z]}` instead of yaw/pitch/roll.

`note` and `phase` are required and are read by humans: `note` is what you see and
why, `phase` is a short label such as `approach`, `grasp`, `insert`, `retreat`.
