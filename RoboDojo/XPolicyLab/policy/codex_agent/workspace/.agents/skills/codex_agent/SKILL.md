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

The current camera images are already attached to the turn. Inspect them directly;
do not use shell commands to reopen, crop, resize, tile, draw grids on, or repeatedly
measure the same images. Do not run ad-hoc colour cut-offs, masks, pixel scans, or
uncalibrated pixel-to-world calculations. Their apparent precision is misleading.

Use `view_image` only for a specific older cached frame when the current views and
text history leave one concrete ambiguity. Inspect at most two such frames, once
each, then decide. Use the shell only to list or read those explicitly named image
files; do not explore the repository or read source code, config, task cards,
README files, `rollout.jsonl`, or another episode. Write any necessary temporary
artifact only under the current episode's `scratch/`, never `/tmp`. Tool use must
support the immediate motion decision, not become a separate vision pipeline.

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
- Stay inside the per-decision motion scale stated in `AGENTS.md`; split longer
  travel across multiple observation-driven decisions.

## 4. Coordinate motion with grasp and release

Prefer to finish approach and alignment before changing the gripper. During most
travel use `"keep"`; a target you submitted is not proof that the arm arrived or
that contact is safe. A short final motion may change the gripper at the same time
when the manipulation itself requires it, such as closing during the final
downward grasp. Use that only after fresh views establish alignment and clearance.

After positioning, inspect the next observation before changing the gripper. Use
the active arm's wrist close-up for the jaws and nearby object, and the head view
to check the wider scene, clearance, and support. Confirm that the observed arm
state is consistent with the intended pose. Normally issue a separate gripper
decision with position and orientation set to `"keep"`; combine them only for the
short contact motion described above.

Never close the gripper merely to probe alignment or estimate object width. The
gripper number does not provide that information. Close only after the fresh head
and wrist views show that the object is between the fingers at the intended grasp
depth.

Before opening, verify from the fresh views that the object is fully supported by
the destination or the other arm. After opening, observe once more before moving
away. If the object still appears trapped by the fingers or unsupported, keep it
supported and disengage deliberately; do not withdraw and declare success.

## 5. Read the feedback from your last decision

The turn text states whether the previous decision ran or failed. Read it together
with the fresh measured arm state before repeating a request. If a target was
refused, fix the request rather than sending it again.

## 6. Watch the budget

The turn text states how many decisions and simulator steps remain. Travel and
reposition early while the budget is large; reserve the last decisions for precise
alignment and for the ending the success condition requires.

## 7. Reply with exactly one JSON object

No prose before or after it, no Markdown fence, no explanation, and never a second
action:

```json
{"left":  {"position": "keep" | [x, y, z],
           "orientation": "keep" | [w, x, y, z],
           "gripper": "keep" | "open" | "close"},
  "right": {"position": "keep" | [x, y, z],
           "orientation": "keep" | [w, x, y, z],
           "gripper": "keep" | "open" | "close"},
  "note": "one or two sentences describing what you see and why you chose this motion",
  "phase": "short label"}
```

Quaternion order is exactly `[w, x, y, z]` as stated in `AGENTS.md`.

`note` and `phase` are required and are read by humans. Aim to keep `note` within
30 characters: state only the decisive current evidence and action purpose; omit
pixel coordinates, pose arrays, dimensions, repeated history and tool mechanics.
`phase` must not exceed 10 characters and should be a label such as `approach`,
`grasp`, `insert`, or `retreat`.
