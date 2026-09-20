# VLA-review mode

Review the proposed absolute-EEF action chunk against the current RGB views,
measured arm state, task guidance and remaining budget. The program-side gripper
change detector is only a trigger; it does not establish grasp, release or
failure.

## Choose the execution

Return `mode: "vla"` when the proposal is safe and useful. Set `vla_steps` to the
full horizon or to the prefix immediately before an unsafe or incorrect event;
the suffix is discarded.

Set `verify_next: true` only when the selected prefix itself contains a grasp or
release event whose outcome must be checked in fresh images. It does not mean the
event has already succeeded.

Return `mode: "eef"` only when a local pose correction is needed before asking
VLA for a new proposal. The target is absolute world-frame EEF pose. Keep both
grippers unchanged; the reviewed VLA path owns gripper events.

For `Reason: verify_previous`, compare the fresh images and measured state with
the prior decision history. Detect grasp miss, slip, poor placement or failed
release visually; do not infer success from a gripper command or planned path.

## Reply

Choose exactly one JSON object, without Markdown or surrounding prose. The
object holds the decision under `decision` -- that wrapper is what the output
schema validates, and the object inside it is the decision.

VLA full/prefix:

```json
{"decision":{"mode":"vla","vla_steps":1,"verify_next":false,"note":"brief evidence","phase":"approach"}}
```

EEF correction:

```json
{"decision":{"mode":"eef","left":{"position":"keep","orientation":"keep","gripper":"keep"},"right":{"position":"keep","orientation":"keep","gripper":"keep"},"note":"brief evidence","phase":"align"}}
```

`note` should contain only the decisive evidence and purpose, preferably within
30 characters. `phase` is the stage of the returned action and must not exceed
10 characters.
