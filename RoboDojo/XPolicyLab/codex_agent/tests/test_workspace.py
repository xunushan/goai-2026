"""Regression tests for the Codex workspace: ``AGENTS.md`` and the skill.

These two files are the policy's standing instructions. Codex reads them itself,
so nothing in ``bridge/`` or on the GPU side checks their wording -- which means
a careless edit here silently changes how the model behaves, and the first sign
of it is a wasted official run.

Two rules are enforced, both inherited from the operator:

**What the robot is may be stated; how the task is scored may not.** Reward term
names, thresholds, spawn ranges and the benchmark's own identity must never
appear. The same forbidden lists the old prompt test used are reused verbatim,
so nothing that was previously written down got lost in the move.

**Do not state a number we cannot stand behind.** In these files the only numbers
allowed at all are the two endpoints of the gripper scale and the per-decision
travel limit -- that last one because our own controller enforces it and refuses
a decision that exceeds it. Coordinates, ranges, workspace boxes and statistics
from the demonstrations are all out; the model steered by them.

And the drift check that has no other home: the numbers and camera names in
``AGENTS.md`` must equal the ones ``deploy.yml`` actually configures, because
there is no mechanism that keeps two copies of a fact in agreement.

Runs offline: no Codex, no simulator, no network.

    python tests/test_workspace.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

from codex_agent.bridge.bridge import CAMERA_NAMES  # noqa: E402

WORKSPACE = _PKG / "workspace"
AGENTS = WORKSPACE / "AGENTS.md"
SKILL = WORKSPACE / ".agents" / "skills" / "codex_agent" / "SKILL.md"
CONFIG = WORKSPACE / ".codex" / "config.toml"

# Strings that must never reach the model, each with the file it would leak from.
FORBIDDEN_STRINGS = {
    # task/RoboDojo/tasks/plug_in_charger.py
    "is_A_depth_in_B": "reward term name (tasks/plug_in_charger.py)",
    "is_A_in_B": "reward term name (tasks/plug_in_charger.py)",
    "is_axis_up": "reward term name (tasks/plug_in_charger.py)",
    "all_robot_back_to_origin": "reward term name (tasks/plug_in_charger.py)",
    "z_threshold": "reward parameter name (tasks/plug_in_charger.py)",
    "step_lim": "internal step constant (tasks/plug_in_charger.py)",
    # env/reward_manager/reward_manager.py
    "reward_manager": "scoring machinery",
    "RewardManager": "scoring machinery",
    "pos_threshold": "reward parameter name (reward_manager.py)",
    "rot_threshold": "reward parameter name (reward_manager.py)",
    "threshold": "any reward parameter name",
    # task/RoboDojo/tasks/stack_blocks.py
    "is_stacked": "reward term name (tasks/stack_blocks.py)",
    "is_all_gripper_open": "reward term name (tasks/stack_blocks.py)",
    "REWARD_XY_BY_INDEX": "reward parameter table (tasks/stack_blocks.py)",
    "get_score": "scoring entry point (tasks/stack_blocks.py)",
    # task/RoboDojo/config/plug_in_charger.yml
    "xlim": "spawn randomization key (config/plug_in_charger.yml)",
    "ylim": "spawn randomization key (config/plug_in_charger.yml)",
    "rotate_deg": "spawn randomization key (config/plug_in_charger.yml)",
    "rotate_rand": "spawn randomization key (config/plug_in_charger.yml)",
    "relative_plane": "spawn randomization key (config/plug_in_charger.yml)",
    "allow_duplicate": "spawn randomization key (config/plug_in_charger.yml)",
    # task/RoboDojo/config/stack_blocks.yml
    "select_mode": "spawn randomization key (config/stack_blocks.yml)",
    "select_category_nums": "spawn randomization key (config/stack_blocks.yml)",
    "select_instance_nums": "spawn randomization key (config/stack_blocks.yml)",
    "ProhibitedArea": "spawn exclusion region (config/stack_blocks.yml)",
    "hierarchical": "spawn selection mode (config/stack_blocks.yml)",
    # the benchmark's own identity
    "plug_in_charger": "benchmark task name",
    "stack_blocks": "benchmark task name",
    "RoboDojo": "benchmark name",
}

# Restatements of the tolerances in words.
FORBIDDEN_PHRASES = ("10 degree", "20 degree", "within 1.5", "1.5 cm of", "insertion depth")

# Numbers that must never appear: the start pose, the demonstration-derived
# workspace box, and the scoring thresholds. Values, not spellings, so a
# reformatted copy is still caught.
FORBIDDEN_NUMBERS = {
    -0.2995: "the start pose x of the left arm",
    0.3005: "the start pose x of the right arm",
    -0.3523: "the start pose y",
    0.9215: "the start pose z",
    0.707: "a component of the start orientation",
    0.7072: "a component of the start orientation",
    -0.50: "the workspace box's left x bound",
    0.01: "the workspace box's inner x bound",
    -0.41: "the workspace box's y lower bound",
    0.02: "the workspace box's y upper bound",
    1.15: "the workspace box's z upper bound",
    0.015: "z_threshold for the insertion depth (tasks/plug_in_charger.py)",
    0.15: "pos_threshold for returning home (reward_manager.py)",
    0.0175: "xy_threshold for a stacked pair (tasks/stack_blocks.py)",
}

# The only numbers these files are allowed to state, and what each is for. The
# per-decision limit is here because our controller enforces it; the gripper
# endpoint because it is a scale the robot's own reading is expressed in.
ALLOWED_NUMBERS = {
    0.05: "the per-decision travel limit, in metres",
    0.35: "the per-decision rotation limit, in radians",
    1.0: "the open end of the gripper scale",
}

_NUMBER = re.compile(r"-?\d+\.\d+")
_BARE_INF = re.compile(r"\b-?inf(inity)?\b", re.IGNORECASE)
_BARE_NAN = re.compile(r"\bnan\b", re.IGNORECASE)

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL  {label}")


def instructions() -> dict[str, str]:
    return {"AGENTS.md": AGENTS.read_text(encoding="utf-8"), "SKILL.md": SKILL.read_text(encoding="utf-8")}


# --------------------------------------------------------------------------- #


def test_the_files_exist() -> None:
    print("the workspace is present where Codex looks for it")
    for path in (AGENTS, SKILL, CONFIG):
        check(path.is_file(), f"{path.relative_to(_PKG)} exists")
    check(SKILL.parent.name == "codex_agent", "the skill directory is named after its frontmatter name")

    head = SKILL.read_text(encoding="utf-8").split("---")[1]
    check("name: codex_agent" in head, "the skill frontmatter declares its name")
    check("description:" in head, "the skill frontmatter declares a description")


def test_no_benchmark_internals() -> None:
    print("no benchmark-internal string reaches the model")
    for name, text in instructions().items():
        lowered = text.lower()
        for token, why in FORBIDDEN_STRINGS.items():
            check(token.lower() not in lowered, f"[{name}] {token!r} must not appear ({why})")
        for phrase in FORBIDDEN_PHRASES:
            check(phrase not in lowered, f"[{name}] the phrase {phrase!r} must not appear")


def test_only_standable_numbers() -> None:
    print("the only numbers stated are the ones we can stand behind")
    for name, text in instructions().items():
        for token in _NUMBER.findall(text):
            value = float(token)
            check(
                value in ALLOWED_NUMBERS,
                f"[{name}] {token} is not a number these files may state "
                f"(allowed: {sorted(ALLOWED_NUMBERS)})",
            )
            check(
                abs(value - 0.0) > 1e-9,
                f"[{name}] {token} is a zero written with decimals",
            )
            for forbidden, why in FORBIDDEN_NUMBERS.items():
                check(
                    abs(value - forbidden) > 1e-9,
                    f"[{name}] the number {value} must not appear ({why})",
                )
        check(not _BARE_INF.search(text), f"[{name}] never renders an infinity")
        check(not _BARE_NAN.search(text), f"[{name}] never renders a NaN")
        check("-0.00" not in text, f"[{name}] has no negative zero")

    agents = instructions()["AGENTS.md"]
    check("0.05 m" in agents, "the travel limit is stated in metres")
    check("0.35 rad" in agents, "the rotation limit is stated in radians")
    check("1.00 is fully open" in agents, "the gripper scale's open end is stated")

    # The closed end is deliberately not a number. What the jaws stop at depends
    # on what is between them, so quoting a value over-claims a mechanical stop.
    squashed = " ".join(agents.lower().split())
    check("fully closed" not in squashed, "the closed gripper is not claimed to be a stop")
    check("does not measure object width" in squashed,
          "the gripper reading is not presented as an object-width sensor")
    check("0.00 is fully closed" not in agents,
          "the configured closed command is not quoted as a mechanical fact")


def test_embodiment_facts_present() -> None:
    print("the facts the model needs are present")
    agents = instructions()["AGENTS.md"]
    for fragment in (
        "1.00 is fully open",
        "+y points forward",
        "+z points up",
        'named\n  "left" works on the -x side',  # the observed working sides
        "end-effector reference point",
        "6 degrees of freedom",
        "parallel-jaw gripper",
    ):
        check(fragment in agents, f"AGENTS.md states {fragment!r}")

    check("+x points forward" not in agents, "the world axes are not stated backwards")
    check("+y left" not in agents, "y is forward, not left")
    check("0.26 m" not in agents, "no borrowed link geometry from another robot")
    check("REACHABLE ENVELOPE" not in agents, "no workspace envelope is stated")
    check("clamped" not in agents, "no clamping promise is stated as geometry")


def test_the_per_decision_motion_scale_is_stated() -> None:
    print("the policy motion scale is stated")
    agents = " ".join(instructions()["AGENTS.md"].split())
    check("0.05 m" in agents, "the translation scale is stated")
    check("0.35 rad" in agents, "the rotation scale is stated")
    check("multiple observation-driven decisions" in agents, "long travel is split by policy")


def test_the_camera_names_are_the_ones_actually_attached() -> None:
    """Two copies of one fact, with nothing else keeping them in agreement."""
    print("the camera views named are the ones the adapter sends")
    check(tuple(CAMERA_NAMES) == ("cam_head", "cam_left_wrist", "cam_right_wrist"),
          "bridge schema exposes the three supported robot camera names")
    check("ATTACHED VIEWS" not in instructions()["AGENTS.md"],
          "the attachment order is per-turn, so it belongs in the turn text, not here")


def test_the_skill_states_the_decision_procedure() -> None:
    print("the skill routes each decision mode to one focused reference")
    skill = instructions()["SKILL.md"]
    for fragment in (
        "AGENTS.md",
        "references/codex-only.md",
        "references/vla-review.md",
        "exactly one reference",
    ):
        check(fragment in skill, f"SKILL.md mentions {fragment!r}")
    references = SKILL.parent / "references"
    check((references / "codex-only.md").is_file(), "codex-only reference exists")
    check((references / "vla-review.md").is_file(), "vla-review reference exists")


def test_the_bridge_does_not_own_policy_text() -> None:
    """The workspace says what the robot is; the transport must not repeat it.

    Only the embodiment contract is checked, not the benchmark's name: a module
    docstring may say which project this belongs to, and forbidding that would
    forbid explaining the code. What must not happen is a second copy of the
    facts the model steers by, because the copies would drift and only one of
    them is the one Codex reads.
    """
    print("the strategy text lives in the workspace, not in the transport")
    for path in sorted((_PKG / "bridge").glob("*.py")):
        lowered = path.read_text(encoding="utf-8").lower()
        for phrase in ("+y points forward", "+z points up", "fully open", "end-effector reference"):
            check(phrase not in lowered, f"[{path.name}] repeats the embodiment contract")


def main() -> int:
    for test in (
        test_the_files_exist,
        test_no_benchmark_internals,
        test_only_standable_numbers,
        test_embodiment_facts_present,
        test_the_per_decision_motion_scale_is_stated,
        test_the_camera_names_are_the_ones_actually_attached,
        test_the_skill_states_the_decision_procedure,
        test_the_bridge_does_not_own_policy_text,
    ):
        test()
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} of {CHECKS} checks:")
        for label in FAILURES[:40]:
            print(f"  - {label}")
        return 1
    print(f"ok: {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
