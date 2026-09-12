"""Unit tests for :mod:`codex_agent.protocol`.

The model is an LLM: it will produce near-miss payloads. These tests pin down
what is accepted, what is quietly repaired, and what is reported back.

    python tests/test_protocol.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

from codex_agent.motion import (  # noqa: E402
    absolute_quat_from_relative_rpy,
    normalize_quat_wxyz,
    quat_angle_between,
    quat_wxyz_to_matrix,
    relative_rpy,
)
from codex_agent.protocol import ParseError, parse_decision  # noqa: E402

HOME_LEFT = np.array([0.7070, 0.0, 0.0, 0.7072])
HOME_RIGHT = np.array([0.7070, 0.0, 0.0, 0.7072])
GRIP_OPEN, GRIP_CLOSE = 1.0, 0.36

_FAILURES: list[str] = []
_CHECKS = 0


def check(condition: bool, label: str) -> None:
    global _CHECKS
    _CHECKS += 1
    if not condition:
        _FAILURES.append(label)
        print(f"  FAIL  {label}")


def parse(payload, **kwargs):
    options = dict(
        home_quat_left=HOME_LEFT,
        home_quat_right=HOME_RIGHT,
        gripper_open=GRIP_OPEN,
        gripper_close=GRIP_CLOSE,
    )
    options.update(kwargs)
    return parse_decision(payload, **options)


def close(a, b, tol=1e-9) -> bool:
    return bool(
        np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=tol, rtol=0)
    )


# --------------------------------------------------------------------------- #


def test_full_payload() -> None:
    print("a well-formed payload")
    d = parse(
        {
            "left": {"position": [-0.2, -0.15, 0.95], "orientation": [0.0, 0.0, 0.3], "gripper": "close"},
            "right": {"position": [0.2, -0.15, 0.95], "orientation": "keep", "gripper": "open"},
            "note": "moving the left arm above the plug",
            "phase": "approach",
        }
    )
    check(d.problems == [], f"no problems reported, got {d.problems}")
    check(close(d.left.position, [-0.2, -0.15, 0.95]), "left position parsed")
    check(d.left.gripper == GRIP_CLOSE, "left gripper closed to the configured value")
    check(close(d.right.position, [0.2, -0.15, 0.95]), "right position parsed")
    check(d.right.gripper == GRIP_OPEN, "right gripper opened to the configured value")
    check(d.right.quat is None, "right orientation stayed keep")
    check(d.note == "moving the left arm above the plug" and d.phase == "approach", "prose carried through")
    check(not d.is_noop, "a moving payload is not a no-op")

    # the 3-list orientation is relative to HOME, composed as R_home @ R_rpy
    expected = absolute_quat_from_relative_rpy([0.0, 0.0, 0.3], HOME_LEFT)
    check(close(d.left.quat, expected, 1e-12), "relative rpy was composed with HOME")
    check(
        close(relative_rpy(d.left.quat, HOME_LEFT), [0.0, 0.0, 0.3], 1e-9),
        "the composed orientation reads back as the requested relative rpy",
    )


def test_keep_tokens() -> None:
    print("keep spellings")
    for token in ("keep", "KEEP", " keep ", "hold", "same", "unchanged", "none", "null", ""):
        d = parse({"left": token, "right": token})
        check(d.left.is_keep and d.right.is_keep, f"{token!r} is treated as keep")
        check(d.is_noop, f"{token!r} yields a no-op")
    d = parse({})
    check(d.is_noop, "an empty object is a no-op")
    d = parse({"left": {"position": "keep", "orientation": "keep", "gripper": "keep"}, "right": None})
    check(d.is_noop and d.problems == [], "all-keep fields are a clean no-op")
    # a JSON null for a whole arm is tolerated as keep
    d = parse({"left": None, "right": "keep"})
    check(d.is_noop, "null arms are treated as keep")


def test_orientation_forms() -> None:
    print("orientation forms")
    # absolute quaternion as a 4-list
    quat = [0.9, 0.1, 0.2, 0.3]
    d = parse({"left": {"orientation": quat}})
    check(d.problems == [], f"4-list quaternion accepted: {d.problems}")
    check(abs(float(np.linalg.norm(d.left.quat)) - 1.0) < 1e-12, "the quaternion is normalized")
    check(
        close(quat_wxyz_to_matrix(d.left.quat), quat_wxyz_to_matrix(quat), 1e-12),
        "normalizing preserved the rotation",
    )

    # absolute quaternion as a mapping
    d = parse({"left": {"orientation": {"quat": quat}}}, )
    check(d.problems == [] and close(d.left.quat, np.array(quat) / np.linalg.norm(quat), 1e-12),
          "{\"quat\": [...]} accepted")

    # an unnormalized quaternion is repaired, not rejected
    d = parse({"left": {"orientation": [2.0, 0.0, 0.0, 0.0]}})
    check(d.problems == [] and close(d.left.quat, [1.0, 0.0, 0.0, 0.0], 1e-12),
          "an unnormalized quaternion is normalized")

    # a relative rpy of all zeros is the HOME orientation, not the identity.
    # HOME_LEFT as published is not exactly unit norm, so the reconstructed
    # quaternion matches the *normalized* HOME rather than the raw numbers.
    d = parse({"left": {"orientation": [0.0, 0.0, 0.0]}})
    check(close(d.left.quat, normalize_quat_wxyz(HOME_LEFT), 1e-9),
          "relative [0,0,0] resolves to HOME")
    check(
        abs(math.degrees(quat_angle_between(d.left.quat, [1.0, 0.0, 0.0, 0.0]))) > 1.0,
        "and HOME is deliberately not the identity rotation",
    )

    # a degenerate quaternion is rejected with a reason
    d = parse({"left": {"orientation": [0.0, 0.0, 0.0, 0.0]}})
    check(d.left.quat is None, "a zero quaternion falls back to keep")
    check(any("invalid quaternion" in p for p in d.problems), f"and is reported: {d.problems}")

    # 2 numbers is neither form
    d = parse({"left": {"orientation": [0.1, 0.2]}})
    check(d.left.quat is None, "a 2-list orientation falls back to keep")
    check(any("orientation must be" in p for p in d.problems), f"and is reported: {d.problems}")

    # rotation / quat are accepted as aliases for orientation
    d = parse({"left": {"rotation": [0.0, 0.0, 0.1]}})
    check(d.problems == [] and d.left.quat is not None, "'rotation' is an alias for 'orientation'")


def test_position_forms() -> None:
    print("position forms")
    d = parse({"left": {"position": {"xyz": [-0.2, -0.15, 0.95]}}})
    check(close(d.left.position, [-0.2, -0.15, 0.95]), "a mapping position is unwrapped")

    for bad in ([-0.2, -0.15], [-0.2, -0.15, 0.95, 1.0], ["a", "b", "c"], [np.nan, 0.0, 0.0]):
        d = parse({"left": {"position": bad}})
        check(d.left.position is None, f"position {bad} falls back to keep")
        check(d.problems != [], f"position {bad} is reported")
    # problems must not abort the rest of the payload
    d = parse({"left": {"position": ["a", "b", "c"]}, "right": {"position": [0.2, -0.15, 0.95]}})
    check(close(d.right.position, [0.2, -0.15, 0.95]), "a bad left field does not discard the right arm")


def test_gripper_forms() -> None:
    print("gripper forms")
    for token, expected in (("open", GRIP_OPEN), ("release", GRIP_OPEN), ("close", GRIP_CLOSE),
                            ("grasp", GRIP_CLOSE), ("grip", GRIP_CLOSE)):
        d = parse({"left": {"gripper": token}})
        check(d.left.gripper == expected, f"gripper token {token!r} -> {expected}")

    d = parse({"left": {"gripper": 0.5}})
    check(d.left.gripper == 0.5, "a numeric gripper passes through")
    d = parse({"left": {"gripper": 1.7}})
    check(d.left.gripper == 1.0, "a numeric gripper is clipped to [0, 1]")
    d = parse({"left": {"gripper": -3}})
    check(d.left.gripper == 0.0, "a negative gripper is clipped to 0")

    for bad in ("squeeze", True):
        d = parse({"left": {"gripper": bad}})
        check(d.left.gripper is None, f"gripper {bad!r} falls back to keep")
        check(d.problems != [], f"gripper {bad!r} is reported")


def test_error_reporting() -> None:
    print("error reporting")
    try:
        parse([1, 2, 3])
    except ParseError as exc:
        check("expected a JSON object" in str(exc), "a non-object payload raises ParseError")
    else:
        check(False, "a non-object payload should raise ParseError")

    d = parse({"left": {"position": [-0.2, -0.15, 0.95], "wobble": 3}})
    check(any("unrecognised keys" in p for p in d.problems), f"unknown keys are reported: {d.problems}")
    check(close(d.left.position, [-0.2, -0.15, 0.95]), "and the known fields still parse")

    d = parse({"left": "walk forward", "right": "keep"})
    check(d.left.is_keep, "a prose value for an arm falls back to keep")
    check(any("must be an object" in p for p in d.problems), f"and is reported: {d.problems}")

    # prose fields are truncated rather than allowed to bloat the log or the thread
    d = parse({"note": "x" * 5000, "phase": "y" * 500})
    check(len(d.note) == 1000 and len(d.phase) == 120, "long prose is truncated")

    d = parse({"left": {"position": [-0.2, -0.15, 0.95]}})
    check(d.raw == {"left": {"position": [-0.2, -0.15, 0.95]}}, "the raw payload is retained for audit")


def test_gripper_bounds_follow_config() -> None:
    print("gripper bounds come from the config, not from constants")
    d = parse({"left": {"gripper": "close"}}, gripper_close=0.42)
    check(d.left.gripper == 0.42, "a configured close value is used verbatim")
    d = parse({"left": {"gripper": "open"}}, gripper_open=0.9)
    check(d.left.gripper == 0.9, "a configured open value is used verbatim")


def main() -> int:
    for test in (
        test_full_payload,
        test_keep_tokens,
        test_orientation_forms,
        test_position_forms,
        test_gripper_forms,
        test_error_reporting,
        test_gripper_bounds_follow_config,
    ):
        test()
    print()
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)} of {_CHECKS} checks:")
        for label in _FAILURES[:40]:
            print(f"  - {label}")
        return 1
    print(f"ok: {_CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
