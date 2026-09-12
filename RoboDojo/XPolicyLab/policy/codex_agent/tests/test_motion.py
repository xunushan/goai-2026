"""Unit tests for :mod:`codex_agent.motion`.

Pure numpy (plus scipy for cross-checking the rotations), no simulator, no
network, no Codex. Run with::

    python tests/test_motion.py
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
    ArmCommand,
    ArmState,
    Box,
    GuardrailConfig,
    MotionConfig,
    absolute_quat_from_relative_rpy,
    check_arm_command,
    hold_chunk,
    interpolate_chunk,
    matrix_to_quat_wxyz,
    normalize_quat_wxyz,
    plan_arm,
    quat_angle_between,
    quat_wxyz_to_matrix,
    quat_wxyz_to_rpy,
    relative_rpy,
    rpy_to_quat_wxyz,
    slerp,
)

HOME_QUAT = np.array([0.7070, 0.0, 0.0, 0.7072])
HOME_POS = np.array([-0.2995, -0.3523, 0.9215])
BOX = Box(x=(-0.50, -0.05), y=(-0.41, 0.02), z=(0.88, 1.11))
CFG = MotionConfig(
    delta_p_max_m=0.015,
    delta_theta_max_deg=5.0,
    settle_steps=3,
    gripper_open=1.0,
    gripper_close=0.36,
)
GUARD = GuardrailConfig(workspace=BOX, reject_margin_m=0.03, max_target_distance_m=0.45)

_FAILURES: list[str] = []
_CHECKS = 0


def check(condition: bool, label: str) -> None:
    global _CHECKS
    _CHECKS += 1
    if not condition:
        _FAILURES.append(label)
        print(f"  FAIL  {label}")


def close(a, b, tol=1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=tol, rtol=0))


def state(pos=HOME_POS, quat=HOME_QUAT, gripper=1.0) -> ArmState:
    return ArmState(
        pos=np.asarray(pos, dtype=np.float64),
        quat=normalize_quat_wxyz(quat),
        gripper=float(gripper),
    )


# --------------------------------------------------------------------------- #
# pose algebra
# --------------------------------------------------------------------------- #


def test_quat_matrix_roundtrip() -> None:
    print("pose algebra: quaternion <-> matrix")
    rng = np.random.default_rng(0)
    for _ in range(500):
        q = normalize_quat_wxyz(rng.normal(size=4))
        back = matrix_to_quat_wxyz(quat_wxyz_to_matrix(q))
        check(
            close(q, back, 1e-7) or close(q, -back, 1e-7),
            f"quat->matrix->quat round trip ({q})",
        )
        m = quat_wxyz_to_matrix(q)
        check(close(m @ m.T, np.eye(3), 1e-9), "rotation matrix is orthonormal")
        check(abs(float(np.linalg.det(m)) - 1.0) < 1e-9, "rotation matrix determinant is 1")


def test_rpy_matches_scipy() -> None:
    print("pose algebra: rpy convention against scipy")
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        print("  SKIP  scipy is not installed")
        return
    samples = [
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.0, -0.4, 0.0],
        [0.0, 0.0, 0.9],
        [0.2, -0.3, 1.1],
        [-0.5, 0.25, -0.7],
    ]
    for rpy in samples:
        ours = quat_wxyz_to_matrix(rpy_to_quat_wxyz(rpy))
        theirs = Rotation.from_euler("xyz", rpy).as_matrix()
        check(close(ours, theirs, 1e-12), f"rpy->matrix matches scipy for {rpy}")
        # scipy returns xyzw; ours is wxyz
        sx, sy, sz, sw = Rotation.from_euler("xyz", rpy).as_quat()
        check(
            close(rpy_to_quat_wxyz(rpy), [sw, sx, sy, sz], 1e-12)
            or close(rpy_to_quat_wxyz(rpy), [-sw, -sx, -sy, -sz], 1e-12),
            f"rpy->quat (wxyz) matches scipy for {rpy}",
        )


def test_rpy_roundtrip() -> None:
    print("pose algebra: rpy round trip")
    rng = np.random.default_rng(1)
    for _ in range(300):
        rpy = rng.uniform(-1.0, 1.0, size=3)  # well inside the gimbal-lock band
        back = quat_wxyz_to_rpy(rpy_to_quat_wxyz(rpy))
        check(close(rpy, back, 1e-9), f"rpy round trip for {rpy}")

    # Relative rpy is defined so that the HOME orientation maps to all zeros.
    check(close(relative_rpy(HOME_QUAT, HOME_QUAT), [0.0, 0.0, 0.0], 1e-12), "relative_rpy(HOME)=0")
    for _ in range(200):
        rpy = rng.uniform(-0.8, 0.8, size=3)
        absolute = absolute_quat_from_relative_rpy(rpy, HOME_QUAT)
        check(
            close(relative_rpy(absolute, HOME_QUAT), rpy, 1e-9),
            "relative_rpy is the inverse of absolute_quat_from_relative_rpy",
        )


def test_slerp() -> None:
    print("slerp")
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])  # 90 deg about x
    check(
        abs(math.degrees(quat_angle_between(a, b)) - 90.0) < 1e-9,
        "quat_angle_between measures 90 degrees",
    )
    mid = slerp(a, b, 0.5)
    check(
        abs(math.degrees(quat_angle_between(a, mid)) - 45.0) < 1e-9,
        "slerp midpoint is 45 degrees from the start",
    )
    for t in np.linspace(0.0, 1.0, 21):
        q = slerp(a, b, float(t))
        check(abs(float(np.linalg.norm(q)) - 1.0) < 1e-12, f"slerp norm at t={t}")
        # constant angular speed: angle from a grows linearly in t
        check(
            abs(math.degrees(quat_angle_between(a, q)) - 90.0 * float(t)) < 1e-7,
            f"slerp is linear in angle at t={t}",
        )
    check(close(slerp(a, b, 0.0), a), "slerp at t=0 is the start")
    check(
        close(slerp(a, b, 1.0), b) or close(slerp(a, b, 1.0), -b),
        "slerp at t=1 is the target",
    )

    # q and -q are the same rotation: slerp must take the short way round.
    for t in (0.25, 0.5, 0.75):
        check(close(slerp(a, -b, t), slerp(a, b, t), 1e-12), f"slerp shortest path at t={t}")
    check(
        close(slerp(a, a, 0.5), a, 1e-12),
        "slerp between identical quaternions is a no-op",
    )
    check(
        abs(math.degrees(quat_angle_between(b, -b))) < 1e-9,
        "quat_angle_between treats q and -q as the same rotation",
    )


def test_normalize_rejects_garbage() -> None:
    print("normalize_quat_wxyz rejection")
    for bad in ([0, 0, 0, 0], [1, 2, 3], [np.nan, 0, 0, 1], [np.inf, 0, 0, 1]):
        try:
            normalize_quat_wxyz(bad)
        except ValueError:
            check(True, f"normalize rejects {bad}")
        else:
            check(False, f"normalize accepted {bad}")


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def test_plan_arm_step_counts() -> None:
    print("plan_arm step counts")
    now = state()
    # 0.30 m at 0.015 m/step -> exactly 20 steps
    plan = plan_arm(now, ArmCommand(position=now.pos + np.array([0.30, 0.0, 0.0])), CFG)
    check(plan.n_raw == 20, f"0.30 m / 0.015 m needs 20 steps, got {plan.n_raw}")

    # 90 deg at 5 deg/step -> 18 steps. Note HOME is itself a 90 deg z-rotation,
    # so a relative rpy has to be composed with it rather than used raw.
    q90 = absolute_quat_from_relative_rpy([0.0, 0.0, math.pi / 2], HOME_QUAT)
    check(
        abs(math.degrees(quat_angle_between(HOME_QUAT, q90)) - 90.0) < 1e-6,
        "the test's 90 deg target really is 90 deg from HOME",
    )
    plan = plan_arm(now, ArmCommand(quat=q90), CFG)
    check(plan.n_raw == 18, f"90 deg / 5 deg needs 18 steps, got {plan.n_raw}")

    # the slower axis sets the pace
    plan = plan_arm(
        now, ArmCommand(position=now.pos + np.array([0.03, 0.0, 0.0]), quat=q90), CFG
    )
    check(plan.n_raw == 18, f"rotation dominates translation, got {plan.n_raw}")

    # a pure keep is one step (plus settle), not zero
    plan = plan_arm(now, ArmCommand(), CFG)
    check(plan.n_raw == 1, f"a keep plans 1 step, got {plan.n_raw}")
    check(not plan.moved, "a keep reports moved=False")

    # a pure gripper command is also one step
    plan = plan_arm(now, ArmCommand(gripper=0.36), CFG)
    check(plan.n_raw == 1, f"a gripper-only command plans 1 step, got {plan.n_raw}")
    check(plan.moved, "a gripper-only command reports moved=True")


# --------------------------------------------------------------------------- #
# interpolator
# --------------------------------------------------------------------------- #


def test_chunk_shape_and_gripper_timing() -> None:
    print("interpolate_chunk: shapes, gripper timing, settle")
    left = state()
    right = state(pos=[0.3005, -0.3523, 0.9215])
    target = left.pos + np.array([0.30, 0.0, 0.0])
    chunk, info = interpolate_chunk(
        left, ArmCommand(position=target, gripper=0.36), right, ArmCommand(), 45, CFG
    )
    check(len(chunk) == 23, f"20 interpolation + 3 settle steps, got {len(chunk)}")
    check(info.n_raw_left == 20 and info.n_raw_right == 1, "chunk info records both plans")
    check(not info.truncated and not info.gripper_deferred, "a 20-step move is not truncated")

    keys = {"left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"}
    for i, step in enumerate(chunk):
        check(set(step) == keys, f"step {i} has exactly the four protocol keys")
        for key, value in step.items():
            check(isinstance(value, np.ndarray), f"step {i}.{key} is an ndarray")
            check(value.dtype == np.float32, f"step {i}.{key} is float32")
            check(np.isfinite(value).all(), f"step {i}.{key} is finite")
        check(step["left_ee_pose"].shape == (7,), f"step {i} left pose is 7-dim")
        check(step["left_ee_joint_state"].shape == (1,), f"step {i} left gripper is 1-dim")
        check(step["right_ee_pose"].shape == (7,), f"step {i} right pose is 7-dim")
        check(step["right_ee_joint_state"].shape == (1,), f"step {i} right gripper is 1-dim")

    # the gripper holds its start value until the arm has arrived (index 19 is
    # the arrival step, where the change lands)
    for i in range(19):
        check(
            abs(float(chunk[i]["left_ee_joint_state"][0]) - 1.0) < 1e-6,
            f"gripper is unchanged during the move (step {i})",
        )
    check(
        abs(float(chunk[19]["left_ee_joint_state"][0]) - 0.36) < 1e-6,
        "gripper reaches its target on the arrival step",
    )

    # settle steps are byte-identical to the arrival step
    for i in range(20, 23):
        for key in keys:
            check(
                chunk[i][key].tobytes() == chunk[19][key].tobytes(),
                f"settle step {i}.{key} is byte-identical to the arrival step",
            )

    # the last interpolation step lands exactly on the target
    check(close(chunk[19]["left_ee_pose"][:3], target, 1e-6), "arrival step hits the target position")
    # the kept arm does not move at all
    for i, step in enumerate(chunk):
        check(
            close(step["right_ee_pose"], np.concatenate([right.pos, right.quat]), 1e-6),
            f"the kept arm never moves (step {i})",
        )


def test_truncation_stops_short() -> None:
    print("interpolate_chunk: truncation stops short of the target")
    left = state()
    right = state(pos=[0.3005, -0.3523, 0.9215])
    target = left.pos + np.array([0.30, 0.0, 0.0])  # wants 20 steps
    cap = 8
    chunk, info = interpolate_chunk(
        left, ArmCommand(position=target, gripper=0.36), right, ArmCommand(), cap, CFG
    )
    check(info.truncated, "the chunk is reported as truncated")
    check(len(chunk) == cap, f"a truncated chunk fills the cap exactly, got {len(chunk)}")
    check(info.truncated_arms == ["left"], f"only the moving arm is listed, got {info.truncated_arms}")
    check(info.gripper_deferred, "a truncated chunk defers the gripper")

    # t = i / n_raw, so after 5 interpolation steps we are 5/20 of the way there
    check(
        close(chunk[4]["left_ee_pose"][:3], left.pos + (target - left.pos) * (5.0 / 20.0), 1e-6),
        "truncated waypoints use t = i / n_raw",
    )
    arrived = float(np.linalg.norm(chunk[4]["left_ee_pose"][:3] - target))
    check(arrived > 0.1, f"the truncated chunk stops well short of the target ({arrived:.3f} m)")
    for i in range(5):
        check(
            abs(float(chunk[i]["left_ee_joint_state"][0]) - 1.0) < 1e-6,
            f"the gripper is held during a truncated chunk (step {i})",
        )


def test_truncation_with_colliding_gripper() -> None:
    """A truncated chunk must not close the gripper on a deferred decision."""
    print("interpolate_chunk: gripper change under a truncating cap")
    left = state(gripper=1.0)
    right = state(pos=[0.3005, -0.3523, 0.9215])
    target = left.pos + np.array([0.30, 0.0, 0.0])  # wants 20 steps
    command = ArmCommand(position=target, gripper=0.36)

    chunk, info = interpolate_chunk(left, command, right, ArmCommand(), 4, CFG)
    check(len(chunk) == 4, f"cap=4 yields 4 steps, got {len(chunk)}")
    check(info.gripper_deferred, "the gripper change is deferred when truncated")
    for i, step in enumerate(chunk):
        check(
            abs(float(step["left_ee_joint_state"][0]) - 1.0) < 1e-6,
            f"the gripper stays open when deferred (step {i})",
        )

    # A gripper-only command needs n_raw = 1, so cap=2 already has room for it:
    # one interpolation step that closes the jaws, plus one settle step.
    chunk, info = interpolate_chunk(left, ArmCommand(gripper=0.36), right, ArmCommand(), 2, CFG)
    check(len(chunk) == 2, f"cap=2 yields 2 steps, got {len(chunk)}")
    check(not info.gripper_deferred, "a 1-step command fits under cap=2")
    for i in range(2):
        check(
            abs(float(chunk[i]["left_ee_joint_state"][0]) - 0.36) < 1e-6,
            f"the gripper is closed from the first step (step {i})",
        )
    check(
        close(chunk[1]["left_ee_pose"], chunk[0]["left_ee_pose"], 1e-7),
        "the settle step holds the arrival pose",
    )


def test_tiny_cap_still_yields_a_chunk() -> None:
    print("interpolate_chunk: cap=1 never produces an empty chunk")
    left = state()
    right = state(pos=[0.3005, -0.3523, 0.9215])
    for cap in (1, 2, 3):
        chunk, _ = interpolate_chunk(
            left,
            ArmCommand(position=left.pos + np.array([0.30, 0.0, 0.0])),
            right,
            ArmCommand(),
            cap,
            CFG,
        )
        check(1 <= len(chunk) <= cap, f"cap={cap} yields a chunk of length {len(chunk)}")


def test_hold_chunk() -> None:
    print("hold_chunk")
    left = state(gripper=0.36)
    right = state(pos=[0.3005, -0.3523, 0.9215])
    for steps in (1, 2, 45):
        chunk = hold_chunk(left, right, steps)
        check(len(chunk) == steps, f"hold_chunk({steps}) has {steps} steps")
        for step in chunk:
            check(
                close(step["left_ee_pose"], np.concatenate([left.pos, left.quat]), 1e-6),
                "the held arm stays where it was",
            )
    check(len(hold_chunk(left, right, 0)) == 1, "hold_chunk(0) is clamped to one step")
    check(len(hold_chunk(left, right, -5)) == 1, "hold_chunk(-5) is clamped to one step")
    chunk = hold_chunk(left, right, 5)
    chunk[0]["left_ee_pose"][0] = 999.0
    check(
        float(chunk[1]["left_ee_pose"][0]) != 999.0,
        "hold_chunk steps do not share a pose buffer",
    )


# --------------------------------------------------------------------------- #
# guardrail
# --------------------------------------------------------------------------- #


def test_guardrail_matrix() -> None:
    """Workspace rejection vs clamping, kept separate from the distance limit.

    The real ``BOX`` is wide enough that its corners are further from HOME than
    ``max_target_distance_m``, which would make the distance limit fire before
    the workspace check. A small box centred on the arm isolates the behaviour
    under test.
    """
    print("guardrail")
    now = state(pos=np.array([-0.275, -0.20, 1.00]), quat=HOME_QUAT)
    small = Box(x=(-0.30, -0.25), y=(-0.25, -0.15), z=(0.95, 1.05))
    guard = GuardrailConfig(workspace=small, reject_margin_m=0.03, max_target_distance_m=0.45)

    inside = check_arm_command(ArmCommand(position=now.pos + np.array([-0.01, 0.0, 0.0])), now, guard)
    check(inside.ok and not inside.clamped, "a small in-box move is accepted")
    check(inside.feedback("left") == "", "an accepted move has no feedback to report")

    far = check_arm_command(ArmCommand(position=now.pos + np.array([0.60, 0.0, 0.0])), now, guard)
    check(not far.ok, "a 0.60 m jump is rejected")
    check("exceeds the per-decision limit" in far.reason, "the rejection explains the distance limit")

    # 0.01 m past the +x face: inside the margin, so clamped onto the boundary
    edge = check_arm_command(ArmCommand(position=np.array([-0.24, -0.20, 1.00])), now, guard)
    check(edge.ok and edge.clamped, "a 0.01 m overshoot is clamped")
    check(close(edge.command.position, np.array([-0.25, -0.20, 1.00]), 1e-12),
          "the clamped target lands on the boundary")
    check("clamped" in edge.feedback("left"), "clamping is reported back to the model")

    # 0.20 m past the same face: beyond the margin, so rejected outright
    outside = check_arm_command(ArmCommand(position=np.array([-0.05, -0.20, 1.00])), now, guard)
    check(not outside.ok, "a 0.20 m overshoot is rejected, not clamped")
    check("outside the valid workspace" in outside.reason, "the rejection names the workspace")
    check("0.200" in outside.reason, "the rejection quantifies the overshoot")

    for bad in ([np.nan, 0.0, 0.0], [np.inf, 0.0, 0.0]):
        outcome = check_arm_command(ArmCommand(position=np.array(bad)), now, guard)
        check(not outcome.ok, f"a non-finite position {bad} is rejected")

    # Rotations are relative to HOME, so the target quaternion has to be built
    # the same way the protocol builds it.
    spin = check_arm_command(
        ArmCommand(quat=absolute_quat_from_relative_rpy([0.0, 0.0, math.pi], HOME_QUAT)),
        now,
        guard,
    )
    check(not spin.ok, "a 180 deg rotation in one decision is rejected")
    check("too large for a single decision" in spin.reason, "the rejection explains the limit")
    quarter = check_arm_command(
        ArmCommand(quat=absolute_quat_from_relative_rpy([0.0, 0.0, 0.2], HOME_QUAT)),
        now,
        guard,
    )
    check(quarter.ok, "a 0.2 rad rotation is accepted")

    keep = check_arm_command(ArmCommand(), now, guard)
    check(keep.ok and keep.command.is_keep, "a bare keep passes through untouched")

    # a clamped command keeps the other fields intact
    mixed = check_arm_command(
        ArmCommand(position=np.array([-0.24, -0.20, 1.00]), gripper=0.36), now, guard
    )
    check(mixed.ok and mixed.clamped and mixed.command.gripper == 0.36, "clamping preserves the gripper")


def test_random_sweep() -> None:
    """2000 random decisions must always produce a legal chunk."""
    print("random sweep of 2000 decisions")
    rng = np.random.default_rng(20260912)
    worst = 0.0
    for _ in range(2000):
        left = state(gripper=float(rng.uniform(0.36, 1.0)))
        right = state(
            pos=[0.3005, -0.3523, 0.9215], gripper=float(rng.uniform(0.36, 1.0))
        )

        def command() -> ArmCommand:
            if rng.random() < 0.25:
                return ArmCommand()
            pos = left.pos + rng.normal(scale=0.08, size=3) if rng.random() < 0.7 else None
            quat = rpy_to_quat_wxyz(rng.normal(scale=0.4, size=3)) if rng.random() < 0.5 else None
            grip = float(rng.uniform(0.36, 1.0)) if rng.random() < 0.4 else None
            return ArmCommand(position=pos, quat=quat, gripper=grip)

        left_cmd, right_cmd = command(), command()
        cap = int(rng.integers(1, 60))
        chunk, info = interpolate_chunk(left, left_cmd, right, right_cmd, cap, CFG)

        assert 1 <= len(chunk) <= cap, f"chunk length {len(chunk)} outside [1,{cap}]"
        for step in chunk:
            assert set(step) == {
                "left_ee_pose",
                "left_ee_joint_state",
                "right_ee_pose",
                "right_ee_joint_state",
            }
            for key, value in step.items():
                assert value.dtype == np.float32, key
                assert np.isfinite(value).all(), key
            assert step["left_ee_pose"].shape == (7,)
            assert step["left_ee_joint_state"].shape == (1,)
            for key in ("left", "right"):
                quat = step[f"{key}_ee_pose"][3:7]
                worst = max(worst, abs(float(np.linalg.norm(quat)) - 1.0))
        # per-step limits hold on every moving arm
        for previous, current in zip(chunk, chunk[1:]):
            for key in ("left", "right"):
                step_distance = float(
                    np.linalg.norm(current[f"{key}_ee_pose"][:3] - previous[f"{key}_ee_pose"][:3])
                )
                assert step_distance <= CFG.delta_p_max_m + 1e-5, (key, step_distance)
                angle = quat_angle_between(
                    previous[f"{key}_ee_pose"][3:7], current[f"{key}_ee_pose"][3:7]
                )
                assert angle <= CFG.delta_theta_max_rad + 1e-5, (key, angle)
    check(worst < 1e-5, f"every emitted quaternion is unit norm (worst error {worst:.2e})")


def main() -> int:
    for test in (
        test_quat_matrix_roundtrip,
        test_rpy_matches_scipy,
        test_rpy_roundtrip,
        test_slerp,
        test_normalize_rejects_garbage,
        test_plan_arm_step_counts,
        test_chunk_shape_and_gripper_timing,
        test_truncation_stops_short,
        test_truncation_with_colliding_gripper,
        test_tiny_cap_still_yields_a_chunk,
        test_hold_chunk,
        test_guardrail_matrix,
        test_random_sweep,
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
