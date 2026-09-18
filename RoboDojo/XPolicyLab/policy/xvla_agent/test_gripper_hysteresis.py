"""apply_gripper_hysteresis 本地单元测试（无 GPU / 无 torch 依赖）。

运行：python test_gripper_hysteresis.py   （也可用 pytest 收集 test_*）
"""
import numpy as np

from gripper_hysteresis import HysteresisConfig, apply_gripper_hysteresis


def _actions(left_seq, right_seq=None):
    """构造 ee-dict 动作 list（16 维：pose(7) + joint_state(1) × 双臂）。"""
    t = len(left_seq)
    right_seq = right_seq if right_seq is not None else [0.0] * t
    acts = []
    for i in range(t):
        acts.append(
            {
                "left_ee_pose": np.zeros(7, dtype=np.float32),
                "left_ee_joint_state": np.asarray([left_seq[i]], dtype=np.float32),
                "right_ee_pose": np.zeros(7, dtype=np.float32),
                "right_ee_joint_state": np.asarray([right_seq[i]], dtype=np.float32),
            }
        )
    return acts


def _left(acts):
    return [float(a["left_ee_joint_state"][0]) for a in acts]


def test_binary_deadzone_holds():
    """binary 死区保持：闭合出发，[0.5,0.6,0.65,0.8] → [0,0,0,1]（设计文档示例）。"""
    acts = _actions([0.5, 0.6, 0.65, 0.8])
    apply_gripper_hysteresis(acts, left_init=0.2, right_init=0.2, mode="binary")
    assert _left(acts) == [0.0, 0.0, 0.0, 1.0], _left(acts)


def test_binary_open_init_reversal():
    """binary 初始张开(init=0.9)：死区内保持，<lo 才闭合。"""
    acts = _actions([0.7, 0.5, 0.2, 0.4])
    apply_gripper_hysteresis(acts, left_init=0.9, right_init=0.9, mode="binary")
    assert _left(acts) == [1.0, 1.0, 0.0, 0.0], _left(acts)


def test_direction_latch_monotone():
    """direction_latch 连续迟滞：初始闭合(0.2)，死区内单调钳制，>hi 翻转张开。"""
    acts = _actions([0.5, 0.6, 0.65, 0.8])
    apply_gripper_hysteresis(acts, left_init=0.2, right_init=0.2, mode="direction_latch")
    assert np.allclose(_left(acts), [0.2, 0.2, 0.2, 0.8]), _left(acts)


def test_direction_latch_no_reversal_in_deadzone():
    """禁止死区内方向反转：张开到 0.9 后死区内回落被压住，<lo 才闭合。"""
    acts = _actions([0.75, 0.5, 0.45, 0.2])
    apply_gripper_hysteresis(acts, left_init=0.9, right_init=0.9, mode="direction_latch")
    assert np.allclose(_left(acts), [0.9, 0.9, 0.9, 0.2]), _left(acts)


def test_left_right_independent():
    """双臂独立：left/right 各自按初值迟滞，互不干扰。"""
    acts = _actions([0.8], right_seq=[0.1])
    apply_gripper_hysteresis(acts, left_init=0.2, right_init=0.9, mode="binary")
    assert acts[0]["left_ee_joint_state"][0] == 1.0  # left 初值闭合，0.8>hi → 张开
    assert acts[0]["right_ee_joint_state"][0] == 0.0  # right 初值张开，0.1<lo → 闭合


def test_stateless_pure():
    """无跨请求状态：相同输入+init → 相同输出；init 不同 → 输出不同。"""
    acts1 = _actions([0.5, 0.8])
    acts2 = _actions([0.5, 0.8])
    apply_gripper_hysteresis(acts1, left_init=0.2, right_init=0.2, mode="binary")
    apply_gripper_hysteresis(acts2, left_init=0.2, right_init=0.2, mode="binary")
    assert _left(acts1) == _left(acts2)
    acts3 = _actions([0.5, 0.8])
    apply_gripper_hysteresis(acts3, left_init=0.9, right_init=0.9, mode="binary")
    assert _left(acts3) != _left(acts1)


def test_invalid_mode_raises():
    try:
        HysteresisConfig(mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad mode")


def test_apply_invalid_mode_raises():
    acts = _actions([0.5])
    try:
        apply_gripper_hysteresis(
            acts, left_init=0.2, right_init=0.2, mode="bogus"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad apply mode")


def test_nonfinite_init_raises():
    for value in (float("nan"), float("inf"), float("-inf")):
        acts = _actions([0.5])
        try:
            apply_gripper_hysteresis(
                acts, left_init=value, right_init=0.2, mode="direction_latch"
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for left_init={value}")


def test_out_of_range_init_raises():
    for value in (-0.01, 1.01):
        acts = _actions([0.5])
        try:
            apply_gripper_hysteresis(
                acts, left_init=0.2, right_init=value, mode="binary"
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for right_init={value}")


def test_invalid_range_raises():
    try:
        HysteresisConfig(lo=0.7, hi=0.3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for lo >= hi")


if __name__ == "__main__":
    import sys
    import traceback

    tests = [
        fn
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
