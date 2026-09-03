#!/usr/bin/env python3
"""本地单测：优化 A（夹爪读回子步缓存），无需 Isaac。

用法：cd goai_2026 && python scripts/test_optA_control_readback.py
覆盖：
1) EpochReadbackCache —— 同 (epoch,rid) 命中、epoch/rid 变化淘汰、clear。
   （A2 读回缓存的核心逻辑；robot_manager.get_end_effector_real_val 只是其薄封装。）
2) control_manager.pop 回归 —— 确认 A1 已回退：pop 仍按原语义每 env 调 2 次 get_action、
   非活跃 env 返回空占位、缺失键从 prev 回填（保证未误改共享框架）。
说明：get_end_effector_real_val 方法层因依赖 isaaclab 无法在本机导入，其正确性由
robot_manager.py py_compile + 服务器复测（PROFILE_EVAL 剖面 + score 对比）验证。
"""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(mod_name, rel_path):
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CM = _load("ctrl_mgr_mod", "RoboDojo/env/robot_manager/control_manager.py")
RC = _load("readback_cache_mod", "RoboDojo/env/robot_manager/readback_cache.py")

OBS_KEYS = [
    "left_arm_joint_state",
    "left_gripper_joint_state",
    "right_arm_joint_state",
    "right_gripper_joint_state",
]
GRIP_KEYS = ["left_gripper_joint_state", "right_gripper_joint_state"]


class _FakeRobotMgr:
    def __init__(self, num_envs):
        self.num_envs = num_envs

    def get_robot_obs_name(self):
        return list(OBS_KEYS)


def _full_ctrl():
    return {
        "left_arm_joint_state": {"position": [0.1 * (i + 1) for i in range(6)], "velocity": [0.0] * 6},
        "left_gripper_joint_state": {"position": [0.42, 0.421], "velocity": [0.0, 0.0]},
        "right_arm_joint_state": {"position": [0.2 * (i + 1) for i in range(6)], "velocity": [0.0] * 6},
        "right_gripper_joint_state": {"position": [0.43, 0.431], "velocity": [0.0, 0.0]},
    }


def test_readback_cache_epoch_hit_and_evict():
    """A2 核心：同 epoch 同 rid 命中同一对象，epoch/rid 变化或 clear 后淘汰。"""
    cache = RC.EpochReadbackCache()
    data0 = [1.0, 2.0]
    assert cache.get(0, "robot_a") is None  # 未命中
    cache.put(0, "robot_a", data0)
    assert cache.get(0, "robot_a") is data0  # 同 epoch 命中 → 无需再 GPU 读回
    assert cache.get(1, "robot_a") is None  # epoch 前进 → 淘汰重读
    assert cache.get(0, "robot_b") is None  # rid 变 → 淘汰重读
    cache.put(1, "robot_b", [3.0])
    assert cache.get(1, "robot_b") == [3.0]
    cache.clear()
    assert cache.get(1, "robot_b") is None
    print("PASS: EpochReadbackCache 同 epoch 命中、epoch/rid 淘汰、clear")


def test_control_manager_pop_regression():
    """回归：control_manager.pop 保持原语义（每 env 2 次 get_action、空占位、prev 回填）。"""
    call_state = {"n": 0}
    orig_get_action = CM.MetaControl.get_action

    def fake_get_action(self, robot_manager, env_idx):
        call_state["n"] += 1
        res = {}
        for key in robot_manager.get_robot_obs_name():
            res[key] = dict(self.control_info_dict[key]) if key in self.control_info_dict else None
        return res

    CM.MetaControl.get_action = fake_get_action
    try:
        num_envs = 3
        cm = CM.ControlManager(num_envs=num_envs, robot_manager=_FakeRobotMgr(num_envs))
        for e in range(num_envs):
            cm.prev_control[e] = {k: {"position": [9.9], "velocity": [0.0]} for k in GRIP_KEYS}
        ctrl0 = _full_ctrl()
        ctrl1 = _full_ctrl()
        for gk in GRIP_KEYS:  # env1 缺 gripper → 验证从 prev 回填
            del ctrl1[gk]
        cm.push([0, 1], [[ctrl0], [ctrl1]])

        call_state["n"] = 0
        res = cm.pop([0, 1])
        # 原语义：update_current_missing + update_prev_control 各一次 get_action
        assert call_state["n"] == 2 * 2, f"get_action 调用次数 = {call_state['n']}，期望 4（每 env 2 次）"
        assert len(res) == num_envs and res[2] == []  # 含非活跃 env 空占位
        assert res[1].control_info_dict["left_gripper_joint_state"] == cm.prev_control[1]["left_gripper_joint_state"]
        assert cm.prev_control[0]["left_gripper_joint_state"]["position"] == [0.42, 0.421]
    finally:
        CM.MetaControl.get_action = orig_get_action
    print("PASS: control_manager.pop 原语义回归（A1 已回退，未改共享框架）")


if __name__ == "__main__":
    test_readback_cache_epoch_hit_and_evict()
    test_control_manager_pop_regression()
    print("\nALL TESTS PASSED (optimization A local logic)")
