#!/usr/bin/env python3
"""本地单测：优化 B（X_VLA server 批量推理），无需 GPU/Isaac。

用法：cd goai_2026 && /opt/anaconda3/envs/lerobot/bin/python scripts/test_optB_batch_infer.py

覆盖的核心不变式（与服务器上数值校验互补）：
1) 批量路径 get_action_batch(env_idx_list) 与顺序路径返回结果**逐 env 逐位一致**，
   当每 env 用自己的 generator 预抽 x1、再整批 generate_actions 时；
   fake model.generate_actions 在 x1=None 时内部从 generator 抽（模拟顺序），
   x1 给定则直接用（模拟批量），两者应产出完全相同的行。
2) 批量路径只调用一次 processor / 一次 generate_actions（不是每 env 一次）。
3) generator 状态推进一致：两次连续重规划后各 env 生成器噪声序列仍一致。
4) 单 env（len==1）时即使 batch_inference=true 也走顺序路径（行为不变）。

说明：GPU 上的真实数值等价（批 GEMM 浮点舍入）无法在本机验证，由部署后
服务器端"批量 vs 顺序"同输入对拍 + 正式评测 score 带核验。
"""

import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_VLA_DIR = os.path.join(ROOT, "RoboDojo/XPolicyLab/policy/X_VLA")


def _load(mod_name, rel_path):
    path = os.path.join(X_VLA_DIR, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODEL = _load("xvla_model_mod", "model.py")

A, D = 30, 20
CAMS = 3


class FakeActionSpace:
    dim_action = D


class FakeModel:
    num_actions = A
    action_space = FakeActionSpace()

    def generate_actions(self, *, input_ids, image_input, image_mask, domain_id,
                         proprio, steps, generator=None, x1=None):
        """行独立的伪生成：x1 未给则按 generator 抽（模拟顺序），给了直接用。

        返回 [B, num_actions, dim]。为校验"噪声归属"，让输出=噪声本身 × 每 env
        专属常数（proprio 首元素），若批量把某 env 噪声串到别的行会立刻不相等。
        """
        B = input_ids.shape[0]
        if x1 is None:
            x1 = torch.randn(B, A, D, generator=generator, dtype=torch.float32)
        else:
            x1 = x1.to(dtype=torch.float32)
        # 依赖每行 propio 首元素的缩放，确保逐行输入确实"只影响本行"
        scale = proprio[:, 0:1, None]  # [B,1,1]
        return x1 * scale


class FakeProcessor:
    def __call__(self, images, language_instruction):
        B = len(images) if isinstance(images[0], (list, tuple)) else 1
        if B == 1 and not isinstance(images[0], (list, tuple)):
            images = [images]
        N = max(len(s) for s in images)
        return {
            "input_ids": torch.arange(B * 8).view(B, 8).long(),
            "image_input": torch.zeros(B, N, 3, 64, 64),
            "image_mask": torch.ones(B, N, dtype=torch.bool),
        }


def _make_obs(env_idx, seed):
    rng = np.random.default_rng(seed)
    images = [rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8) for _ in range(CAMS)]
    proprio = np.asarray([float(seed % 7 + 1)] + [0.0] * (D - 1), dtype=np.float32)
    return {
        "images": images,
        "proprio": proprio,
        "prompt": f"task-instruction-env{env_idx}",
        "episode_idx": f"ep{env_idx}",
    }


def _make_model(batch_inference=True, policy_seed=123):
    m = MODEL.Model.__new__(MODEL.Model)
    m.model_cfg = {"domain_id": 0}
    m.device = torch.device("cpu")
    m.processor = FakeProcessor()
    m.model = FakeModel()
    m.batch_inference = batch_inference
    m.model_chunk_size = A
    m.actions_per_chunk = A
    m.denoise_steps = 5
    m.temporal_ensemble_active = False
    m.temporal_ensemble_coeff = None
    m.temporal_ensemble_horizon = None
    m._temporal_ensemblers = {}
    m._hysteresis_cfg = SimpleNamespace(enabled=False)
    m.log_io = False
    m.gripper_mode = "continuous"
    m.gripper_threshold = 0.7
    m.policy_seed = policy_seed
    m._policy_generators = {}
    m._policy_noise_draws = {}
    m.observation_window = [None]
    return m


def _feed(m, env_indices, seeds):
    """模拟 update_obs_batch 缓存各 env obs。"""
    m._latest_env_idx_list = [int(i) for i in env_indices]
    m._raw_by_env = {int(i): {"episode_idx": f"ep{i}"} for i in env_indices}
    m._latest_by_env = {int(i): _make_obs(i, s) for i, s in zip(env_indices, seeds)}
    m._request_index = 0


def _chunk_to_16d(actions_list):
    """action dict list -> (T,16) float，便于逐位比较。"""
    return np.stack(
        [
            np.concatenate(
                [
                    act["left_ee_pose"],
                    act["left_ee_joint_state"],
                    act["right_ee_pose"],
                    act["right_ee_joint_state"],
                ]
            ).astype(np.float32)
            for act in actions_list
        ]
    )


def _fake_infer(self, observation, steps=None, generator=None):
    """复刻顺序路径 infer 的调法：processor 单样本 → generate_actions 内部抽噪声。"""
    pil_images = [MODEL.Image.fromarray(image) for image in observation["images"]]
    prompt = observation["prompt"]
    inputs = self.processor(images=pil_images, language_instruction=prompt)
    proprio = torch.as_tensor(observation["proprio"], dtype=torch.float32).unsqueeze(0)
    domain_id = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        action = self.model.generate_actions(
            **inputs,
            proprio=proprio,
            domain_id=domain_id,
            steps=int(steps if steps is not None else self.denoise_steps),
            generator=generator,
        )
    return action.squeeze(0).float().numpy()


def test_batch_equals_sequential():
    env_indices = [0, 1, 2, 5]  # 非连续 idx，验证行顺序映射
    seeds = [10, 11, 12, 13]

    seq_m = _make_model(batch_inference=False)
    _feed(seq_m, env_indices, seeds)
    seq_m.infer = _fake_infer.__get__(seq_m, MODEL.Model)
    # 两次连续重规划（验证 generator 状态推进一致）
    res_seq = [seq_m.get_action_batch(env_indices) for _ in range(2)]

    bat_m = _make_model(batch_inference=True)
    _feed(bat_m, env_indices, seeds)
    res_bat = [bat_m.get_action_batch(env_indices) for _ in range(2)]

    for round_i in range(2):
        assert len(res_bat[round_i]) == len(env_indices) == len(res_seq[round_i])
        for k, env_idx in enumerate(env_indices):
            seq_chunk = _chunk_to_16d(res_seq[round_i][k])
            bat_chunk = _chunk_to_16d(res_bat[round_i][k])
            assert seq_chunk.shape == bat_chunk.shape, (seq_chunk.shape, bat_chunk.shape)
            np.testing.assert_array_equal(
                bat_chunk, seq_chunk,
                err_msg=f"round{round_i} env{env_idx} 批量≠顺序（逐位）",
            )
    # generator 噪声序列应同步推进到"第 2 次"同值（即两次重规划都没串行错位）
    print("PASS: 批量 get_action_batch == 顺序逐 env 逐位一致（含 2 次重规划）")


def test_batch_calls_once():
    env_indices = [0, 1, 2]
    seeds = [1, 2, 3]
    calls = {"gen": 0, "proc": 0}
    orig_gen = FakeModel.generate_actions
    orig_proc = FakeProcessor.__call__

    def counting_gen(self, **kw):
        calls["gen"] += 1
        return orig_gen(self, **kw)

    def counting_proc(self, images, language_instruction):
        calls["proc"] += 1
        return orig_proc(self, images, language_instruction)

    FakeModel.generate_actions = counting_gen
    FakeProcessor.__call__ = counting_proc
    try:
        bat_m = _make_model(batch_inference=True)
        _feed(bat_m, env_indices, seeds)
        bat_m.get_action_batch(env_indices)
    finally:
        FakeModel.generate_actions = orig_gen
        FakeProcessor.__call__ = orig_proc
    assert calls["gen"] == 1, f"批量应只调 1 次 generate_actions，实际 {calls['gen']}"
    assert calls["proc"] == 1, f"批量应只调 1 次 processor，实际 {calls['proc']}"
    print("PASS: 批量路径 3 env 只调 1 次 processor + 1 次 generate_actions")


def test_single_env_uses_sequential_path():
    seq_m = _make_model(batch_inference=False)
    _feed(seq_m, [0], [5])
    seq_m.infer = _fake_infer.__get__(seq_m, MODEL.Model)
    res_seq = seq_m.get_action_batch([0])

    bat_m = _make_model(batch_inference=True)
    _feed(bat_m, [0], [5])
    bat_m.infer = _fake_infer.__get__(bat_m, MODEL.Model)
    res_bat = bat_m.get_action_batch([0])

    np.testing.assert_array_equal(
        _chunk_to_16d(res_bat[0]), _chunk_to_16d(res_seq[0])
    )
    print("PASS: 单 env（batch_inference=true）仍走顺序路径，结果一致")


if __name__ == "__main__":
    test_batch_equals_sequential()
    test_batch_calls_once()
    test_single_env_uses_sequential_path()
    print("\nALL TESTS PASSED (optimization B local logic)")
