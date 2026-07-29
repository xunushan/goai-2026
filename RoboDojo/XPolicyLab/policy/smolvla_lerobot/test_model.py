"""CPU-only unit tests; no checkpoint weights or network access required."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from XPolicyLab.policy.smolvla_lerobot.model import (
    ACTION_DIM,
    Model,
    _chw_uint8,
    _prompt,
    _resolve_checkpoint,
)
from XPolicyLab.policy.smolvla_lerobot.deploy import eval_one_episode


class SmolVLALeRobotTest(unittest.TestCase):
    def test_hwc_uint8_is_converted_to_chw(self):
        converted = _chw_uint8(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(converted.shape, (3, 480, 640))
        self.assertEqual(converted.dtype, np.uint8)
        self.assertTrue(converted.flags.c_contiguous)

    def test_float_image_is_scaled_to_uint8(self):
        converted = _chw_uint8(np.full((3, 8, 8), 0.5, dtype=np.float32))
        self.assertEqual(converted.shape, (3, 8, 8))
        self.assertTrue(np.all(converted == 127))

    def test_prompt_resolution(self):
        self.assertEqual(
            _prompt({"instruction": " stack blocks "}, "fallback"),
            "stack blocks",
        )
        self.assertEqual(_prompt({}, "fallback"), "fallback")

    def test_checkpoint_resolver(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pretrained_model"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"test")
            resolved = _resolve_checkpoint({"checkpoint_path": directory})
            self.assertEqual(resolved, checkpoint.resolve())

    def test_encode_direct_joint_observation(self):
        model = object.__new__(Model)
        model.default_prompt = "stack blocks"
        model.action_type = "joint"
        model.robot_action_dim_info = None
        observation = {
            "state": np.arange(ACTION_DIM, dtype=np.float32),
            "images": {
                "cam_high": np.zeros((16, 16, 3), dtype=np.uint8),
                "cam_left_wrist": np.zeros((16, 16, 3), dtype=np.uint8),
                "cam_right_wrist": np.zeros((16, 16, 3), dtype=np.uint8),
            },
        }
        encoded = model._encode(observation)
        self.assertEqual(encoded["state"].shape, (14,))
        self.assertEqual(encoded["images"]["cam_high"].shape, (3, 16, 16))
        self.assertEqual(encoded["task"], "stack blocks")

    def test_checkpoint_feature_contract(self):
        model = object.__new__(Model)

        def feature(shape):
            return SimpleNamespace(shape=shape)

        model.policy = SimpleNamespace(
            config=SimpleNamespace(
                input_features={
                    "observation.state": feature((14,)),
                    "observation.images.camera1": feature((3, 256, 256)),
                    "observation.images.camera2": feature((3, 256, 256)),
                    "observation.images.camera3": feature((3, 256, 256)),
                },
                output_features={"action": feature((14,))},
            )
        )
        model._validate_config()

    def test_invalid_state_dimension_is_rejected(self):
        model = object.__new__(Model)
        model.default_prompt = "task"
        model.action_type = "joint"
        model.robot_action_dim_info = None
        with self.assertRaisesRegex(ValueError, "Expected 14D"):
            model._encode(
                {
                    "state": np.zeros(6, dtype=np.float32),
                    "images": {
                        name: np.zeros((3, 8, 8), dtype=np.uint8)
                        for name in (
                            "cam_high",
                            "cam_left_wrist",
                            "cam_right_wrist",
                        )
                    },
                }
            )

    def test_chunk_execution_captures_every_video_frame(self):
        class FakeEnv:
            def __init__(self):
                self.steps = 0
                self.observations = 0

            def is_episode_end(self):
                return self.steps >= 10

            def get_obs(self):
                self.observations += 1
                return {"frame": self.observations}

            def take_action(self, action):
                self.steps += 1

        class FakeClient:
            def __init__(self):
                self.inference_calls = 0

            def call(self, func_name, **kwargs):
                if func_name == "get_action":
                    self.inference_calls += 1
                    return list(range(10))
                return None

        env = FakeEnv()
        client = FakeClient()
        eval_one_episode(env, client)
        self.assertEqual(env.steps, 10)
        self.assertEqual(env.observations, 10)
        self.assertEqual(client.inference_calls, 1)


if __name__ == "__main__":
    unittest.main()
