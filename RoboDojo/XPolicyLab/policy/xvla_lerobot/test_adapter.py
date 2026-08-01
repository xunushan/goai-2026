"""Fast adapter tests that do not load an X-VLA checkpoint."""

from __future__ import annotations

import ast
import importlib
import sys
import tempfile
import types
from pathlib import Path

import numpy as np


def _import_model_without_transformers():
    """Stub only heavyweight model classes; exercise the real adapter helpers."""
    package = "XPolicyLab.policy.xvla_lerobot"
    model_module_name = f"{package}.xvla.models.modeling_xvla"
    processor_module_name = f"{package}.xvla.models.processing_xvla"
    model_module = types.ModuleType(model_module_name)
    model_module.XVLA = type("XVLA", (), {})
    processor_module = types.ModuleType(processor_module_name)
    processor_module.XVLAProcessor = type("XVLAProcessor", (), {})
    sys.modules[model_module_name] = model_module
    sys.modules[processor_module_name] = processor_module
    return importlib.import_module(f"{package}.model")


def test_adapter_has_no_legacy_policy_imports():
    policy_dir = Path(__file__).resolve().parent
    for filename in ("model.py", "deploy.py"):
        tree = ast.parse((policy_dir / filename).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(
            name.startswith("XPolicyLab.policy.X_VLA") for name in imports
        )


def test_rotation_roundtrip_and_30_to_25_resampling():
    adapter = _import_model_without_transformers()
    quaternion = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.9238795, 0.0, 0.3826834, 0.0]],
        dtype=np.float32,
    )
    rotation6d = adapter._quat_wxyz_to_rotation6d(quaternion)
    roundtrip = adapter._rotation6d_to_quat_wxyz(rotation6d)
    alignment = np.abs(np.sum(quaternion * roundtrip, axis=-1))
    np.testing.assert_allclose(alignment, 1.0, atol=1e-5)

    anchors = np.zeros((30, 20), dtype=np.float32)
    anchors[:, 3:9] = rotation6d[0]
    anchors[:, 13:19] = rotation6d[0]
    anchors[:, 0] = np.linspace(1 / 30, 1, 30)
    anchors[:, 9] = np.linspace(0, 1, 30)
    anchors[:, 19] = np.linspace(1, 0, 30)
    state = np.array(
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        dtype=np.float32,
    )
    controls = adapter.resample_one_second_chunk(state, anchors, control_hz=25)
    assert controls.shape == (25, 16)
    assert np.isfinite(controls).all()
    np.testing.assert_allclose(controls[-1, 0], 1.0)


def test_empty_batch_stops_without_policy_request():
    from XPolicyLab.policy.xvla_lerobot.deploy import eval_one_episode_batch

    class Environment:
        def is_episode_end(self):
            return False

        def get_running_env_idx_list(self):
            return []

    class Client:
        def __init__(self):
            self.calls = []

        def call(self, **kwargs):
            self.calls.append(kwargs)

    client = Client()
    eval_one_episode_batch(Environment(), client)
    assert client.calls == [{"func_name": "reset"}]


def test_peft_checkpoint_and_separate_processor_are_loaded():
    adapter = _import_model_without_transformers()
    loaded = {}

    class FakeNetwork:
        num_actions = 30
        action_mode = "arx_ee6d"

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    class FakeXVLA:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            loaded["base"] = path
            return FakeNetwork()

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path):
            loaded["processor"] = path
            return cls()

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, path, **kwargs):
            loaded["adapter"] = path
            return model

    adapter.XVLA = FakeXVLA
    adapter.XVLAProcessor = FakeProcessor
    peft_module = types.ModuleType("peft")
    peft_module.PeftModel = FakePeftModel
    sys.modules["peft"] = peft_module

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base_dir = root / "base"
        adapter_dir = root / "adapter"
        processor_dir = root / "processor"
        for path in (base_dir, adapter_dir, processor_dir):
            path.mkdir()
        (base_dir / "config.json").write_text("{}", encoding="utf-8")
        (adapter_dir / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "../base"}', encoding="utf-8"
        )
        (processor_dir / "preprocessor_config.json").write_text(
            "{}", encoding="utf-8"
        )
        adapter.Model(
            {
                "checkpoint_path": str(adapter_dir),
                "processor_path": str(processor_dir),
                "device": "cpu",
                "log_io": False,
            }
        )
        assert loaded == {
            "base": str(base_dir.resolve()),
            "processor": str(processor_dir.resolve()),
            "adapter": str(adapter_dir.resolve()),
        }
        loaded.clear()
        adapter.Model(
            {
                "checkpoint_path": str(adapter_dir),
                "device": "cpu",
                "log_io": False,
            }
        )
        assert loaded == {
            "base": str(base_dir.resolve()),
            "processor": str(base_dir.resolve()),
            "adapter": str(adapter_dir.resolve()),
        }


if __name__ == "__main__":
    test_adapter_has_no_legacy_policy_imports()
    test_rotation_roundtrip_and_30_to_25_resampling()
    test_empty_batch_stops_without_policy_request()
    test_peft_checkpoint_and_separate_processor_are_loaded()
    print("xvla_lerobot adapter tests passed")
