"""save_images.py 的本地单元测试（无 GPU / 无 torch / 无 Isaac）。

运行：python3 test_save_images.py（需 numpy + PIL）

覆盖：开关的零副作用、目录与命名约定、逐 episode/逐相机/逐帧的落盘、帧号在
reset 后归零、相机名 slug 化与缺名回退、写盘失败不抛异常、配置校验。
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from save_images import EpisodeImageWriter, SaveImagesConfig  # noqa: E402

_IMG = np.zeros((48, 64, 3), dtype=np.uint8)
_IMG[..., 1] = 128


def _writer(root, **kwargs):
    cfg = SaveImagesConfig(
        enabled=True, root=str(root), jpeg_quality=kwargs.pop("quality", 80)
    )
    return EpisodeImageWriter(cfg, run_timestamp="20260919_101112", **kwargs)


def test_disabled_touches_nothing():
    """enabled=false：不建目录、不写文件、describe 为 off。"""
    root = Path(tempfile.mkdtemp()) / "sim_images"
    cfg = SaveImagesConfig.from_model_cfg({})
    assert cfg.enabled is False
    assert cfg.root == "/data/outputs/sim_images"
    assert cfg.jpeg_quality == 90
    writer = EpisodeImageWriter(cfg, task_name="stack_bowls", camera_names=["cam_head"])
    assert writer.describe() == "off"
    assert writer.save_observation(episode_idx="e", env_idx=1, images=[_IMG]) == []
    assert writer.run_dir.parent.exists() is False


def test_defaults_and_enabled_parsing():
    """字符串形式的 enabled 按 hysteresis/pace 旧口径解析。"""
    cfg = SaveImagesConfig.from_model_cfg(
        {"save_images": {"enabled": "true", "root": "/tmp/x", "jpeg_quality": 75}}
    )
    assert (cfg.enabled, cfg.root, cfg.jpeg_quality) == (True, "/tmp/x", 75)
    for value, expected in (("false", False), ("off", False), ("", False), (1, True)):
        got = SaveImagesConfig.from_model_cfg({"save_images": {"enabled": value}})
        assert got.enabled is expected, (value, got.enabled)


def test_layout_and_frame_numbering():
    """目录：<root>/<时间戳>_<task>/env<idx>_<episode>/<相机>/<帧号>.jpg。"""
    root = Path(tempfile.mkdtemp()) / "sim_images"
    writer = _writer(root, task_name="stack_bowls", camera_names=["cam_head", "cam_left_wrist"])

    writer.save_observation(episode_idx="4e0badbd", env_idx=1, images=[_IMG, _IMG])
    writer.save_observation(episode_idx="4e0badbd", env_idx=1, images=[_IMG, _IMG])
    writer.save_observation(episode_idx="9f00aa11", env_idx=2, images=[_IMG, _IMG])

    assert writer.run_dir.name == "20260919_101112_stack_bowls"
    episode_dir = writer.run_dir / "env1_4e0badbd"
    assert sorted(p.name for p in episode_dir.iterdir()) == [
        "cam_head",
        "cam_left_wrist",
    ], sorted(p.name for p in episode_dir.iterdir())
    # 本 episode 内帧号 000000/000001；另一 episode 是独立计数
    assert sorted(p.name for p in (episode_dir / "cam_head").iterdir()) == [
        "000000.jpg",
        "000001.jpg",
    ]
    assert (writer.run_dir / "env2_9f00aa11" / "cam_head" / "000000.jpg").exists()

    path = episode_dir / "cam_head" / "000000.jpg"
    with Image.open(path) as image:
        assert (image.format, image.mode, image.size) == ("JPEG", "RGB", (64, 48))


def test_reset_restarts_frame_counter():
    """reset 后帧号回到 000000，已写文件不受影响。"""
    root = Path(tempfile.mkdtemp()) / "sim_images"
    writer = _writer(root, task_name="t", camera_names=["cam_head"])
    writer.save_observation(episode_idx="e", env_idx=0, images=[_IMG])
    writer.reset()
    writer.save_observation(episode_idx="e", env_idx=0, images=[_IMG])
    # 计数器归零 → 第二次仍写 000000（覆盖），而不是递增成 000001
    names = sorted(p.name for p in (writer.run_dir / "env0_e" / "cam_head").iterdir())
    assert names == ["000000.jpg"], names


def test_camera_naming_and_slug():
    """相机名做目录安全化；缺名回退 view_<index>；task/episode 缺失有兜底。"""
    root = Path(tempfile.mkdtemp()) / "sim_images"
    writer = _writer(root, task_name="a b/c", camera_names=["cam/head x"])
    writer.save_observation(episode_idx=None, env_idx=3, images=[_IMG, _IMG, _IMG])

    assert writer.run_dir.name == "20260919_101112_a_b_c"
    episode_dir = writer.run_dir / "env3_unknown_episode"
    assert (episode_dir / "cam_head_x" / "000000.jpg").exists()
    assert (episode_dir / "view_1" / "000000.jpg").exists()
    assert (episode_dir / "view_2" / "000000.jpg").exists()

    unnamed = EpisodeImageWriter(
        SaveImagesConfig(enabled=True, root=str(root)),
        task_name=None,
        camera_names=[],
        run_timestamp="ts",
    )
    unnamed.save_observation(episode_idx="e", env_idx=0, images=[_IMG])
    assert unnamed.run_dir.name == "ts_unknown_task"
    assert (unnamed.run_dir / "env0_e" / "view_0" / "000000.jpg").exists()


def test_non_rgb_and_non_uint8_images():
    """单通道 / 灰度 2-D / 浮点输入都能落盘，不因通道数报错。"""
    root = Path(tempfile.mkdtemp()) / "sim_images"
    writer = _writer(root, task_name="t", camera_names=["a", "b", "c"])
    writer.save_observation(
        episode_idx="e",
        env_idx=0,
        images=[np.zeros((48, 64, 1), np.uint8), np.zeros((48, 64), np.float32), _IMG],
    )
    for name in ("a", "b", "c"):
        assert (writer.run_dir / "env0_e" / name / "000000.jpg").exists()


def test_write_failure_does_not_raise():
    """写盘失败只告警一次、返回空列表，绝不中断评测。"""
    root = Path(tempfile.mkdtemp()) / "sim_images"
    writer = _writer(root, task_name="t", camera_names=["cam_head"])
    writer._write_jpeg = lambda path, image: False
    assert writer.save_observation(episode_idx="e", env_idx=0, images=[_IMG]) == []


def test_config_validation():
    for bad in (
        {"save_images": {"jpeg_quality": 0}},
        {"save_images": {"jpeg_quality": 101}},
        {"save_images": {"root": "   "}},
    ):
        try:
            SaveImagesConfig.from_model_cfg(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")
    try:
        SaveImagesConfig.from_model_cfg({"save_images": 5})
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for non-dict save_images")


if __name__ == "__main__":
    import traceback

    tests = [
        fn for name, fn in sorted(globals().items())
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
