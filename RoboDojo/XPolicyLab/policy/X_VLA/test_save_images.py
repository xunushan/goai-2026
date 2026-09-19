"""save_images.py 的本地单元测试（无 GPU / 无 torch / 无 Isaac）。

运行：python3 test_save_images.py（需 numpy + PIL）

覆盖：开关的零副作用、目录与命名约定、逐 episode/逐相机/逐帧的落盘、帧号在
reset 后归零、相机名 slug 化与缺名回退、写盘失败不抛异常、配置校验，以及客户端
不发 episode_idx 时按 request 归零自编号（含多 env 同 episode、reset 换号）。
"""
import re
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
    assert writer.save_observation(
        episode_idx="e", env_idx=1, images=[_IMG], request_index=0
    ) == []
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

    writer.save_observation(
        episode_idx="4e0badbd", env_idx=1, images=[_IMG, _IMG], request_index=0
    )
    writer.save_observation(
        episode_idx="4e0badbd", env_idx=1, images=[_IMG, _IMG], request_index=1
    )
    writer.save_observation(
        episode_idx="9f00aa11", env_idx=2, images=[_IMG, _IMG], request_index=2
    )

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
    writer.save_observation(
        episode_idx="e", env_idx=0, images=[_IMG], request_index=0
    )
    writer.reset()
    writer.save_observation(
        episode_idx="e", env_idx=0, images=[_IMG], request_index=0
    )
    # 计数器归零 → 第二次仍写 000000（覆盖），而不是递增成 000001
    names = sorted(p.name for p in (writer.run_dir / "env0_e" / "cam_head").iterdir())
    assert names == ["000000.jpg"], names


def test_camera_naming_and_slug():
    """相机名做目录安全化；缺名回退 view_<index>；task/episode 缺失有兜底。"""
    root = Path(tempfile.mkdtemp()) / "sim_images"
    writer = _writer(root, task_name="a b/c", camera_names=["cam/head x"])
    # request_index=None：调用方连推理计数都拿不到 → 退回兜底目录名（见下个测试）
    writer.save_observation(
        episode_idx=None, env_idx=3, images=[_IMG, _IMG, _IMG], request_index=None
    )

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
    unnamed.save_observation(
        episode_idx="e", env_idx=0, images=[_IMG], request_index=0
    )
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
        request_index=0,
    )
    for name in ("a", "b", "c"):
        assert (writer.run_dir / "env0_e" / name / "000000.jpg").exists()


def test_write_failure_does_not_raise():
    """写盘失败只告警一次、返回空列表，绝不中断评测。"""
    root = Path(tempfile.mkdtemp()) / "sim_images"
    writer = _writer(root, task_name="t", camera_names=["cam_head"])
    writer._write_jpeg = lambda path, image: False
    assert writer.save_observation(
        episode_idx="e", env_idx=0, images=[_IMG], request_index=0
    ) == []


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


def test_synth_episode_isolates_episodes_by_request_counter():
    """真机不发 episode_idx：request 递增共用一个目录，归零则换目录、帧号重开。"""
    root = Path(tempfile.mkdtemp()) / "sim_images"
    writer = _writer(root, task_name="t", camera_names=["cam_head"])

    for request in (0, 1, 2):
        writer.save_observation(
            episode_idx=None, env_idx=0, images=[_IMG], request_index=request
        )
    dirs = sorted(p.name for p in writer.run_dir.iterdir())
    assert len(dirs) == 1, dirs
    assert re.fullmatch(r"env0_ep001_[0-9a-f]{4}", dirs[0]), dirs[0]
    assert sorted(p.name for p in (writer.run_dir / dirs[0] / "cam_head").iterdir()) == [
        "000000.jpg",
        "000001.jpg",
        "000002.jpg",
    ]

    # request 归零 = 新 episode：换编号、帧号从 000000 重开
    for request in (0, 1):
        writer.save_observation(
            episode_idx=None, env_idx=0, images=[_IMG], request_index=request
        )
    dirs = sorted(p.name for p in writer.run_dir.iterdir())
    assert len(dirs) == 2, dirs
    second = [name for name in dirs if not name.startswith("env0_ep001")][0]
    assert re.fullmatch(r"env0_ep002_[0-9a-f]{4}", second), second
    assert sorted(p.name for p in (writer.run_dir / second / "cam_head").iterdir()) == [
        "000000.jpg",
        "000001.jpg",
    ]


def test_synth_episode_shared_across_envs_within_one_request():
    """同一次请求里的多个 env 属于同一 episode，各自一个 env<idx>_<编号> 目录。"""
    root = Path(tempfile.mkdtemp()) / "sim_images"
    writer = _writer(root, task_name="t", camera_names=["cam_head"])
    for env_idx in (0, 1):
        writer.save_observation(
            episode_idx=None, env_idx=env_idx, images=[_IMG], request_index=0
        )
    dirs = sorted(p.name for p in writer.run_dir.iterdir())
    assert len(dirs) == 2, dirs
    assert dirs[0].split("_", 1)[0] == "env0" and dirs[1].split("_", 1)[0] == "env1"
    assert dirs[0].split("_", 1)[1] == dirs[1].split("_", 1)[1], dirs


def test_reset_starts_new_synth_episode():
    """reset 后即使 request 还是 0，也换一个新编号（否则两个 episode 会串目录）。"""
    root = Path(tempfile.mkdtemp()) / "sim_images"
    writer = _writer(root, task_name="t", camera_names=["cam_head"])
    writer.save_observation(
        episode_idx=None, env_idx=0, images=[_IMG], request_index=0
    )
    writer.reset()
    writer.save_observation(
        episode_idx=None, env_idx=0, images=[_IMG], request_index=0
    )
    dirs = sorted(p.name for p in writer.run_dir.iterdir())
    assert len(dirs) == 2, dirs
    for name, prefix in zip(dirs, ("env0_ep001", "env0_ep002")):
        assert name.startswith(prefix + "_"), name
        assert (writer.run_dir / name / "cam_head" / "000000.jpg").exists()


def test_request_index_is_required():
    """落盘器必须拿到 request_index：调用方忘了传要当场报错，不能静默混目录。"""
    root = Path(tempfile.mkdtemp()) / "sim_images"
    writer = _writer(root, task_name="t", camera_names=["cam_head"])
    try:
        writer.save_observation(episode_idx=None, env_idx=0, images=[_IMG])
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError when request_index is omitted")


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
