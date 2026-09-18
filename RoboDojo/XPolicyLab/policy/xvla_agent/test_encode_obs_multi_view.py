"""encode_obs 多视角图像单元测试（无 GPU）。

优先在真实依赖环境（如 lerobot conda env：torch/cv2/XPolicyLab/xvla 齐全）
直接 import model.py 走完整链路；若缺 torch/cv2，则用 stub 顶替顶层
依赖，仅验证 encode_obs 的图像提取与顺序逻辑。

运行：
  /opt/anaconda3/envs/lerobot/bin/python test_encode_obs_multi_view.py   # 真实依赖
  /opt/anaconda3/bin/python3 test_encode_obs_multi_view.py               # stub 兜底
"""
import os
import sys
import types

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))  # RoboDojo
_XVLA_ROOT = os.path.join(_DIR, "xvla")


def _has_real_deps():
    try:
        import cv2  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


if _has_real_deps():
    # 真实依赖（lerobot env）：走完整 import（真实 XPolicyLab / xvla 链路）。
    sys.path.insert(0, _REPO_ROOT)
    sys.path.insert(0, _DIR)
    sys.path.insert(0, _XVLA_ROOT)
    import model  # noqa: E402
else:
    # 缺 torch/cv2 环境：stub 顶替顶层依赖（encode_obs 路径不触碰这些依赖）。
    _stub_module("torch")
    _stub_module("cv2")

    _stub_module("XPolicyLab")
    _stub_module("XPolicyLab.utils")
    _stub_module("XPolicyLab.model_template", ModelTemplate=object)
    _stub_module(
        "XPolicyLab.utils.checkpoint_resolver",
        resolve_checkpoint_root=lambda *a, **k: None,
    )
    _stub_module(
        "XPolicyLab.utils.process_data",
        decode_image_bit=lambda buf: np.zeros((8, 8, 3), dtype=np.uint8),
        get_robot_action_dim_info=lambda *a, **k: None,
    )

    _stub_module("xvla")
    _stub_module("xvla.models")
    _stub_module("xvla.models.modeling_xvla", XVLA=object)
    _stub_module("xvla.models.processing_xvla", XVLAProcessor=object)

    sys.path.insert(0, _DIR)
    import model  # noqa: E402

from model import encode_obs  # noqa: E402


# ---- 测试数据构造 ----
def _solid_image(rgb, size=(16, 16)):
    """纯色 HWC uint8 图像。rgb=(r,g,b) 按 numpy 通道序。"""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[..., 0] = rgb[0]
    img[..., 1] = rgb[1]
    img[..., 2] = rgb[2]
    return img


def _valid_state():
    return {
        "left_ee_pose": np.array([0.1, 0.2, 0.3, 1, 0, 0, 0], dtype=np.float32),
        "left_ee_joint_state": np.array([0.5], dtype=np.float32),
        "right_ee_pose": np.array([-0.1, 0.2, 0.3, 1, 0, 0, 0], dtype=np.float32),
        "right_ee_joint_state": np.array([0.5], dtype=np.float32),
    }


def _observation(head=None, left=None, right=None, **extra):
    vision = {}
    if head is not None:
        vision["cam_head"] = {"color": head}
    if left is not None:
        vision["cam_left_wrist"] = {"color": left}
    if right is not None:
        vision["cam_right_wrist"] = {"color": right}
    obs = {"vision": vision, "state": _valid_state(), "instruction": "test"}
    obs.update(extra)
    return obs


# RED / GREEN / BLUE 三种纯色，用于验证视角顺序。
HEAD_RGB = (255, 0, 0)     # cam_head   全红
LEFT_RGB = (0, 255, 0)     # cam_left_wrist 全绿
RIGHT_RGB = (0, 0, 255)    # cam_right_wrist 全蓝


def _mean_rgb(img):
    return tuple(int(round(float(v))) for v in np.asarray(img).mean(axis=(0, 1)))


def test_three_views_order_matches_config():
    """3 路：images 顺序必须严格等于 camera_names 顺序。"""
    obs = _observation(
        head=_solid_image(HEAD_RGB),
        left=_solid_image(LEFT_RGB),
        right=_solid_image(RIGHT_RGB),
    )
    out = encode_obs(
        obs, "test",
        camera_names=["cam_head", "cam_left_wrist", "cam_right_wrist"],
    )
    assert len(out["images"]) == 3, len(out["images"])
    assert _mean_rgb(out["images"][0]) == HEAD_RGB, _mean_rgb(out["images"][0])
    assert _mean_rgb(out["images"][1]) == LEFT_RGB, _mean_rgb(out["images"][1])
    assert _mean_rgb(out["images"][2]) == RIGHT_RGB, _mean_rgb(out["images"][2])


def test_two_views_subset():
    """2 路：只取配置的两个相机，顺序保持。"""
    obs = _observation(
        head=_solid_image(HEAD_RGB),
        left=_solid_image(LEFT_RGB),
        right=_solid_image(RIGHT_RGB),
    )
    out = encode_obs(
        obs, "test",
        camera_names=["cam_head", "cam_right_wrist"],
    )
    assert len(out["images"]) == 2, len(out["images"])
    assert _mean_rgb(out["images"][0]) == HEAD_RGB
    assert _mean_rgb(out["images"][1]) == RIGHT_RGB


def test_one_view():
    """1 路：仅 cam_head。"""
    obs = _observation(
        head=_solid_image(HEAD_RGB),
        left=_solid_image(LEFT_RGB),
        right=_solid_image(RIGHT_RGB),
    )
    out = encode_obs(obs, "test", camera_names=["cam_head"])
    assert len(out["images"]) == 1, len(out["images"])
    assert _mean_rgb(out["images"][0]) == HEAD_RGB


def test_camera_names_priority_over_legacy():
    """配置 camera_names 时忽略 legacy 顶层 images dict（若存在）。"""
    obs = _observation(
        head=_solid_image(HEAD_RGB),
        left=_solid_image(LEFT_RGB),
        right=_solid_image(RIGHT_RGB),
    )
    obs["images"] = {"cam_high": _solid_image((9, 9, 9))}
    out = encode_obs(
        obs, "test",
        camera_names=["cam_head", "cam_left_wrist", "cam_right_wrist"],
    )
    assert len(out["images"]) == 3
    assert _mean_rgb(out["images"][0]) == HEAD_RGB


def test_no_camera_names_legacy_single_cam_head():
    """不配置 camera_names（None）：保持原行为，vision 候选取 cam_head 单路。"""
    obs = _observation(
        head=_solid_image(HEAD_RGB),
        left=_solid_image(LEFT_RGB),
        right=_solid_image(RIGHT_RGB),
    )
    out = encode_obs(obs, "test")
    assert len(out["images"]) == 1, len(out["images"])
    assert _mean_rgb(out["images"][0]) == HEAD_RGB


def test_no_camera_names_legacy_top_level_images():
    """不配置 camera_names + 顶层 images dict：走第一分支取 cam_high。"""
    obs = _observation(head=_solid_image(HEAD_RGB))
    obs["images"] = {"cam_high": _solid_image((7, 7, 7))}
    out = encode_obs(obs, "test")
    assert len(out["images"]) == 1
    assert _mean_rgb(out["images"][0]) == (7, 7, 7)


def test_no_camera_names_empty_list_falls_back():
    """配置为空列表 []：等效不配置，回退 legacy 单路。"""
    obs = _observation(
        head=_solid_image(HEAD_RGB),
        left=_solid_image(LEFT_RGB),
        right=_solid_image(RIGHT_RGB),
    )
    out = encode_obs(obs, "test", camera_names=[])
    assert len(out["images"]) == 1
    assert _mean_rgb(out["images"][0]) == HEAD_RGB


def test_missing_camera_raises_keyerror():
    """camera_names 中相机不存在：必须显式报错，禁止静默落错视角。"""
    obs = _observation(
        head=_solid_image(HEAD_RGB),
        left=_solid_image(LEFT_RGB),
        right=_solid_image(RIGHT_RGB),
    )
    try:
        encode_obs(obs, "test", camera_names=["cam_head", "cam_missing"])
    except KeyError as e:
        assert "cam_missing" in str(e)
    else:
        raise AssertionError("expected KeyError for missing camera")


def test_camera_without_color_field_raises():
    """相机 dict 缺 color/rgb：显式报错。"""
    obs = _observation(head=_solid_image(HEAD_RGB))
    obs["vision"]["cam_left_wrist"] = {"depth": np.zeros((8, 8), np.uint16)}
    try:
        encode_obs(obs, "test", camera_names=["cam_head", "cam_left_wrist"])
    except KeyError as e:
        assert "cam_left_wrist" in str(e)
    else:
        raise AssertionError("expected KeyError for missing color/rgb field")


def test_float_images_clamped_to_uint8():
    """float [0,1] 图像被 clip + 缩放为 uint8（与 legacy 行为一致）。"""
    obs = _observation(head=_solid_image(HEAD_RGB))
    # 0.5 → 127.5 → uint8 127/128；断言 dtype 且均值在 [126, 129]。
    obs["vision"]["cam_head"]["color"] = np.full((8, 8, 3), 0.5, dtype=np.float64)
    out = encode_obs(obs, "test", camera_names=["cam_head"])
    img = out["images"][0]
    assert img.dtype == np.uint8, img.dtype
    assert np.all((img >= 126) & (img <= 129)), img.min()


def test_images_are_distinct_arrays():
    """多路输出互不共享底层内存，后续处理不会相互污染。"""
    obs = _observation(
        head=_solid_image(HEAD_RGB),
        left=_solid_image(LEFT_RGB),
        right=_solid_image(RIGHT_RGB),
    )
    out = encode_obs(
        obs, "test",
        camera_names=["cam_head", "cam_left_wrist", "cam_right_wrist"],
    )
    a, b, c = out["images"]
    assert a is not b and b is not c
    b[0, 0] = 0  # 修改一路不影响其他路
    assert _mean_rgb(a) == HEAD_RGB
    assert _mean_rgb(c) == RIGHT_RGB


if __name__ == "__main__":
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
