"""Pi_05 model.py 日志与 checkpoint 解析 —— 本地单元测试（无 GPU，stub openpi）。

覆盖本次新增的 [pi05][io] 日志纯函数与 Model._log_request 输出契约，以及
_resolve_pi05_model_root 的候选目录选择逻辑。模型权重加载（需 GPU/openpi ckpt）不测。

运行：
    python test_model_logging.py      # 直跑（无 pytest 依赖）
    pytest test_model_logging.py      # 亦可
"""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

# ---- 目录定位：测试文件位于 RoboDojo/XPolicyLab/policy/Pi_05/ ----
_PI05_DIR = Path(__file__).resolve().parent
_ROBO_ROOT = _PI05_DIR.parents[2]  # _PI05_DIR 已是文件父目录 → parents[0]=policy, [2]=RoboDojo
_MODEL_PY = _PI05_DIR / "model.py"


def _stub_openpi() -> None:
    """注入 openpi 空模块，避免触发 jax/cuda（无 GPU 环境亦不可 import 权重）。"""
    _ROBO_ROOT  # noqa: B018  (确保模块级已计算)
    for name in [
        "openpi",
        "openpi.policies",
        "openpi.policies.policy_config",
        "openpi.shared",
        "openpi.shared.normalize",
        "openpi.training",
        "openpi.training.config",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))


def _load_model_module():
    _stub_openpi()
    if str(_ROBO_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROBO_ROOT))  # import XPolicyLab.*
    spec = importlib.util.spec_from_file_location("pi05_model_test", _MODEL_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = _load_model_module()
_finite_round = m._finite_round
_array_log = m._array_log
_action_log = m._action_log
Model = m.Model
_resolve_pi05_model_root = m._resolve_pi05_model_root


# ---------------------------------------------------------------- helpers
def _capture_log_request(**attrs):
    """构造未初始化 Model 实例调用 _log_request，返回解析后的两行 JSON dict。"""
    obj = object.__new__(Model)
    obj._encoded_obs_by_env = attrs.pop("encoded", {})
    obj._raw_obs_by_env = attrs.pop("raw", {})
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        obj._log_request(
            attrs.pop("request", 0),
            attrs.pop("env_idx", 0),
            attrs.pop("raw_actions", np.zeros((1, 14))),
            attrs.pop("action", {}),
        )
    lines = [ln for ln in buffer.getvalue().strip().splitlines() if ln]
    assert len(lines) == 2, f"expected 2 log lines, got {len(lines)}"
    return [json.loads(ln.split(" ", 1)[1]) for ln in lines]


# ---------------------------------------------------------------- tests
def test_finite_round():
    assert _finite_round(1.23456789) == 1.2346
    assert _finite_round(-0.0) == -0.0
    assert _finite_round(np.float32(2.5)) == 2.5
    assert _finite_round(np.nan) is None
    assert _finite_round(np.inf) is None
    assert _finite_round(-np.inf) is None
    assert _finite_round("abc") is None
    assert _finite_round(None) is None


def test_array_log_basic():
    res = _array_log(np.array([1.0, 2.0, 3.0]))
    assert res["shape"] == [3]
    assert res["n_nan_inf"] == 0
    assert res["min"] == 1.0 and res["max"] == 3.0 and res["mean"] == 2.0
    assert res["values"] == [1.0, 2.0, 3.0]


def test_array_log_nan_inf():
    res = _array_log(np.array([1.0, np.nan, np.inf, 3.0]))
    assert res["n_nan_inf"] == 2  # nan + inf 各计
    assert res["min"] == 1.0 and res["max"] == 3.0  # 有限值范围
    assert res["values"] == [1.0, None, None, 3.0]  # 非有限值显示 None


def test_array_log_empty_and_ndim():
    empty = _array_log(np.array([]))
    assert "n_nan_inf" not in empty and "values" not in empty
    two_d = _array_log(np.arange(6).reshape(2, 3))
    assert two_d["shape"] == [2, 3]
    assert len(two_d["values"]) == 6


def test_array_log_truncate_values():
    res = _array_log(np.arange(40, dtype=np.float32))
    assert len(res["values"]) == 24  # 截断在前 24 个


def test_action_log_forms():
    # dict（unpack_robot_state source_type=obs 的返回）
    d = _action_log({"arm_0": np.zeros(5), "ee_0": np.array([1.0])})
    assert set(d) == {"arm_0", "ee_0"}
    assert d["ee_0"]["shape"] == [1] and d["ee_0"]["values"] == [1.0]
    # list[dict]（多步 chunk）
    steps = _action_log([{"arm_0": np.zeros(2)}, {"arm_0": np.ones(2)}])
    assert steps["n_steps"] == 2 and "step0" in steps
    # 裸 ndarray
    arr = _action_log(np.zeros(14))
    assert arr["shape"] == [14]


def test_log_request_observation_event():
    raw_actions = np.zeros((1, 14))
    encoded = {
        3: {
            "state": np.full(16, 0.5, dtype=np.float32),
            "images": {
                "cam_high": np.zeros((3, 480, 640), dtype=np.uint8),
                "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
                "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            },
            "prompt": "x" * 300,  # 应被截断到 200
        }
    }
    raw = {3: {"episode_idx": "c7b0fb6f", "task_name": "plug_in_charger"}}
    obs_line, act_line = _capture_log_request(
        request=5, env_idx=3, raw_actions=raw_actions, action={"ee_0": np.array([1.0])}, encoded=encoded, raw=raw
    )
    assert obs_line["event"] == "client_observation"
    assert obs_line["request"] == 5 and obs_line["env_idx"] == 3
    assert obs_line["episode_idx"] == "c7b0fb6f"
    assert obs_line["task_name"] == "plug_in_charger"
    assert len(obs_line["prompt"]) == 200  # 截断
    assert set(obs_line["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert obs_line["images"]["cam_high"]["shape"] == [3, 480, 640]
    assert obs_line["state"]["mean"] == 0.5 and obs_line["state"]["n_nan_inf"] == 0


def test_log_request_action_event_nan_detection():
    raw_actions = np.zeros((1, 14))
    raw_actions[0, 0] = np.nan
    encoded = {0: {"state": np.zeros(16), "images": {}, "prompt": ""}}
    raw = {0: {}}
    _, act_line = _capture_log_request(
        request=0, env_idx=0, raw_actions=raw_actions,
        action={"arm_0": np.zeros(5), "ee_0": np.array([1.0]), "arm_1": np.ones(5), "ee_1": np.array([0.0])},
        encoded=encoded, raw=raw,
    )
    assert act_line["event"] == "server_actions"
    assert act_line["request"] == 0 and act_line["env_idx"] == 0
    # raw chunk 首元素 nan → n_nan_inf=1
    assert act_line["raw_actions"]["n_nan_inf"] == 1
    # unpack 后的动作各维均有限
    assert act_line["actions"]["arm_0"]["n_nan_inf"] == 0
    assert act_line["actions"]["ee_1"]["values"] == [0.0]


def _resolve_with_tmp_tree(files: dict[Path, str]):
    """files: {相对路径: 是否目录("dir")}，返回 (model_cfg, tmp_root)。"""
    tmp = Path(tempfile.mkdtemp(prefix="pi05_resolve_"))
    for rel, kind in files.items():
        target = tmp / rel
        if kind == "dir":
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
    return tmp


def test_resolve_model_root_direct_dir():
    tmp = _resolve_with_tmp_tree({"params/x": "f", "assets/y": "f"})
    try:
        root = _resolve_pi05_model_root({"model_path": str(tmp), "checkpoint_num": 59999})
        assert root == tmp.resolve()  # resolver 默认 resolve=True（macOS /var→/private/var）
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_model_root_step_selection():
    tmp = _resolve_with_tmp_tree({"59999/params/x": "f", "60000/params/x": "f"})
    try:
        root = _resolve_pi05_model_root({"model_path": str(tmp), "checkpoint_num": 59999})
        assert root.name == "59999", f"expected step dir 59999, got {root}"
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_model_root_no_dir_returns_leaf():
    # candidate 目录不存在时返回首个候选（即使不存在）→ 不 raise
    tmp = Path(tempfile.mkdtemp(prefix="pi05_resolve_missing_"))
    try:
        missing = tmp / "nope"
        root = _resolve_pi05_model_root({"model_path": str(missing), "checkpoint_num": 3})
        assert root == missing.resolve()  # 同上：resolve 前缀对齐
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- runner
def _run_all() -> int:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
