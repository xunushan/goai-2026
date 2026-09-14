"""pace.py 本地单元测试（无 GPU / 无 torch 依赖）。

运行：python test_pace.py   （也可用 pytest 收集 test_*）
"""
import sys
import traceback

import numpy as np

from pace import (
    ARM_XYZ_IDX,
    PaceConfig,
    PaceSelector,
    arm_speed_profile,
    select_execution_horizon,
    smooth_profile,
    valley_prominences,
)

LEFT_XYZ = list(ARM_XYZ_IDX["left"])
RIGHT_XYZ = list(ARM_XYZ_IDX["right"])


def _chunk(speeds, *, right_speeds=None, dim=20):
    """把逐步速度序列铺成 chunk：单轴直线运动，v[j] = speeds[j]。"""
    rows = len(speeds) + 1
    chunk = np.zeros((rows, dim), dtype=np.float64)
    x = 0.0
    for j in range(rows):
        chunk[j, LEFT_XYZ[0]] = x
        if j < len(speeds):
            x += speeds[j]
    for j, speed in enumerate(right_speeds or []):
        chunk[j + 1, RIGHT_XYZ[1]] = chunk[j, RIGHT_XYZ[1]] + speed
    if right_speeds and len(right_speeds) < rows - 1:
        chunk[len(right_speeds) + 1 :, RIGHT_XYZ[1]] = chunk[len(right_speeds), RIGHT_XYZ[1]]
    return chunk


# 测试用 δ_T：必须大于浮点噪声（累加求坐标会带来 ~1e-18 的哑差异），
# 又小于真实谷深（~1e-3 m）。真实部署的 δ_T 由标定给出，恒为正，见
# test_zero_threshold_accepts_float_noise 的说明。
DELTA_T = 1e-4


def _cfg(**kwargs):
    defaults = dict(enabled=True, h_max=20, h_min=1, d_min=1, smooth="none", smooth_window=1)
    defaults.update(kwargs)
    return PaceConfig(**defaults)


# ---------------------------------------------------------------------------
# 速度剖面 / 平滑
# ---------------------------------------------------------------------------

def test_speed_profile_index_convention():
    """v[j] = ||a[j+1] - a[j]||，长度 L-1。"""
    chunk = np.zeros((4, 20))
    chunk[0, LEFT_XYZ[0]] = 0.0
    chunk[1, LEFT_XYZ[0]] = 0.3
    chunk[2, LEFT_XYZ[0]] = 0.3
    chunk[3, LEFT_XYZ[0]] = 0.4
    profile = arm_speed_profile(chunk, ARM_XYZ_IDX["left"])
    assert profile.shape == (3,), profile.shape
    assert np.allclose(profile, [0.3, 0.0, 0.1]), profile


def test_speed_profile_uses_only_that_arm():
    """另一臂运动不影响本臂剖面。"""
    chunk = _chunk([0.02] * 5, right_speeds=[0.5] * 5)
    left = arm_speed_profile(chunk, ARM_XYZ_IDX["left"])
    right = arm_speed_profile(chunk, ARM_XYZ_IDX["right"])
    assert np.allclose(left, 0.02), left
    assert np.allclose(right, 0.5), right


def test_smooth_profile_preserves_length_and_edges():
    """边缘复制填充：长度不变，常值序列平滑后不变。"""
    flat = np.full(9, 0.03)
    out = smooth_profile(flat, 5, "moving_average")
    assert out.shape == flat.shape
    assert np.allclose(out, 0.03), out


def test_smooth_profile_none_is_identity():
    profile = np.array([0.1, 0.5, 0.2])
    assert np.allclose(smooth_profile(profile, 5, "none"), profile)


def test_smooth_profile_window_larger_than_profile():
    """窗口大于序列长度时自动收缩到不超过长度的奇数，不应报错或缩短序列。"""
    profile = np.array([0.1, 0.4, 0.2, 0.3])
    out = smooth_profile(profile, 99, "moving_average")
    assert out.shape == profile.shape
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# prominence
# ---------------------------------------------------------------------------

def test_valley_prominence_hand_computed():
    """Φ = min(左包围最高速, 右包围最高速) - 谷底速度。"""
    profile = np.array([0.02, 0.02, 0.006, 0.002, 0.006, 0.02, 0.02, 0.02])
    idxs, proms = valley_prominences(profile, lo=0, hi=profile.size - 1)
    assert idxs == [3], idxs
    assert abs(proms[0] - (0.02 - 0.002)) < 1e-12, proms


def test_valley_on_slow_segment_has_small_prominence():
    """整体都很慢时的浅谷 prominence 小 —— 论文 §3.3 要避免这种伪边界。"""
    shallow = np.array([0.004, 0.004, 0.0039, 0.004, 0.004])
    deep = np.array([0.05, 0.05, 0.001, 0.05, 0.05])
    _, p_shallow = valley_prominences(shallow, lo=0, hi=shallow.size - 1)
    _, p_deep = valley_prominences(deep, lo=0, hi=deep.size - 1)
    assert p_shallow[0] < 1e-3, p_shallow
    assert p_deep[0] > 0.04, p_deep


def test_valley_prominence_respects_hi_bound():
    """搜索范围被 hi 截断：hi 之外的谷值不出现。"""
    profile = np.array([0.02, 0.02, 0.001, 0.02, 0.02, 0.02, 0.002, 0.02, 0.02])
    idxs_all, _ = valley_prominences(profile, lo=0, hi=profile.size - 1)
    idxs_cut, _ = valley_prominences(profile, lo=0, hi=5)
    assert idxs_all == [2, 6], idxs_all
    assert idxs_cut == [2], idxs_cut


def test_valley_prominence_needs_both_neighbours():
    """严格局部极小：平台/单调段不产生候选。"""
    flat = np.full(6, 0.01)
    assert valley_prominences(flat, 0, 5) == ([], [])
    assert valley_prominences(np.zeros(2), 0, 1) == ([], [])


# ---------------------------------------------------------------------------
# horizon 选择
# ---------------------------------------------------------------------------

def test_no_valley_falls_back_to_h_max():
    """匀速运动无真实谷值 → 回落 H_max（论文 §3.3）。"""
    chunk = _chunk([0.02] * 19)
    decision = select_execution_horizon(chunk, _cfg(), delta_t=DELTA_T)
    assert decision.h == 20, decision.as_dict()
    assert decision.fallback is True


def test_zero_threshold_accepts_float_noise():
    """δ_T = 0 会把浮点哑差异当谷值 —— 记录为已知陷阱（不是 bug）。

    匀速轨迹经累加求坐标后 diff 出 ~1e-18 的哑差异，prominence >= 0 即被接受。
    真实标定（ρ>0）给出的 δ_T 恒为正且远大于该噪声，故部署中不会触发；
    但不要手工把 threshold_delta 配成 0。
    """
    chunk = _chunk([0.02] * 19)
    noisy = select_execution_horizon(chunk, _cfg(), delta_t=0.0)
    clean = select_execution_horizon(chunk, _cfg(), delta_t=DELTA_T)
    assert noisy.h < 20 and noisy.fallback is False
    assert max(c["prominence"] for c in noisy.candidates) < 1e-12
    assert clean.h == 20 and clean.fallback is True


def test_valley_selects_its_index_plus_one():
    """谷值在 j → 执行 h = j+1 步后在边界重规划。"""
    speeds = [0.02] * 5 + [0.012, 0.006, 0.002, 0.006, 0.012] + [0.02] * 9
    chunk = _chunk(speeds)
    decision = select_execution_horizon(chunk, _cfg(), delta_t=DELTA_T)
    accepted = [c["h"] for c in decision.candidates if c["accepted"]]
    assert accepted == [8], decision.candidates
    assert decision.h == 8, decision.as_dict()
    assert decision.fallback is False


def test_smoothing_keeps_valley_nearby():
    """开启平滑（默认 S）后谷值位置不应跑偏。"""
    speeds = [0.02] * 6 + [0.012, 0.006, 0.001, 0.006, 0.012] + [0.02] * 9
    chunk = _chunk(speeds)
    decision = select_execution_horizon(
        chunk, _cfg(smooth="moving_average", smooth_window=5), delta_t=DELTA_T
    )
    assert 6 <= decision.h <= 12, decision.as_dict()


def test_high_threshold_rejects_all_candidates():
    """δ_T 越高接受越少（Table 3）：足够高时退化为 H_max。"""
    speeds = [0.02] * 5 + [0.012, 0.006, 0.002, 0.006, 0.012] + [0.02] * 9
    chunk = _chunk(speeds)
    accepted = select_execution_horizon(chunk, _cfg(), delta_t=0.001)
    rejected = select_execution_horizon(chunk, _cfg(), delta_t=10.0)
    assert accepted.h == 8
    assert rejected.h == 20 and rejected.fallback is True


def test_threshold_missing_falls_back_instead_of_guessing():
    """无标定 δ_T 时不应瞎猜阈值：直接回落 H_max 并置标记。"""
    speeds = [0.02] * 5 + [0.002] + [0.02] * 13
    chunk = _chunk(speeds)
    decision = select_execution_horizon(chunk, _cfg(), delta_t=None)
    assert decision.h == 20
    assert decision.threshold_missing is True
    assert all(c["accepted"] is False for c in decision.candidates)


def test_d_min_drops_nearby_weaker_valley():
    """d_min 生效时同一簇只保留最显著的谷值（改变最早的 h）。"""
    speeds = [0.02, 0.02, 0.02, 0.015, 0.02, 0.02, 0.001, 0.02, 0.02, 0.02] + [0.02] * 9
    chunk = _chunk(speeds)
    tight = select_execution_horizon(chunk, _cfg(d_min=2), delta_t=DELTA_T)
    wide = select_execution_horizon(chunk, _cfg(d_min=10), delta_t=DELTA_T)
    assert tight.h == 4, tight.as_dict()          # 浅谷 h=4 + 深谷 h=7 都保留
    assert wide.h == 7, wide.as_dict()            # 深谷(Φ更大)胜出，浅谷被间距规则剔除


def test_h_min_clamps_up():
    """h_min 是成本保护：选出过早的边界时上抬到 h_min。"""
    speeds = [0.005, 0.001] + [0.02] * 17
    chunk = _chunk(speeds)
    decision = select_execution_horizon(chunk, _cfg(h_min=5), delta_t=DELTA_T)
    assert decision.h == 5, decision.as_dict()


def test_multi_arm_takes_earliest_boundary():
    """多臂取最早的可接受边界（论文 §3.3）。"""
    left_speeds = [0.02] * 11 + [0.001] + [0.02] * 7
    right_speeds = [0.02] * 1 + [0.001] + [0.02] * 17
    chunk = _chunk(left_speeds, right_speeds=right_speeds)
    decision = select_execution_horizon(chunk, _cfg(), delta_t=DELTA_T)
    arms = {c["arm"]: c["h"] for c in decision.candidates if c["accepted"]}
    assert arms == {"left": 12, "right": 2}, decision.candidates
    assert decision.h == 2, decision.as_dict()


def test_h_max_capped_by_chunk_length():
    """cfg.h_max 大于实际 chunk 长度时按 chunk 长度截断。"""
    chunk = _chunk([0.02] * 9)
    decision = select_execution_horizon(chunk, _cfg(h_max=30), delta_t=DELTA_T)
    assert decision.h == 10, decision.as_dict()
    assert decision.h_max == 10


def test_h_max_limits_search_window():
    """h_max 之外的谷值不参与选择。"""
    speeds = [0.02] * 2 + [0.001] + [0.02] * 6 + [0.0005] + [0.02] * 8
    chunk = _chunk(speeds)
    narrow = select_execution_horizon(chunk, _cfg(h_max=6), delta_t=DELTA_T)
    wide = select_execution_horizon(chunk, _cfg(h_max=20), delta_t=DELTA_T)
    assert narrow.h == 3, narrow.as_dict()
    assert wide.h == 3, wide.as_dict()


# ---------------------------------------------------------------------------
# 配置解析 / Selector
# ---------------------------------------------------------------------------

def test_config_defaults_track_actions_per_chunk():
    """h_max 缺省时跟随 actions_per_chunk（= 论文的固定 horizon 基线）。"""
    cfg = PaceConfig.from_model_cfg({}, 30)
    assert cfg.enabled is False
    assert cfg.h_max == 30
    assert cfg.h_min == 5
    assert cfg.d_min == 10
    assert cfg.threshold_for("anything") is None


def test_config_parses_deploy_yml_shape():
    cfg = PaceConfig.from_model_cfg(
        {
            "pace": {
                "enabled": "true",
                "h_max": 24,
                "h_min": 3,
                "d_min": 8,
                "smooth": "moving_average",
                "smooth_window": 7,
                "threshold_delta": 0.002,
                "threshold_delta_by_task": {"stack_bowls": 0.0015},
            }
        },
        30,
    )
    assert cfg.enabled is True and cfg.h_max == 24 and cfg.h_min == 3
    assert cfg.d_min == 8 and cfg.smooth_window == 7
    assert cfg.threshold_for("stack_bowls") == 0.0015
    assert cfg.threshold_for("stack_blocks") == 0.002
    assert cfg.threshold_for(None) == 0.002


def test_config_null_h_max_falls_back_to_actions_per_chunk():
    cfg = PaceConfig.from_model_cfg({"pace": {"enabled": True, "h_max": None}}, 12)
    assert cfg.h_max == 12


def test_config_does_not_break_small_actions_per_chunk():
    """向后兼容：缺省 pace 段 + 既有 actions_per_chunk < 默认 h_min(5) 不得报错。

    h_max 缺省跟随 actions_per_chunk，若不把缺省 h_min 夹到 h_max，既有的
    actions_per_chunk=3 配置会在 __init__ 抛「h_min must be <= h_max」。
    """
    for chunk in (1, 3, 4, 5):
        cfg = PaceConfig.from_model_cfg({}, chunk)
        assert cfg.enabled is False
        assert cfg.h_max == chunk
        assert cfg.h_min == min(5, chunk)
    # 但显式配置 h_min > h_max 仍需报错
    try:
        PaceConfig.from_model_cfg({"pace": {"enabled": True, "h_min": 9}}, 8)
    except ValueError:
        pass
    else:
        raise AssertionError("explicit h_min > h_max must be rejected")


def test_config_rejects_bad_values():
    for bad in (
        {"pace": {"enabled": True, "h_min": 0}},
        {"pace": {"enabled": True, "h_min": 31, "h_max": 30}},
        {"pace": {"enabled": True, "d_min": 0}},
        {"pace": {"enabled": True, "smooth": "gaussian"}},
        {"pace": {"enabled": True, "arms": ["middle"]}},
        {"pace": {"enabled": True, "threshold_delta_by_task": {"t": -1.0}}},
    ):
        try:
            PaceConfig.from_model_cfg(bad, 30)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"expected rejection for {bad}")


def test_selector_warns_once_per_task_and_is_stateless():
    """Selector 只缓存「已告警任务」，reset() 后重新告警。"""
    selector = PaceSelector(PaceConfig(enabled=True, h_max=20, h_min=1, smooth="none"))
    chunk = _chunk([0.02] * 19)
    first = selector.select(chunk, task_name="stack_blocks")
    assert first.threshold_missing is True and first.h == 20
    assert selector._warned_missing_threshold == {"stack_blocks"}
    selector.reset()
    assert selector._warned_missing_threshold == set()


def test_selector_uses_per_task_threshold():
    cfg = PaceConfig(
        enabled=True,
        h_max=20,
        h_min=1,
        d_min=1,
        smooth="none",
        threshold_delta=10.0,
        threshold_delta_by_task={"stack_bowls": 0.001},
    )
    selector = PaceSelector(cfg)
    speeds = [0.02] * 5 + [0.012, 0.006, 0.002, 0.006, 0.012] + [0.02] * 9
    chunk = _chunk(speeds)
    assert selector.select(chunk, task_name="stack_bowls").h == 8
    assert selector.select(chunk, task_name="stack_blocks").h == 20


def test_decision_dict_is_json_friendly():
    import json

    chunk = _chunk([0.02] * 19)
    decision = select_execution_horizon(chunk, _cfg(), delta_t=DELTA_T, task_name="t")
    json.dumps(decision.as_dict())
    assert decision.as_dict()["h"] == 20


if __name__ == "__main__":
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
