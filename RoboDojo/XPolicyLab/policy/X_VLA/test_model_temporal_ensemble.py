"""Integration checks for X_VLA model.py temporal ensemble (sliding-window).

Stubs heavy/unavailable deps (torch/cv2/PIL/XPolicyLab/xvla/gripper_hysteresis),
loads the real model.py, and drives get_action_batch with a fake infer().

Run:  /opt/anaconda3/bin/python3 test_model_temporal_ensemble.py
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

_CUR_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------- stubs
def _make_module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


_make_module("cv2")
_make_module("PIL", Image=type("Image", (), {}))

torch_stub = _make_module("torch", Generator=type("Generator", (), {}))

_make_module("XPolicyLab", __path__=[])
_make_module(
    "XPolicyLab.model_template",
    ModelTemplate=type("ModelTemplate", (), {}),
)
_make_module("XPolicyLab.utils", __path__=[])
_make_module(
    "XPolicyLab.utils.checkpoint_resolver",
    resolve_checkpoint_root=lambda model_cfg, **kw: None,
)
_make_module(
    "XPolicyLab.utils.process_data",
    decode_image_bit=lambda value: value,
    get_robot_action_dim_info=lambda env_cfg: None,
)

_make_module("xvla", __path__=[])
_make_module("xvla.models", __path__=[])
_make_module(
    "xvla.models.modeling_xvla",
    XVLA=type("XVLA", (), {}),
)
_make_module(
    "xvla.models.processing_xvla",
    XVLAProcessor=type("XVLAProcessor", (), {}),
)


class _HCfg:
    enabled = False
    mode = "direction_latch"
    lo = 0.3
    hi = 0.7

    @classmethod
    def from_model_cfg(cls, model_cfg):
        cfg = model_cfg.get("hysteresis") or {}
        obj = cls()
        obj.enabled = bool(cfg.get("enabled", False))
        obj.mode = str(cfg.get("mode", "direction_latch"))
        obj.lo = float(cfg.get("lo", 0.3))
        obj.hi = float(cfg.get("hi", 0.7))
        return obj


_make_module(
    "gripper_hysteresis",
    HysteresisConfig=_HCfg,
    apply_gripper_hysteresis=lambda *a, **k: None,
)

# temporal_ensemble is a real numpy-only sibling module.
sys.path.insert(0, str(_CUR_DIR))

spec = importlib.util.spec_from_file_location("x_vla_model", str(_CUR_DIR / "model.py"))
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)


# ---------------------------------------------------------------- helpers
class _FakeProcessor:
    pass


class _FakeModel:
    num_actions = 30

    def eval(self):
        return None


class _TestModel(model_mod.Model):
    def _get_device(self, device_arg):
        return "device-sentinel"

    def _load_processor(self, model_cfg):
        return _FakeProcessor()

    def _load_model(self, model_cfg):
        return _FakeModel()


BASE_CFG = {
    "action_type": "ee",
    "device": "cpu",
    "gripper_mode": "continuous",
    "gripper_threshold": 0.7,
    "steps": 10,
    "log_io": False,
    "policy_seed": None,
}


def make_model(cfg):
    merged = dict(BASE_CFG)
    merged.update(cfg)
    return _TestModel(merged)


def prime_action(m, chunk, env_idx=0):
    """Set minimal internal state so get_action_batch can run once."""
    m.observation_window = [{"env_idx": env_idx}]
    m._latest_by_env = {env_idx: {"images": [], "proprio": np.zeros(20)}}
    m._raw_by_env = {env_idx: {"state": {}}}
    m._latest_env_idx_list = [env_idx]

    def fake_infer(observation, steps=None, generator=None):
        return np.asarray(chunk, dtype=np.float32)

    m.infer = types.MethodType(fake_infer, m)


def chunk_for(step, model_chunk_size=30):
    """Deterministic [30,20] chunk; row[i] carries value step*1000 + i.

    The re-plan step identity dominates so weighted averages over predictions
    made at different re-plans are distinguishable.
    """
    rows = []
    for i in range(model_chunk_size):
        row = np.zeros(20, dtype=np.float32)
        row[0] = step * 1000 + i   # left xyz x — carries re-plan+row value
        row[3:6] = 0.01            # left rotate6d (a1 = a2 along +x)
        row[6:9] = 0.01
        row[9] = 0.9               # left gripper open
        row[10] = step * 1000 + i
        row[13:16] = 0.01
        row[16:19] = 0.01
        row[19] = 0.9
        rows.append(row)
    return np.stack(rows)


# ---------------------------------------------------------------- tests
def expect_value_error(cfg, fragment, label):
    try:
        make_model(cfg)
    except ValueError as exc:
        if fragment in str(exc):
            print(f"[PASS] {label}")
            return True
        print(f"[FAIL] {label}: wrong message: {exc}")
        return False
    print(f"[FAIL] {label}: expected ValueError")
    return False


def test_validation():
    ok = True
    ok &= expect_value_error(
        {"actions_per_chunk": 0, "temporal_ensemble_coeff": 0.01},
        "actions_per_chunk", "K=0 rejected",
    )
    ok &= expect_value_error(
        {"actions_per_chunk": 31, "temporal_ensemble_coeff": 0.01},
        "actions_per_chunk", "K=31 rejected",
    )
    ok &= expect_value_error(
        {"actions_per_chunk": 5, "temporal_ensemble_coeff": 0.01,
         "temporal_ensemble_horizon": 1},
        "temporal_ensemble_horizon", "horizon=1 rejected",
    )
    ok &= expect_value_error(
        {"actions_per_chunk": 5, "temporal_ensemble_coeff": 0.01,
         "temporal_ensemble_horizon": 31},
        "temporal_ensemble_horizon", "horizon=31 rejected",
    )
    ok &= expect_value_error(
        {"actions_per_chunk": 5, "temporal_ensemble_coeff": float("inf")},
        "temporal_ensemble_coeff", "coeff=inf rejected",
    )
    return ok


def run_sliding(m, n_replans, model_chunk_size=30):
    """Run n_replans re-plans spaced actions_per_chunk apart; return outputs."""
    all_chunks = []
    for s in range(n_replans):
        step = s * m.actions_per_chunk
        prime_action(m, chunk_for(step, model_chunk_size))
        actions = m.get_action_batch()[0]
        chunk = np.stack([
            np.concatenate([
                a["left_ee_pose"][:1],        # left xyz x (row value carried)
            ])
            for a in actions
        ])
        all_chunks.append(chunk.reshape(-1))
    return all_chunks


def test_active_sliding():
    # K=5, horizon=30, coeff=0.01: active, returns 5 actions per re-plan.
    m = make_model({
        "actions_per_chunk": 5,
        "temporal_ensemble_coeff": 0.01,
        "temporal_ensemble_horizon": 30,
    })
    assert m.temporal_ensemble_active is True, "K < horizon should be active"
    out = run_sliding(m, 4)  # re-plans at steps 0,5,10,15
    # First re-plan is verbatim (single prediction): c0[0..4]
    assert np.allclose(out[0], [0, 1, 2, 3, 4]), f"first re-plan raw, got {out[0]}"
    # Oldest prediction is weighted most (w0 = 1 = exp(-0*coeff)).
    # Second re-plan: step 5 averages c0[5] (oldest, w0) and c5[0] (newest, w1)
    w0, w1 = 1.0, float(np.exp(-0.01))
    expected5 = (w0 * 5 + w1 * 5000) / (w0 + w1)
    assert abs(out[1][0] - expected5) < 1e-3, f"step5 {out[1][0]} != {expected5}"
    # Third re-plan: step 10 averages c0[10] (w0), c5[5] (w1), c10[0] (w2)
    w2 = float(np.exp(-0.02))
    expected10 = (w0 * 10 + w1 * 5005 + w2 * 10000) / (w0 + w1 + w2)
    assert abs(out[2][0] - expected10) < 1e-3, f"step10 {out[2][0]} != {expected10}"
    # Fourth re-plan: step 15 averages c0[15] (w0), c5[10] (w1), c10[5] (w2),
    # c15[0] (w3)
    w3 = float(np.exp(-0.03))
    expected15 = (w0 * 15 + w1 * 5010 + w2 * 10005 + w3 * 15000) / (
        w0 + w1 + w2 + w3
    )
    assert abs(out[3][0] - expected15) < 1e-3, f"step15 {out[3][0]} != {expected15}"
    print("[PASS] active sliding: K=5 returns 5 actions, weighted averages align")
    return True


def test_no_overlap_direct():
    # K=30 (== horizon): ensemble disabled, raw chunk direct.
    m = make_model({
        "actions_per_chunk": 30,
        "temporal_ensemble_coeff": 0.01,
        "temporal_ensemble_horizon": 30,
    })
    assert m.temporal_ensemble_active is False, "K == horizon should be inactive"
    out = run_sliding(m, 2)
    assert np.allclose(out[0], np.arange(30)), f"direct 0..29, got {out[0]}"
    assert np.allclose(out[1], np.arange(30000, 30030)), \
        f"direct rows of c30, got {out[1]}"
    assert m._temporal_ensemblers == {}, "no ensembler state should be created"
    print("[PASS] K == horizon: raw chunk direct, ensemble off")
    return True


def test_horizon_less_than_chunk():
    # horizon=10, K=5: active; chunk fed is sliced to horizon=10.
    m = make_model({
        "actions_per_chunk": 5,
        "temporal_ensemble_coeff": 0.5,
        "temporal_ensemble_horizon": 10,
    })
    assert m.temporal_ensemble_active is True
    out = run_sliding(m, 3)  # replans 0,5,10
    # step 5 average of c0[5] (oldest, w0) and c5[0] (newest, w1)
    w0, w1 = 1.0, float(np.exp(-0.5))
    expected5 = (w0 * 5 + w1 * 5000) / (w0 + w1)
    assert abs(out[1][0] - expected5) < 1e-4, f"step5 {out[1][0]} != {expected5}"
    print("[PASS] horizon < model_chunk_size: sliced horizon still aligns")
    return True


def test_disabled_default():
    m = make_model({"actions_per_chunk": 30})
    assert m.temporal_ensemble_active is False
    assert m.temporal_ensemble_coeff is None
    out = run_sliding(m, 1)
    assert np.allclose(out[0], np.arange(30))
    print("[PASS] default (no coeff): disabled, chunk direct")
    return True


def test_reset_clears():
    m = make_model({
        "actions_per_chunk": 5,
        "temporal_ensemble_coeff": 0.01,
        "temporal_ensemble_horizon": 30,
    })
    run_sliding(m, 2)
    assert len(m._temporal_ensemblers) == 1
    m.reset()
    assert m._temporal_ensemblers == {}, "reset should clear ensemblers"
    print("[PASS] reset() clears per-env ensembler state")
    return True


def test_counts_ramp():
    # horizon=10, K=1: counts ramp 1->10 then stay saturated; no OOB.
    m = make_model({
        "actions_per_chunk": 1,
        "temporal_ensemble_coeff": 0.01,
        "temporal_ensemble_horizon": 10,
    })
    assert m.temporal_ensemble_active is True
    counts = []
    for s in range(15):
        prime_action(m, chunk_for(s, 30))
        m.get_action_batch()
        ens = m._temporal_ensemblers[0]
        counts.append(ens.last_prediction_count)
    assert counts[0] == 1
    assert max(counts) == 10, f"counts should saturate at horizon=10, got {max(counts)}"
    assert counts[9] == 10 and counts[14] == 10, f"steady-state {counts}"
    print("[PASS] K=1 counts ramp 1->10 and saturate without index OOB")
    return True


def main():
    ok = True
    ok &= test_validation()
    ok &= test_active_sliding()
    ok &= test_no_overlap_direct()
    ok &= test_horizon_less_than_chunk()
    ok &= test_disabled_default()
    ok &= test_reset_clears()
    ok &= test_counts_ramp()
    print("ALL INTEGRATION TESTS PASSED" if ok else "SOME INTEGRATION TESTS FAILED")


if __name__ == "__main__":
    main()
