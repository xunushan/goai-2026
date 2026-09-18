"""Standalone verification of temporal_ensemble.ServerTemporalEnsembler.

Run with a numpy-enabled python:  /opt/anaconda3/bin/python3 test_temporal_ensemble.py

Checks:
  1. Sliding-window output == brute-force reference (re-plan every K steps).
  2. K=1 output == LeRobot ACTTemporalEnsembler byte-for-byte (numpy vs torch).
  3. Boundary cases: empty-window pop, bad chunk shape, long-run saturation
     without index-out-of-bounds on the exponential weight table.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporal_ensemble import ServerTemporalEnsembler


def chunk_gen_factory(H, D):
    def chunk_gen(s):
        rng = np.random.RandomState(s % 1000)
        base = np.linspace(s, s + H - 1, H).reshape(H, 1).repeat(D, axis=1)
        return base + 0.1 * rng.randn(H, D)

    return chunk_gen


def reference_ensemble(coeff, H, K, chunk_gen, total_steps):
    """Brute force: at step t, average all predictions aligned to t.

    Predictions are weighted by chronological order (oldest = w_0), matching
    LeRobot's ACTTemporalEnsembler count-based weights.
    """
    replan_times = list(range(0, total_steps, K))
    chunks = {s: chunk_gen(s) for s in replan_times}
    w = np.exp(-coeff * np.arange(H))
    actions, counts = [], []
    for t in range(total_steps):
        preds = []
        for s in replan_times:
            if s > t:
                break
            idx = t - s
            if 0 <= idx < H:
                preds.append(chunks[s][idx])
        n = len(preds)
        ws = w[:n]
        val = (
            sum(ws[i] * preds[i] for i in range(n)) / ws.sum()
            if n
            else np.full(preds[0].shape, np.nan)
        )
        actions.append(val)
        counts.append(n)
    return np.array(actions), np.array(counts)


def sliding_ensemble(coeff, H, K, chunk_gen, total_steps):
    ens = ServerTemporalEnsembler(coeff, H)
    actions, counts = [], []
    for t in range(total_steps):
        if t % K == 0:
            ens.update(chunk_gen(t))
        a = ens.pop()
        actions.append(a)
        counts.append(ens.last_prediction_count)
    return np.array(actions), np.array(counts)


def lerobot_ensemble(coeff, H, chunk_gen, total_steps):
    import torch
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    ens = ACTTemporalEnsembler(coeff, H)
    actions = []
    for t in range(total_steps):
        a = ens.update(torch.tensor(chunk_gen(t), dtype=torch.float32)[None])
        actions.append(a.numpy().reshape(-1))
    return np.array(actions)


def test_against_reference():
    D = 20
    failures = 0
    for coeff in (0.0, 0.01, 0.5):
        for H, Ks in ((2, (1,)), (3, (1, 2)), (10, (1, 2, 5)), (30, (1, 5, 29))):
            for K in Ks:
                for total in (5, 47, 100):
                    chunk_gen = chunk_gen_factory(H, D)
                    ref_a, ref_c = reference_ensemble(coeff, H, K, chunk_gen, total)
                    my_a, my_c = sliding_ensemble(coeff, H, K, chunk_gen, total)
                    if not np.allclose(ref_a, my_a, atol=1e-5):
                        failures += 1
                        print(
                            f"[FAIL] values coeff={coeff} H={H} K={K} total={total}"
                        )
                    if not np.array_equal(ref_c, my_c):
                        failures += 1
                        print(
                            f"[FAIL] counts coeff={coeff} H={H} K={K} total={total}"
                        )
    if failures:
        print(f"[FAIL] {failures} reference mismatches")
        return False
    print("[PASS] sliding-window output == brute-force reference "
          "(coeff 0/0.01/0.5 x H 2/3/10/30 x K 1..29 x total 5/47/100)")
    return True


def test_against_lerobot():
    D = 20
    H = 30
    total = 47
    for coeff in (0.01, 0.5):
        chunk_gen = chunk_gen_factory(H, D)
        le_a = lerobot_ensemble(coeff, H, chunk_gen, total)
        my_a, _ = sliding_ensemble(coeff, H, 1, chunk_gen, total)
        if not np.allclose(le_a, my_a, atol=1e-6):
            print(f"[FAIL] K=1 differs from LeRobot coeff={coeff}")
            return False
    print("[PASS] K=1 output == LeRobot ACTTemporalEnsembler (byte-for-byte)")
    return True


def test_boundaries():
    ens = ServerTemporalEnsembler(0.01, 5)
    # pop on empty window
    try:
        ens.pop()
        print("[FAIL] expected RuntimeError on empty-window pop")
        return False
    except RuntimeError:
        pass
    # update with wrong shape
    try:
        ens.update(np.zeros((4, 20)))
        print("[FAIL] expected ValueError on wrong chunk length")
        return False
    except ValueError:
        pass
    try:
        ens.update(np.zeros((5, 20, 1)))
        print("[FAIL] expected ValueError on wrong chunk rank")
        return False
    except ValueError:
        pass
    # update restores a fresh window after full consumption
    ens.update(np.zeros((5, 20)))
    for _ in range(5):
        ens.pop()
    ens.update(np.ones((5, 20)))
    assert np.allclose(ens.pop(), np.ones(20)), "fresh chunk should pop verbatim"
    # long-run saturation: no weight-table index-out-of-bounds, counts plateau
    for H, K in ((30, 1), (30, 5), (10, 2)):
        chunk_gen = chunk_gen_factory(H, 20)
        my_a, my_c = sliding_ensemble(0.01, H, K, chunk_gen, 200)
        if not np.isfinite(my_a).all():
            print(f"[FAIL] non-finite output H={H} K={K}")
            return False
        if my_c[-1] < 2:
            print(f"[FAIL] expected multi-prediction count near end H={H} K={K}, "
                  f"got {my_c[-1]}")
            return False
    print("[PASS] boundary cases: empty pop, bad shape, full-consume restart, "
          "200-step saturation without OOB")
    return True


def main():
    ok = True
    ok &= test_against_reference()
    try:
        ok &= test_against_lerobot()
    except ImportError as exc:
        print(f"[SKIP] LeRobot comparison unavailable: {exc}")
    ok &= test_boundaries()
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()
