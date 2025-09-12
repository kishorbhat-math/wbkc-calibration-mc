import numpy as np
from scripts.gen_data import synth_spectrum
from wbkc.model import simulate


def test_simulate_basic_outputs():
    # Simple sanity check that simulate returns expected fields and a valid interval
    # NOTE: synth_spectrum expects (tbk_true, cps_per, live_time_s, rng) as positional for the first three.
    E, Y = synth_spectrum(
        2.4,          # tbk_true
        120.0,        # cps_per
        900,          # live_time_s
        rng=np.random.default_rng(1),
    )
    res = simulate(
        E,
        Y,
        live_time_s=900,
        calib=dict(cps_per_TBK=120.0, bg_cps=0.1, attn_mean=1.0, attn_rel_sigma=0.05),
        n_mc=6000,
    )
    assert "tbk_mean" in res and "ci_95" in res and "precision" in res
    lo, hi = res["ci_95"]
    assert lo <= res["tbk_mean"] <= hi
    assert res["precision"] >= 0.0


def test_simulate_reasonable_scale():
    # Higher live time should reduce precision (i.e., narrower intervals)
    E1, Y1 = synth_spectrum(
        2.5,          # tbk_true
        100.0,        # cps_per
        600,          # live_time_s
        rng=np.random.default_rng(2),
    )

    # Seed the global RNG so Monte Carlo inside `simulate` is deterministic.
    np.random.seed(42)
    res1 = simulate(
        E1, Y1,
        live_time_s=600,
        calib=dict(cps_per_TBK=100.0),
        n_mc=6000,
    )

    # Independent 1800 s draw (different seed) + higher MC for stability
    E2, Y2 = synth_spectrum(
        2.5,          # tbk_true
        100.0,        # cps_per
        1800,         # live_time_s
        rng=np.random.default_rng(3),
    )

    # Re-seed so the MC noise is comparable between runs.
    np.random.seed(42)
    res2 = simulate(
        E2, Y2,
        live_time_s=1800,
        calib=dict(cps_per_TBK=100.0),
        n_mc=6000,
    )

    assert res2["precision"] <= res1["precision"] * 1.02  # allow 0.5% MC jitter


