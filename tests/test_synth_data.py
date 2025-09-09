import numpy as np
from wbkc.model import simulate
from scripts.gen_data import synth_spectrum, make_pregnancy, make_phantoms

def test_precision_decreases_with_live_time():
    tbk_true = 2.6; cps_per = 100.0
    E1, y1 = synth_spectrum(tbk_true, cps_per_TBK=cps_per, live_time_s=600, seed := None)
    E2, y2 = synth_spectrum(tbk_true, cps_per_TBK=cps_per, live_time_s=1200, seed := None)
    res1 = simulate(E1, y1, live_time_s=600,  calib=dict(cps_per_TBK=cps_per), n_mc=2000)
    res2 = simulate(E2, y2, live_time_s=1200, calib=dict(cps_per_TBK=cps_per), n_mc=2000)
    assert res2["precision"] < res1["precision"] * 0.8  # expect ~1/sqrt(t) improvement

def test_pregnancy_tbk_increases_across_trimesters():
    import pandas as pd
    traj = make_pregnancy(n_subjects=1, baseline_tbk=2.2, trimester_delta=0.25, noise_rel=0.01, seed=99)
    # Simulate spectra at a fixed live time and calibration
    cps_per = 100.0; t = 900
    tbk_series = traj.sort_values(["subject", "trimester"])["TBK_true"].to_list()
    res = []
    for tbk in tbk_series:
        E, y = synth_spectrum(tbk, cps_per_TBK=cps_per, live_time_s=t)
        res.append(simulate(E, y, live_time_s=t, calib=dict(cps_per_TBK=cps_per), n_mc=2000)["tbk_mean"])
    assert res[0] < res[1] < res[2]

def test_phantom_fit_sanity_recovers_within_bounds():
    # Coarse sanity: fitted cps_per_TBK should be within ~10% on noisier data
    from wbkc.calib_fit import fit_params
    df = make_phantoms(n=10, true_cps_per_tbk=100.0, a=0.30, b=0.70, noise_rel=0.03, seed=7)
    est = fit_params(df, init=(100.0, 0.25, 0.75))
    assert abs(est["cps_per_TBK"] - 100.0) / 100.0 < 0.10
