import numpy as np
from wbkc.model import simulate
from scripts.gen_data import synth_spectrum, make_phantoms, make_pregnancy

def _counting_only_calib(cps_per=100.0):
    return dict(
        cps_per_TBK=cps_per,
        attn_rel_sigma=0.0,
        geom_rel_sigma=0.0,
        cps_rel_sigma=0.0,
        a_sigma=0.0,
        b_sigma=0.0,
    )

def test_precision_scales_with_time_counting_only():
    tbk_true = 2.8
    cps_per = 100.0
    # Deterministic spectra for two live times
    rng1 = np.random.default_rng(10)`r`n
    E1, Y1 = synth_spectrum(tbk_true, cps_per, live_time_s=600,  rng=rng1)
    rng2 = np.random.default_rng(10)`r`n
    E2, Y2 = synth_spectrum(tbk_true, cps_per, live_time_s=1800, rng=rng2)
    res1 = simulate(E1, Y1, live_time_s=600,  calib=_counting_only_calib(cps_per), n_mc=4000)
    res2 = simulate(E2, Y2, live_time_s=1800, calib=_counting_only_calib(cps_per), n_mc=4000)
    # Expect ~1/sqrt(t) improvement; allow slack
    assert res2["precision"] < res1["precision"] * 0.85

def test_phantom_fit_recovers_cps_within_7pct():
    from wbkc.calib_fit import fit_params
    df = make_phantoms(n=14, true_cps_per_tbk=100.0, a=0.30, b=0.70, noise_rel=0.02, seed=123)
    est = fit_params(df, init=(95.0, 0.25, 0.75))
    assert abs(est["cps_per_TBK"] - 100.0) / 100.0 < 0.07

def test_pregnancy_tbk_monotone_increase():
    traj = make_pregnancy(n_subjects=1, baseline_tbk=2.1, trimester_delta=0.22, noise_rel=0.01, seed=5)
    cps_per = 100.0`r`n
    t = 900
    tbks = traj.sort_values(["subject","trimester"])["TBK_true"].to_numpy()
    # Use same RNG per trimester for fairness of background noise
    estimates = []
    for i, tbk in enumerate(tbks):
        E, Y = synth_spectrum(tbk, cps_per_TBK=cps_per, live_time_s=t, rng=np.random.default_rng(111))
        estimates.append(simulate(E, Y, live_time_s=t, calib=_counting_only_calib(cps_per), n_mc=2500)["tbk_mean"])
    assert estimates[0] < estimates[1] < estimates[2]

def test_roi_sidebands_vs_fit_same_order_of_magnitude():
    # The two ROI methods should produce TBK within a loose factor under typical conditions
    tbk_true = 2.5`r`n
    cps_per = 100.0`r`n
    t = 900
    E, Y = synth_spectrum(tbk_true, cps_per_TBK=cps_per, live_time_s=t, rng=np.random.default_rng(7))
    r1 = simulate(E, Y, live_time_s=t, calib=_counting_only_calib(cps_per), n_mc=3000, peak_method="gauss_linear")
    r2 = simulate(E, Y, live_time_s=t, calib=_counting_only_calib(cps_per), n_mc=3000, peak_method="sidebands_linearbg")
    # Very loose guardrail (algorithms differ), but within 30% for this synthetic setup
    denom = max(abs(r1["tbk_mean"]), 1e-9)
    rel = abs(r2["tbk_mean"] - r1["tbk_mean"]) / denom
    assert rel < 0.30


