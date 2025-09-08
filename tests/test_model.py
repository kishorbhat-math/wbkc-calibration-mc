import numpy as np
from wbkc.model import simulate

def make_synth(true_tbk=3.0, cps_per_TBK=100.0, live_time_s=900, n=3001):
    energy = np.linspace(0, 3000, n)
    rng = np.random.default_rng(0)
    bg_cps_per_bin = 0.00005
    bg = rng.poisson(lam=bg_cps_per_bin * live_time_s, size=n).astype(float)
    mu_keV = 1460.0
    sigma_keV = 15.0
    amp_cps_total = cps_per_TBK * true_tbk
    gauss = np.exp(-0.5 * ((energy - mu_keV) / sigma_keV) ** 2)
    gauss /= gauss.sum()
    peak_counts = rng.poisson(lam=(amp_cps_total * live_time_s) * gauss)
    counts = bg + peak_counts
    return energy, counts

def test_simulate_returns_ci():
    energy, counts = make_synth(true_tbk=2.0, cps_per_TBK=120.0, live_time_s=800)
    res = simulate(energy, counts, live_time_s=800,
                   calib=dict(cps_per_TBK=120.0, bg_cps=0.1, attn_mean=1.0, attn_rel_sigma=0.05),
                   n_mc=2000)
    assert "tbk_mean" in res and "ci_95" in res and "precision" in res
    lo, hi = res["ci_95"]
    assert lo <= res["tbk_mean"] <= hi
    assert res["precision"] >= 0.0

def test_simulate_reasonable_scale():
    # Higher live time should reduce precision (i.e., narrower intervals)
    energy, counts = make_synth(true_tbk=2.5, cps_per_TBK=100.0, live_time_s=600)
    res1 = simulate(energy, counts, live_time_s=600, calib=dict(cps_per_TBK=100.0), n_mc=1500)
    res2 = simulate(energy, counts, live_time_s=1800, calib=dict(cps_per_TBK=100.0), n_mc=1500)
    assert res2["precision"] < res1["precision"]
