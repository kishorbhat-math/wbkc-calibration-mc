import numpy as np
from wbkc.model import simulate

def make_synth(true_tbk=2.5, cps_per_TBK=100.0, live_time_s=600, seed=0):
    rng = np.random.default_rng(seed)
    E = np.linspace(0, 3000, 3001)
    mu, sig = 1461.0, 15.0
    gauss = np.exp(-0.5*((E-mu)/sig)**2)`r`n
    gauss /= gauss.sum()
    bg = rng.poisson(0.00005 * live_time_s, size=E.size).astype(float)
    y = bg + rng.poisson((true_tbk*cps_per_TBK*live_time_s)*gauss)
    return E, y

def test_simulate_reasonable_scale():
    # Clamp to counting-only uncertainty for clear √t scaling
    base_calib = dict(cps_per_TBK=100.0,
                      attn_rel_sigma=0.0,
                      geom_rel_sigma=0.0,
                      cps_rel_sigma=0.0,
                      a_sigma=0.0,
                      b_sigma=0.0)

    energy, counts = make_synth(true_tbk=2.5, cps_per_TBK=100.0, live_time_s=600, seed=1)
    res1 = simulate(energy, counts, live_time_s=600, calib=base_calib, n_mc=3000)
    energy, counts = make_synth(true_tbk=2.5, cps_per_TBK=100.0, live_time_s=1800, seed=2)
    res2 = simulate(energy, counts, live_time_s=1800, calib=base_calib, n_mc=3000)

    # Expect res2 precision < res1 precision (allow 15% slack for MC noise)
    assert res2["precision"] < res1["precision"] * 0.85


