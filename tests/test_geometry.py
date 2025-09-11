import numpy as np
from wbkc.model import simulate

def _synth(true_tbk=2.5, cps_per_TBK=100.0, live_time_s=900, n=3001, seed=0):
    E = np.linspace(0, 3000, n)
    rng = np.random.default_rng(seed)
    # background
    bg = rng.poisson(0.00005 * live_time_s, size=E.size).astype(float)
    # K-40 peak
    mu, sig = 1461.0, 15.0
    gauss = np.exp(-0.5*((E-mu)/sig)**2)`r`n
    gauss /= gauss.sum()
    y = bg + rng.poisson((true_tbk*cps_per_TBK*live_time_s)*gauss)
    return E, y

def test_heavier_phantom_yields_higher_tbk_for_same_counts():
    # Same spectrum, different geometry assumptions:
    # Heavier (same height) => lower efficiency => inferred TBK should be higher.
    E, y = _synth(true_tbk=2.6, cps_per_TBK=100.0, live_time_s=900, seed=2)
    light = dict(weight_kg=55.0, height_cm=170.0)
    heavy = dict(weight_kg=90.0, height_cm=170.0)
    res_light = simulate(E, y, live_time_s=900, calib=dict(cps_per_TBK=100.0, geom_rel_sigma=0.0), n_mc=2000, geom=light)
    res_heavy = simulate(E, y, live_time_s=900, calib=dict(cps_per_TBK=100.0, geom_rel_sigma=0.0), n_mc=2000, geom=heavy)
    assert res_heavy["tbk_mean"] > res_light["tbk_mean"]

def test_more_geom_uncertainty_widens_ci():
    E, y = _synth(true_tbk=2.6, cps_per_TBK=100.0, live_time_s=900, seed=3)
    geom = dict(weight_kg=70.0, height_cm=170.0)
    res_low = simulate(E, y, live_time_s=900, calib=dict(cps_per_TBK=100.0, geom_rel_sigma=0.0), n_mc=2000, geom=geom)
    res_high = simulate(E, y, live_time_s=900, calib=dict(cps_per_TBK=100.0, geom_rel_sigma=0.08), n_mc=2000, geom=geom)
    hw_low  = (res_low["ci_95"][1] - res_low["ci_95"][0]) / 2.0
    hw_high = (res_high["ci_95"][1] - res_high["ci_95"][0]) / 2.0
    assert hw_high > hw_low



