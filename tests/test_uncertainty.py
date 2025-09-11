import numpy as np
from wbkc.model import simulate

def _synth(true_tbk=2.5, cps_per_TBK=100.0, live_time_s=900, n=3001, seed=0):
    E = np.linspace(0, 3000, n)
    rng = np.random.default_rng(seed)
    bg = rng.poisson(0.00005 * live_time_s, size=E.size).astype(float)
    mu, sig = 1461.0, 15.0
    gauss = np.exp(-0.5*((E-mu)/sig)**2)`r`n
    gauss /= gauss.sum()
    y = bg + rng.poisson((true_tbk*cps_per_TBK*live_time_s)*gauss)
    return E, y

def test_uncertainty_breakdown_keys_present():
    E, y = _synth(seed=1)
    res = simulate(E, y, live_time_s=900, calib=dict(cps_per_TBK=100.0), n_mc=2000)
    u = res["meta"]["uncertainty"]
    assert "var_total" in u and "components" in u
    for k in ["counting", "attenuation", "geometry", "calibration"]:
        assert k in u["components"]

def test_geometry_uncertainty_increases_geom_fraction():
    E, y = _synth(seed=2)
    base = simulate(E, y, live_time_s=900, calib=dict(cps_per_TBK=100.0, geom_rel_sigma=0.00), n_mc=3000, geom=dict(weight_kg=80.0,height_cm=170.0))
    more = simulate(E, y, live_time_s=900, calib=dict(cps_per_TBK=100.0, geom_rel_sigma=0.08), n_mc=3000, geom=dict(weight_kg=80.0,height_cm=170.0))
    f_base = base["meta"]["uncertainty"]["components"]["geometry"]["fraction_of_explained"]
    f_more = more["meta"]["uncertainty"]["components"]["geometry"]["fraction_of_explained"]
    assert f_more > f_base

def test_calibration_uncertainty_affects_fraction():
    E, y = _synth(seed=3)
    base = simulate(E, y, live_time_s=900, calib=dict(cps_per_TBK=100.0, cps_rel_sigma=0.00), n_mc=3000)
    more = simulate(E, y, live_time_s=900, calib=dict(cps_per_TBK=100.0, cps_rel_sigma=0.05), n_mc=3000)
    f_base = base["meta"]["uncertainty"]["components"]["calibration"]["fraction_of_explained"]
    f_more = more["meta"]["uncertainty"]["components"]["calibration"]["fraction_of_explained"]
    assert f_more > f_base



