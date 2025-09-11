import numpy as np

from wbkc.model import simulate


def _synth(true_tbk=2.5, cps_per_TBK=100.0, live_time_s=900, n=3001, seed=0):
    E = np.linspace(0, 3000, n)
    rng = np.random.default_rng(seed)
    # background
    bg = rng.poisson(0.00005 * live_time_s, size=E.size).astype(float)
    # K-40 peak
    mu, sig = 1461.0, 15.0
    gauss = np.exp(-0.5*((E-mu)/sig)**2)


    gauss /= gauss.sum()
    y = bg + rng.poisson((true_tbk*cps_per_TBK*live_time_s)*gauss)
    return E, y

def test_methods_within_tolerance():
    E, y = _synth(true_tbk=2.5, cps_per_TBK=100.0, live_time_s=900, seed=1)
    res_gauss = simulate(E, y, live_time_s=900, calib=dict(cps_per_TBK=100.0), n_mc=2000, peak_method="gauss_linear")
    res_sideb = simulate(E, y, live_time_s=900, calib=dict(cps_per_TBK=100.0), n_mc=2000, peak_method="sidebands_linearbg", sideband_frac=0.2)
    # Methods should broadly agree within ~10%
    denom = max(1e-9, abs(res_gauss["tbk_mean"]))
    assert abs(res_gauss["tbk_mean"] - res_sideb["tbk_mean"]) / denom < 0.10

def test_manual_roi_runs():
    E, y = _synth(true_tbk=3.0, cps_per_TBK=120.0, live_time_s=800, seed=2)
    res_auto = simulate(E, y, live_time_s=800, calib=dict(cps_per_TBK=120.0), n_mc=1500, peak_method="gauss_linear")
    res_roi  = simulate(E, y, live_time_s=800, calib=dict(cps_per_TBK=120.0), n_mc=1500, peak_method="sidebands_linearbg", roi_keV=(1410, 1510))
    assert res_auto["precision"] >= 0.0 and res_roi["precision"] >= 0.0





