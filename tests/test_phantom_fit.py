import numpy as np
import pandas as pd

from wbkc.calib_fit import fit_params

# NOTE: This function generates purely synthetic phantom data for testing only.
 
def _make_synth_phantoms(n=6, seed=0, true_cps_per_tbk=100.0, a=0.30, b=0.70, noise_rel=0.02):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(50, 95, size=n)       # kg
    height = rng.uniform(155, 185, size=n)     # cm
    tbk_true = rng.uniform(2.0, 4.0, size=n)   # arbitrary TBK units
    ratio = weight / height
    eff = 1.0 / (a * ratio + b)
    cps_eff = true_cps_per_tbk * eff
    cps_meas = tbk_true * cps_eff
    # add small measurement noise
    cps_meas = cps_meas * rng.normal(1.0, noise_rel, size=n)
    df = pd.DataFrame({
        "TBK_true": tbk_true,
        "cps_measured": cps_meas,
        "weight_kg": weight,
        "height_cm": height
    })
    return df

def test_fit_params_recovers_truth_within_tolerance():
    true = dict(cps_per_TBK=100.0, a=0.30, b=0.70)
    df = _make_synth_phantoms(n=12, seed=42, true_cps_per_tbk=true["cps_per_TBK"], a=true["a"], b=true["b"], noise_rel=0.01)
    est = fit_params(df, init=(90.0, 0.2, 0.8))
    # tolerances
    assert abs(est["cps_per_TBK"] - true["cps_per_TBK"]) / true["cps_per_TBK"] < 0.05
    assert abs(est["a"] - true["a"]) < 0.1
    assert abs(est["b"] - true["b"]) < 0.1

