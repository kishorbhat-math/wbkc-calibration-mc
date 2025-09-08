"""
WBKC TBK estimation via Monte Carlo (Poisson + attenuation + calibration placeholders).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from . import signal

@dataclass
class Calib:
    cps_per_TBK: float = 100.0   # counts/s per unit TBK (arbitrary unit)
    bg_cps: float = 0.1          # background counts/s inside ROI (fallback, not used when fit supplies bg)
    attn_mean: float = 1.0       # multiplicative attenuation factor
    attn_rel_sigma: float = 0.05 # relative sigma for attenuation (~lognormal)

def simulate(
    energy_keV: np.ndarray | pd.Series,
    counts: np.ndarray | pd.Series,
    live_time_s: float,
    calib: Dict[str, float] | Calib | None = None,
    n_mc: int = 5000,
    roi_keV: Tuple[float, float] | None = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Estimate TBK with uncertainty from a gamma spectrum using a Gaussian+linear fit
    around the 1461 keV K-40 peak to get peak area (counts) and background under the peak.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(energy_keV, dtype=float)
    y = np.asarray(counts, dtype=float)

    # Light conditioning (optional)
    yc = signal.sg_smooth(signal.detrend(x, y, order=1), window_length=31, polyorder=3)

    if roi_keV is None:
        guess_keV = 1461.0
        sig_counts, bg_counts_fit, fit = signal.estimate_peak_area(x, yc, guess_keV=guess_keV, window_keV=150.0)
    else:
        # If manual ROI is given, fit within that window but keep the center near ROI mid
        lo, hi = roi_keV
        guess_keV = 0.5 * (lo + hi)
        mask = (x >= lo) & (x <= hi)
        sig_counts, bg_counts_fit, fit = signal.estimate_peak_area(x[mask], yc[mask], guess_keV=guess_keV, window_keV=(hi - lo))

    # Calibration defaults
    if calib is None:
        calib = Calib()
    elif isinstance(calib, dict):
        calib = Calib(**{**Calib().__dict__, **calib})

    # Use fitted background under the peak; fall back to bg_cps * t if fit is degenerate
    net_counts = max(sig_counts, 0.0)
    fitted_bg = float(bg_counts_fit)
    live_bg = fitted_bg if fitted_bg > 0 else calib.bg_cps * live_time_s

    # Poisson MC draws for signal and background (counts)
    lam_sig = max(net_counts, 0.0)
    lam_bg  = max(live_bg,   0.0)

    # Attenuation as lognormal
    rel = max(calib.attn_rel_sigma, 1e-6)
    var = (rel * calib.attn_mean) ** 2
    sigma2 = np.log(1 + var / (calib.attn_mean ** 2))
    mu = np.log(calib.attn_mean) - 0.5 * sigma2

    sig_draws = rng.poisson(lam=lam_sig, size=n_mc)
    bg_draws  = rng.poisson(lam=lam_bg,  size=n_mc)
    attn_draws = rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=n_mc)

    cps_net = np.maximum(sig_draws - bg_draws, 0) / max(live_time_s, 1e-9)
    tbk_samples = cps_net / np.maximum(calib.cps_per_TBK * attn_draws, 1e-12)

    tbk_mean = float(np.mean(tbk_samples))
    lo, hi = np.percentile(tbk_samples, [2.5, 97.5])
    ci_95 = (float(lo), float(hi))
    precision = float((hi - lo) / 2 / max(abs(tbk_mean), 1e-12))

    return {
        "tbk_mean": tbk_mean,
        "ci_95": ci_95,
        "precision": precision,
        "samples": tbk_samples,
        "meta": {
            "fit": {
                "mu_keV": float(fit.mu),
                "sigma_keV": float(fit.sigma),
                "A": float(fit.A),
                "bg_slope": float(fit.m),
                "bg_intercept": float(fit.b),
            },
            "sig_counts": float(sig_counts),
            "bg_counts_fit": float(fitted_bg),
            "live_time_s": float(live_time_s),
            "calib": calib.__dict__,
        },
    }
