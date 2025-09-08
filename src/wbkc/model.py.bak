"""
WBKC TBK estimation via Monte Carlo with configurable peak/background method.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from . import signal

@dataclass
class Calib:
    cps_per_TBK: float = 100.0
    bg_cps: float = 0.1
    attn_mean: float = 1.0
    attn_rel_sigma: float = 0.05

def simulate(
    energy_keV: np.ndarray | pd.Series,
    counts: np.ndarray | pd.Series,
    live_time_s: float,
    calib: Dict[str, float] | Calib | None = None,
    n_mc: int = 5000,
    roi_keV: Tuple[float, float] | None = None,
    peak_method: str = "gauss_linear",
    sideband_frac: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Estimate TBK with uncertainty from a gamma spectrum.
    peak_method:
      - "gauss_linear" (default): Gaussian peak + linear background fit
      - "sidebands_linearbg": integrate ROI and subtract background estimated from sidebands (linear across ROI)
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(energy_keV, dtype=float)
    y = np.asarray(counts, dtype=float)

    # Light conditioning
    yc = signal.sg_smooth(signal.detrend(x, y, order=1), window_length=31, polyorder=3)

    # ROI/method dispatch
    info = {}
    if roi_keV is None:
        sig_counts, bg_counts, info = signal.estimate_peak_area(
            x, yc, method=peak_method, guess_keV=1461.0, window_keV=150.0,
            roi_slice=None, sideband_frac=sideband_frac, width_keV_for_auto=100.0
        )
    else:
        lo, hi = roi_keV
        mask = (x >= lo) & (x <= hi)
        roi_slice = slice(int(np.argmax(mask)), int(np.argmax(mask[::-1]) and len(x) - np.argmax(mask[::-1])))
        # Simpler: pass explicit slice via indices
        lo_idx = int(np.searchsorted(x, lo, side="left"))
        hi_idx = int(np.searchsorted(x, hi, side="right"))
        roi_slice = slice(lo_idx, hi_idx)
        sig_counts, bg_counts, info = signal.estimate_peak_area(
            x, yc, method=peak_method, guess_keV=0.5*(lo+hi), window_keV=(hi-lo),
            roi_slice=roi_slice, sideband_frac=sideband_frac, width_keV_for_auto=(hi-lo)
        )

    # Calibration defaults
    if calib is None:
        calib = Calib()
    elif isinstance(calib, dict):
        calib = Calib(**{**Calib().__dict__, **calib})

    net_counts = max(sig_counts, 0.0)
    fitted_bg = float(bg_counts)
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
            "method": peak_method,
            "method_info": info,
            "sig_counts": float(sig_counts),
            "bg_counts": float(fitted_bg),
            "live_time_s": float(live_time_s),
            "calib": calib.__dict__,
        },
    }
