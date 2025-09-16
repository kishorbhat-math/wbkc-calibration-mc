"""
WBKC TBK estimation via Monte Carlo (Poisson + attenuation).

This is a simplified pedagogical scaffold. Replace calibration and physics as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import signal


@dataclass
class Calib:
    cps_per_TBK: float = 100.0  # counts per second per unit TBK (arbitrary unit)
    bg_cps: float = 0.1  # background counts per second within ROI
    attn_mean: float = 1.0  # multiplicative attenuation factor
    attn_rel_sigma: float = 0.05  # relative sigma for attenuation (~lognormal)


def _integrate_roi(
    energy_keV: np.ndarray, counts: np.ndarray, roi: slice
) -> Tuple[float, float]:
    """
    Return (signal_counts, background_counts) in the ROI
    using crude sidebands (outer 20% of ROI) as background estimator.
    """
    roi_counts = counts[roi].astype(float)
    n = len(roi_counts)
    if n < 10:
        return float(roi_counts.sum()), 0.0

    side = max(1, n // 5)
    left_bg = roi_counts[:side].mean()
    right_bg = roi_counts[-side:].mean()
    bg_per_bin = (left_bg + right_bg) / 2.0
    bg_est = bg_per_bin * n
    sig = roi_counts.sum() - bg_est
    return float(max(sig, 0.0)), float(max(bg_est, 0.0))


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
    Estimate TBK with uncertainty from a gamma spectrum.
    Parameters
    ----------
    energy_keV, counts : arrays
        Spectrum arrays (same length). Energy in keV, integer counts per bin.
    live_time_s : float
        Acquisition live time (s).
    calib : dict or Calib
        Calibration parameters. Use Calib defaults or provide keys:
        - cps_per_TBK: counts/s per unit TBK
        - bg_cps: background counts/s inside ROI
        - attn_mean: multiplicative attenuation factor (mean)
        - attn_rel_sigma: relative sigma of attenuation
    n_mc : int
        Number of Monte Carlo samples.
    roi_keV : (lo, hi) or None
        If None, automatically detect K-40 peak and choose +/-50 keV ROI.
    Returns
    -------
    dict with keys: tbk_mean, ci_95 (low, high), precision, samples, meta
    """
    if rng is None:
        rng = np.random.default_rng()

    energy_keV = np.asarray(energy_keV, dtype=float)
    counts = np.asarray(counts, dtype=float)

    # Light signal conditioning
    y = signal.detrend(energy_keV, counts, order=1)
    y = signal.sg_smooth(y, window_length=31, polyorder=3)

    if roi_keV is None:
        peak_idx = signal.detect_peak(energy_keV, y, guess_keV=1460.0, window_keV=150.0)
        roi = signal.find_roi(energy_keV, peak_idx, width_keV=100.0)
    else:
        lo, hi = roi_keV
        lo_idx = np.searchsorted(energy_keV, lo, side="left")
        hi_idx = np.searchsorted(energy_keV, hi, side="right")
        roi = slice(max(0, lo_idx), min(len(energy_keV), hi_idx))

    sig_counts, bg_counts_est = _integrate_roi(energy_keV, y, roi)

    # Default calibration
    if calib is None:
        calib = Calib()
    elif isinstance(calib, dict):
        calib = Calib(**{**Calib().__dict__, **calib})  # overlay onto defaults

    # Expected counts in ROI (signal + background)
    # Translate expected TBK -> expected counts via cps_per_TBK * t * attn
    # Here we invert: observed counts -> TBK estimate.
    net_counts = max(sig_counts, 0.0)
    live_bg = calib.bg_cps * live_time_s
    # Poisson MC on signal and background separately
    # (Use max to protect against negatives)
    lam_sig = max(net_counts, 0.0)
    lam_bg = max(live_bg, 0.0)

    # Attenuation as lognormal with mean attn_mean and rel sigma
    # Map (mu, sigma) of lognormal so E[X]=attn_mean, std=attn_rel_sigma*attn_mean
    rel = max(calib.attn_rel_sigma, 1e-6)
    var = (rel * calib.attn_mean) ** 2
    # Solve for lognormal params
    sigma2 = np.log(1 + var / (calib.attn_mean**2))
    mu = np.log(calib.attn_mean) - 0.5 * sigma2

    sig_draws = rng.poisson(lam=lam_sig, size=n_mc)
    bg_draws = rng.poisson(lam=lam_bg, size=n_mc)
    attn_draws = rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=n_mc)

    # Net CPS and TBK per sample
    cps_net = np.maximum(sig_draws - bg_draws, 0) / max(live_time_s, 1e-9)
    tbk_samples = cps_net / np.maximum(calib.cps_per_TBK * attn_draws, 1e-12)

    tbk_mean = float(np.mean(tbk_samples))
    lo, hi = np.percentile(tbk_samples, [2.5, 97.5])
    ci_95 = (float(lo), float(hi))
    # Precision as half-width / mean (relative 95% half-interval)
    precision = float((hi - lo) / 2 / max(abs(tbk_mean), 1e-12))

    return {
        "tbk_mean": tbk_mean,
        "ci_95": ci_95,
        "precision": precision,
        "samples": tbk_samples,
        "meta": {
            "roi": (int(roi.start), int(roi.stop)),
            "sig_counts": float(sig_counts),
            "bg_counts_est": float(bg_counts_est),
            "live_time_s": float(live_time_s),
            "calib": calib.__dict__,
        },
    }
