"""
WBKC TBK estimation via Monte Carlo with configurable peak/background + geometry-aware calibration.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from . import signal
from .calibration import GeometryFeatures, CPS2TBKCalib

@dataclass
class Calib:
    cps_per_TBK: float = 100.0
    bg_cps: float = 0.1
    attn_mean: float = 1.0
    attn_rel_sigma: float = 0.05
    # Geometry handling (used to parameterize CPS2TBKCalib if user doesn't pass one):
    geom_rel_sigma: float = 0.03
    a: float = 0.30
    b: float = 0.70

def _ensure_geom(geom: GeometryFeatures | Dict | None) -> GeometryFeatures | None:
    if geom is None:
        return None
    if isinstance(geom, GeometryFeatures):
        return geom
    if isinstance(geom, dict):
        return GeometryFeatures.from_dict(geom)
    return None

def simulate(
    energy_keV: np.ndarray | pd.Series,
    counts: np.ndarray | pd.Series,
    live_time_s: float,
    calib: Dict[str, float] | Calib | None = None,
    n_mc: int = 5000,
    roi_keV: Tuple[float, float] | None = None,
    peak_method: str = "gauss_linear",
    sideband_frac: float = 0.2,
    geom: GeometryFeatures | Dict | None = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Estimate TBK with uncertainty from a gamma spectrum.
    Adds geometry-aware efficiency correction with uncertainty:
      cps_per_TBK_effective = cps_per_TBK * attenuation_factor * efficiency_correction
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(energy_keV, dtype=float)
    y = np.asarray(counts, dtype=float)

    # Light conditioning
    yc = signal.sg_smooth(signal.detrend(x, y, order=1), window_length=31, polyorder=3)

    # ROI/method dispatch
    if roi_keV is None:
        sig_counts, bg_counts, info = signal.estimate_peak_area(
            x, yc, method=peak_method, guess_keV=1461.0, window_keV=150.0,
            roi_slice=None, sideband_frac=sideband_frac, width_keV_for_auto=100.0
        )
    else:
        lo, hi = roi_keV
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

    # Poisson parameters for counts
    net_counts = max(sig_counts, 0.0)
    fitted_bg = float(bg_counts)
    live_bg = fitted_bg if fitted_bg > 0 else calib.bg_cps * live_time_s
    lam_sig = max(net_counts, 0.0)
    lam_bg  = max(live_bg,   0.0)

    # Attenuation as lognormal
    rel_attn = max(calib.attn_rel_sigma, 1e-6)
    var_attn = (rel_attn * calib.attn_mean) ** 2
    sigma2_attn = np.log(1 + var_attn / (calib.attn_mean ** 2))
    mu_attn = np.log(calib.attn_mean) - 0.5 * sigma2_attn
    attn_draws = rng.lognormal(mean=mu_attn, sigma=np.sqrt(sigma2_attn), size=n_mc)

    # Geometry efficiency correction (mean+uncertainty)
    geom_feat = _ensure_geom(geom)
    cps2tbk = CPS2TBKCalib(cps_per_TBK=calib.cps_per_TBK, a=calib.a, b=calib.b, geom_rel_sigma=calib.geom_rel_sigma)
    eff_draws = cps2tbk.draw_efficiency_correction(rng, geom_feat, size=n_mc)

    # Monte Carlo draws for counts
    sig_draws = rng.poisson(lam=lam_sig, size=n_mc)
    bg_draws  = rng.poisson(lam=lam_bg,  size=n_mc)

    cps_net = np.maximum(sig_draws - bg_draws, 0) / max(live_time_s, 1e-9)
    # Effective denominator includes attenuation *and* geometry efficiency correction
    denom = np.maximum(cps2tbk.cps_per_TBK * attn_draws * eff_draws, 1e-12)
    tbk_samples = cps_net / denom

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
            "calib": {
                "cps_per_TBK": calib.cps_per_TBK,
                "attn_mean": calib.attn_mean,
                "attn_rel_sigma": calib.attn_rel_sigma,
                "geom_rel_sigma": calib.geom_rel_sigma,
                "a": calib.a, "b": calib.b
            },
            "geom": None if geom_feat is None else {"weight_kg": geom_feat.weight_kg, "height_cm": geom_feat.height_cm},
        },
    }
