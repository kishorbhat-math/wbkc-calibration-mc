"""
WBKC TBK estimation via Monte Carlo with configurable peak/background + geometry-aware calibration.
Adds uncertainty components and a variance breakdown.
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
    # Base calibration & measurement model
    cps_per_TBK: float = 100.0
    bg_cps: float = 0.1
    attn_mean: float = 1.0
    attn_rel_sigma: float = 0.05

    # Geometry efficiency model params (CPS2TBKCalib)
    geom_rel_sigma: float = 0.03
    a: float = 0.30
    b: float = 0.70

    # New: calibration parameter uncertainty (optional; set 0 for none)
    cps_rel_sigma: float = 0.00   # relative sigma on cps_per_TBK (lognormal)
    a_sigma: float = 0.00         # absolute sigma on a (normal)
    b_sigma: float = 0.00         # absolute sigma on b (normal)

def _ensure_geom(geom: GeometryFeatures | Dict | None) -> GeometryFeatures | None:
    if geom is None:
        return None
    if isinstance(geom, GeometryFeatures):
        return geom
    if isinstance(geom, dict):
        return GeometryFeatures.from_dict(geom)
    return None

def _lognormal_draws(rng, mean: float, rel_sigma: float, size: int) -> np.ndarray:
    rel = max(rel_sigma, 1e-12)
    var = (rel * mean) ** 2
    sigma2 = np.log(1.0 + var / (mean ** 2))
    mu = np.log(mean) - 0.5 * sigma2
    return rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=size)

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

    Uncertainty components sampled:
      - Counting statistics (Poisson signal/background)
      - Attenuation factor (lognormal around attn_mean)
      - Geometry efficiency correction (lognormal around efficiency mean)
      - Calibration parameters (optional): cps_per_TBK (lognormal), a/b (normal)

    Returns:
      dict with tbk_mean, ci_95, precision, samples, and meta. meta.uncertainty holds a variance breakdown.
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

    geom_feat = _ensure_geom(geom)

    # Expected counts (Poisson rate parameters)
    net_counts = max(sig_counts, 0.0)
    fitted_bg = float(bg_counts)
    live_bg = fitted_bg if fitted_bg > 0 else calib.bg_cps * live_time_s
    lam_sig = max(net_counts, 0.0)
    lam_bg  = max(live_bg,   0.0)

    # Draws for individual components
    # Counting
    sig_draws = rng.poisson(lam=lam_sig, size=n_mc)
    bg_draws  = rng.poisson(lam=lam_bg,  size=n_mc)

    # Attenuation
    attn_draws = _lognormal_draws(rng, mean=calib.attn_mean, rel_sigma=calib.attn_rel_sigma, size=n_mc)

    # Geometry efficiency correction
    cps2tbk = CPS2TBKCalib(cps_per_TBK=calib.cps_per_TBK, a=calib.a, b=calib.b, geom_rel_sigma=calib.geom_rel_sigma)
    eff_draws = cps2tbk.draw_efficiency_correction(rng, geom_feat, size=n_mc)

    # Calibration parameter uncertainty (optional)
    # cps_per_TBK as lognormal; a and b as normal
    if calib.cps_rel_sigma > 0:
        cps_draws = _lognormal_draws(rng, mean=calib.cps_per_TBK, rel_sigma=calib.cps_rel_sigma, size=n_mc)
    else:
        cps_draws = np.full(n_mc, calib.cps_per_TBK, dtype=float)

    a_draws = rng.normal(loc=calib.a, scale=max(calib.a_sigma, 0.0), size=n_mc) if calib.a_sigma > 0 else np.full(n_mc, calib.a, float)
    b_draws = rng.normal(loc=calib.b, scale=max(calib.b_sigma, 0.0), size=n_mc) if calib.b_sigma > 0 else np.full(n_mc, calib.b, float)

    # If a/b vary, recompute efficiency mean deterministically per draw for provided geometry
    if calib.a_sigma > 0 or calib.b_sigma > 0:
        # Rebuild an efficiency factor per draw using the mean, then multiply by a lognormal geom noise
        if geom_feat is None or geom_feat.weight_kg is None or geom_feat.height_cm is None:
            eff_mean = np.ones(n_mc, float)
        else:
            ratio = float(geom_feat.weight_kg) / max(float(geom_feat.height_cm), 1e-6)
            denom = np.maximum(a_draws * ratio + b_draws, 1e-3)
            eff_mean = 1.0 / denom
        # Multiply by lognormal geom noise around 1.0 with rel sigma geom_rel_sigma
        geom_noise = _lognormal_draws(rng, mean=1.0, rel_sigma=calib.geom_rel_sigma, size=n_mc) if calib.geom_rel_sigma > 0 else np.ones(n_mc, float)
        eff_draws = eff_mean * geom_noise

    # Assemble TBK samples
    cps_net = np.maximum(sig_draws - bg_draws, 0) / max(live_time_s, 1e-9)
    denom_all = np.maximum(cps_draws * attn_draws * eff_draws, 1e-12)
    tbk_samples = cps_net / denom_all

    # Summary stats
    tbk_mean = float(np.mean(tbk_samples))
    lo, hi = np.percentile(tbk_samples, [2.5, 97.5])
    ci_95 = (float(lo), float(hi))
    precision = float((hi - lo) / 2 / max(abs(tbk_mean), 1e-12))

    # --- Variance breakdown (simple clamp method with common random numbers) ---
    # Compute total variance
    var_total = float(np.var(tbk_samples, ddof=1)) if n_mc > 1 else 0.0

    # Helper to recompute with a component clamped to its mean (others vary)
    def _var_with_clamp(component: str) -> float:
        if component == "counting":
            # replace sig/bg with their means
            cps_net_c = np.maximum(lam_sig - lam_bg, 0) / max(live_time_s, 1e-9)
            return float(np.var(cps_net_c / denom_all, ddof=1))
        if component == "attenuation":
            denom = np.maximum(cps_draws * np.full(n_mc, calib.attn_mean) * eff_draws, 1e-12)
            return float(np.var(cps_net / denom, ddof=1))
        if component == "geometry":
            # clamp geometry to mean efficiency for the current a,b (or their means)
            if calib.a_sigma > 0 or calib.b_sigma > 0:
                # Use mean of per-draw eff_mean
                if geom_feat is None or geom_feat.weight_kg is None or geom_feat.height_cm is None:
                    eff_mean_scalar = 1.0
                else:
                    ratio = float(geom_feat.weight_kg) / max(float(geom_feat.height_cm), 1e-6)
                    eff_mean_scalar = float(np.mean(1.0 / np.maximum(a_draws * ratio + b_draws, 1e-3)))
            else:
                if geom_feat is None or geom_feat.weight_kg is None or geom_feat.height_cm is None:
                    eff_mean_scalar = 1.0
                else:
                    ratio = float(geom_feat.weight_kg) / max(float(geom_feat.height_cm), 1e-6)
                    eff_mean_scalar = 1.0 / max(calib.a * ratio + calib.b, 1e-3)
            denom = np.maximum(cps_draws * attn_draws * eff_mean_scalar, 1e-12)
            return float(np.var(cps_net / denom, ddof=1))
        if component == "calibration":
            # clamp cps_per_TBK, a, b to their means; keep geom noise as is but recompute eff mean from mean a,b
            cps_c = float(calib.cps_per_TBK)
            if geom_feat is None or geom_feat.weight_kg is None or geom_feat.height_cm is None:
                eff_mean_scalar = 1.0
            else:
                ratio = float(geom_feat.weight_kg) / max(float(geom_feat.height_cm), 1e-6)
                eff_mean_scalar = 1.0 / max(calib.a * ratio + calib.b, 1e-3)
            # If there is geom noise, keep it; but we want the effect of calibration only, so set cps_draws->mean and a/b->mean while keeping geom lognoise neutralized:
            # simplest: recompute denom with cps_c and with eff_draws replaced by (eff_mean_scalar * geom_noise_mean) where geom_noise_mean ~ 1.0
            denom = np.maximum(cps_c * attn_draws * eff_mean_scalar, 1e-12)
            return float(np.var(cps_net / denom, ddof=1))
        return var_total

    clamp_components = ["counting", "attenuation", "geometry", "calibration"]
    var_reduced = {k: max(var_total - _var_with_clamp(k), 0.0) for k in clamp_components}
    sum_reduced = sum(var_reduced.values()) or 1.0
    frac = {k: (v / sum_reduced) for k, v in var_reduced.items()}

    uncertainty = {
        "var_total": var_total,
        "components": {
            "counting": {"var_drop_if_clamped": var_reduced["counting"], "fraction_of_explained": frac["counting"]},
            "attenuation": {"var_drop_if_clamped": var_reduced["attenuation"], "fraction_of_explained": frac["attenuation"]},
            "geometry": {"var_drop_if_clamped": var_reduced["geometry"], "fraction_of_explained": frac["geometry"]},
            "calibration": {"var_drop_if_clamped": var_reduced["calibration"], "fraction_of_explained": frac["calibration"]},
        },
    }

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
                "a": calib.a, "b": calib.b,
                "cps_rel_sigma": calib.cps_rel_sigma,
                "a_sigma": calib.a_sigma,
                "b_sigma": calib.b_sigma,
            },
            "geom": None if geom_feat is None else {"weight_kg": geom_feat.weight_kg, "height_cm": geom_feat.height_cm},
            "uncertainty": uncertainty,
        },
    }
