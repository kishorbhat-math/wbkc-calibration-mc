"""
Signal processing helpers for WBKC spectra:
- detrend / sg_smooth
- detect_peak / find_roi
- fit_peak_linear_bg / estimate_peak_area
- sideband-based background estimator (linear across ROI)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

def detrend(energy_keV: np.ndarray, counts: np.ndarray, order: int = 1) -> np.ndarray:
    x = np.asarray(energy_keV, float)
    y = np.asarray(counts, float)
    X = np.vander(x - x.mean(), N=order + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    trend = X @ coef
    return y - trend

def sg_smooth(counts: np.ndarray, window_length: int = 31, polyorder: int = 3) -> np.ndarray:
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    wl = max(wl, polyorder + 2 + (polyorder + 2) % 2)
    wl = min(wl, max(5, len(counts) - (1 - len(counts) % 2)))
    if wl < 5 or wl > len(counts):
        return np.asarray(counts, float)
    return savgol_filter(np.asarray(counts, float), window_length=wl, polyorder=polyorder, mode="interp")

def detect_peak(energy_keV: np.ndarray, counts: np.ndarray, guess_keV: float = 1460.0, window_keV: float = 150.0) -> int:
    x = np.asarray(energy_keV, float)
    y = np.asarray(counts, float)
    idx = int(np.argmin(np.abs(x - guess_keV)))
    half_width = int(np.argmin(np.abs(x - (guess_keV + window_keV))) - idx)
    lo = max(0, idx - abs(half_width))
    hi = min(len(y), idx + abs(half_width) + 1)
    if hi <= lo:
        return idx
    local = y[lo:hi]
    return lo + int(np.argmax(local))

def find_roi(energy_keV: np.ndarray, peak_idx: int, width_keV: float = 100.0) -> slice:
    x = np.asarray(energy_keV, float)
    center_keV = x[int(peak_idx)]
    lo_keV = center_keV - width_keV / 2
    hi_keV = center_keV + width_keV / 2
    lo_idx = int(np.searchsorted(x, lo_keV, side="left"))
    hi_idx = int(np.searchsorted(x, hi_keV, side="right"))
    return slice(max(0, lo_idx), min(len(x), hi_idx))

# --- Gaussian + linear background model ---
def _gauss_linear(x, A, mu, sigma, m, b):
    sigma = np.maximum(sigma, 1e-6)
    return m * x + b + A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

@dataclass
class PeakFit:
    A: float
    mu: float
    sigma: float
    m: float
    b: float
    area_counts: float        # area of Gaussian: A * sigma * sqrt(2*pi)
    bg_area_counts: float     # linear background under +/- 3*sigma
    cov: Optional[np.ndarray]

def fit_peak_linear_bg(energy_keV: np.ndarray, counts: np.ndarray, center_guess: float = 1461.0, window_keV: float = 150.0) -> PeakFit:
    x = np.asarray(energy_keV, float)
    y = np.asarray(counts, float)

    # Window around guess
    idx = np.argmin(np.abs(x - center_guess))
    half = np.argmin(np.abs(x - (center_guess + window_keV))) - idx
    lo = max(0, idx - abs(half))
    hi = min(len(x), idx + abs(half) + 1)
    xw = x[lo:hi]
    # mild conditioning for stability
    yw = sg_smooth(detrend(xw, y[lo:hi], order=0), window_length=21, polyorder=3)

    # Initial guesses
    mu0 = xw[np.argmax(yw)]
    sigma0 = max(12.0, min(30.0, window_keV / 10.0))
    A0 = max(1.0, (yw.max() - np.median(yw)))
    m0 = 0.0
    b0 = float(np.median(yw))
    p0 = [A0, mu0, sigma0, m0, b0]
    bounds = ([0.0, mu0 - 30.0, 5.0, -np.inf, -np.inf],
              [np.inf, mu0 + 30.0, 60.0,  np.inf,  np.inf])
    try:
        popt, pcov = curve_fit(_gauss_linear, xw, yw, p0, bounds=bounds, maxfev=20000)
    except Exception:
        popt, pcov = p0, None

    A, mu, sigma, m, b = map(float, popt)
    area = float(A * sigma * np.sqrt(2.0 * np.pi))
    lo_int = mu - 3.0 * sigma
    hi_int = mu + 3.0 * sigma
    bg_area = float(m * 0.5 * (hi_int**2 - lo_int**2) + b * (hi_int - lo_int))
    return PeakFit(A=A, mu=mu, sigma=sigma, m=m, b=b, area_counts=area, bg_area_counts=bg_area, cov=pcov)

# --- Sideband linear background estimator across ROI ---
def _sideband_linear_bg(energy_keV: np.ndarray, counts: np.ndarray, roi: slice, sideband_frac: float = 0.2) -> Tuple[float, float]:
    """
    Estimate background under ROI by fitting a straight line using means of left/right sidebands.
    sideband_frac: fraction of ROI width taken for each sideband (0 < f < 0.45).
    Returns (bg_counts_est, bg_per_bin_at_center) for diagnostics.
    """
    x = np.asarray(energy_keV, float)
    y = np.asarray(counts, float)
    lo, hi = roi.start, roi.stop
    n = max(hi - lo, 1)
    sb = max(1, int(n * sideband_frac))
    # Left and right sidebands
    L = slice(max(0, lo - sb), lo)
    R = slice(hi, min(len(x), hi + sb))
    if (L.stop - L.start) < 1 or (R.stop - R.start) < 1:
        # Fallback: flat background from edges of ROI
        yL = y[lo:lo+max(1, sb)]
        yR = y[hi-max(1, sb):hi]
        bg_per_bin = float((yL.mean() + yR.mean())/2.0)
        return float(bg_per_bin * n), float(bg_per_bin)
    # Linear fit through (xL_mean, yL_mean) and (xR_mean, yR_mean)
    xL, yL = x[L].mean(), y[L].mean()
    xR, yR = x[R].mean(), y[R].mean()
    if xR == xL:
        m = 0.0
        b = float((yL + yR)/2.0)
    else:
        m = (yR - yL) / (xR - xL)
        b = yL - m * xL
    x_lo = x[lo]
    x_hi = x[hi-1]
    bg_counts = float( m * 0.5 * (x_hi**2 - x_lo**2) + b * (x_hi - x_lo) )
    # Per-bin at center (just for meta)
    xC = 0.5 * (x_lo + x_hi)
    bg_center = float(m * xC + b)
    return max(bg_counts, 0.0), bg_center

def estimate_peak_area(
    energy_keV: np.ndarray,
    counts: np.ndarray,
    method: str = "gauss_linear",
    guess_keV: float = 1461.0,
    window_keV: float = 150.0,
    roi_slice: Optional[slice] = None,
    sideband_frac: float = 0.2,
    width_keV_for_auto: float = 100.0
) -> Tuple[float, float, dict]:
    """
    Returns (signal_counts, background_counts, info) around K-40 peak.
    method:
      - "gauss_linear": Gaussian peak + linear background fit (default)
      - "sidebands_linearbg": Integrate ROI and subtract linear background from sidebands
    """
    x = np.asarray(energy_keV, float)
    y = np.asarray(counts, float)

    if method == "gauss_linear":
        fit = fit_peak_linear_bg(x, y, center_guess=guess_keV, window_keV=window_keV)
        sig = max(fit.area_counts, 0.0)
        bg = max(fit.bg_area_counts, 0.0)
        return sig, bg, {
            "method": method, "mu": fit.mu, "sigma": fit.sigma, "A": fit.A, "bg_slope": fit.m, "bg_intercept": fit.b
        }

    # sidebands_linearbg
    if roi_slice is None:
        pidx = detect_peak(x, y, guess_keV, window_keV)
        roi_slice = find_roi(x, pidx, width_keV=width_keV_for_auto)
    lo, hi = roi_slice.start, roi_slice.stop
    roi_counts = np.asarray(y[roi_slice], float)
    sig_raw = float(roi_counts.sum())
    bg_est, _bg_center = _sideband_linear_bg(x, y, roi_slice, sideband_frac=sideband_frac)
    sig = max(sig_raw - bg_est, 0.0)
    return sig, max(bg_est, 0.0), {
        "method": method, "roi_lo_idx": int(lo), "roi_hi_idx": int(hi), "sideband_frac": float(sideband_frac)
    }
