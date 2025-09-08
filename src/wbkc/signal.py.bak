"""
Signal processing helpers for WBKC spectra:
- detrend
- sg_smooth
- detect_peak
- find_roi
- fit_peak_linear_bg / estimate_peak_area: Gaussian peak + linear background
"""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Tuple, Optional

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

def _gauss_linear(x, A, mu, sigma, m, b):
    return m * x + b + A * np.exp(-0.5 * ((x - mu) / np.maximum(sigma, 1e-6)) ** 2)

@dataclass
class PeakFit:
    A: float         # Gaussian amplitude (counts/bin)
    mu: float        # center (keV)
    sigma: float     # sigma (keV)
    m: float         # bg slope (counts/bin/keV)
    b: float         # bg intercept (counts/bin)
    area_counts: float         # integrated counts in peak = A * sigma * sqrt(2pi)
    bg_area_counts: float      # integrated bg under +/- 3*sigma window
    cov: Optional[np.ndarray]  # covariance matrix from curve_fit

def fit_peak_linear_bg(energy_keV: np.ndarray, counts: np.ndarray, center_guess: float = 1460.0, window_keV: float = 150.0) -> PeakFit:
    x = np.asarray(energy_keV, float)
    y = np.asarray(counts, float)

    # Window around guess
    idx = np.argmin(np.abs(x - center_guess))
    half = np.argmin(np.abs(x - (center_guess + window_keV))) - idx
    lo = max(0, idx - abs(half))
    hi = min(len(x), idx + abs(half) + 1)
    xw = x[lo:hi]
    yw = sg_smooth(detrend(xw, y[lo:hi], order=0), window_length=21, polyorder=3)

    # Initial guesses
    mu0 = xw[np.argmax(yw)]
    sigma0 = max(12.0, min(30.0, window_keV / 10.0))
    A0 = max(1.0, (yw.max() - np.median(yw)) )
    m0 = 0.0
    b0 = float(np.median(yw))

    p0 = [A0, mu0, sigma0, m0, b0]
    bounds = (
        [0.0, mu0 - 30.0, 5.0, -np.inf, -np.inf],
        [np.inf, mu0 + 30.0, 60.0,  np.inf,  np.inf],
    )

    try:
        popt, pcov = curve_fit(_gauss_linear, xw, yw, p0, bounds=bounds, maxfev=20000)
    except Exception:
        popt = p0
        pcov = None

    A, mu, sigma, m, b = map(float, popt)
    # Peak area = A * sigma * sqrt(2*pi)
    area = float(A * sigma * np.sqrt(2.0 * np.pi))

    # Background integral under +/- 3 sigma
    lo_int = mu - 3.0 * sigma
    hi_int = mu + 3.0 * sigma
    bg_area = float( m * 0.5 * (hi_int**2 - lo_int**2) + b * (hi_int - lo_int) )

    return PeakFit(A=A, mu=mu, sigma=sigma, m=m, b=b, area_counts=area, bg_area_counts=bg_area, cov=pcov)

def estimate_peak_area(energy_keV: np.ndarray, counts: np.ndarray, guess_keV: float = 1461.0, window_keV: float = 150.0) -> Tuple[float, float, PeakFit]:
    """Returns (signal_counts, background_counts, fit) around K-40 peak."""
    fit = fit_peak_linear_bg(energy_keV, counts, center_guess=guess_keV, window_keV=window_keV)
    sig = max(fit.area_counts, 0.0)
    bg = max(fit.bg_area_counts, 0.0)
    return sig, bg, fit
