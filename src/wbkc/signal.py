"""
Signal processing stubs for WBKC spectra.

These are intentionally light-weight and easily replaceable with your lab's pipeline.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter

def detrend(energy_keV: np.ndarray, counts: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Remove a low-order polynomial trend from counts.
    """
    x = energy_keV
    y = counts.astype(float)
    X = np.vander(x - x.mean(), N=order+1, increasing=True)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    trend = X @ coef
    return y - trend

def sg_smooth(counts: np.ndarray, window_length: int = 31, polyorder: int = 3) -> np.ndarray:
    """
    Savitzky–Golay smoothing. Ensures odd window length and >= polyorder+2.
    """
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    wl = max(wl, polyorder + 2 + (polyorder + 2) % 2)  # odd & big enough
    wl = min(wl, len(counts) - (1 - len(counts) % 2))  # not longer than data
    if wl < 5:
        return counts
    return savgol_filter(counts, window_length=wl, polyorder=polyorder, mode="interp")

def detect_peak(energy_keV: np.ndarray, counts: np.ndarray, guess_keV: float = 1460.0, window_keV: float = 150.0) -> int:
    """
    Crude K-40 peak detection near ~1460 keV.
    """
    idx = np.argmin(np.abs(energy_keV - guess_keV))
    half = np.argmin(np.abs(energy_keV - (guess_keV + window_keV))) - idx
    lo = max(0, idx - abs(half))
    hi = min(len(counts), idx + abs(half) + 1)
    local = counts[lo:hi]
    if local.size == 0:
        return idx
    local_idx = np.argmax(local)
    return lo + local_idx

def find_roi(energy_keV: np.ndarray, peak_idx: int, width_keV: float = 100.0) -> slice:
    """
    ROI around detected peak with +/- width_keV/2 window.
    """
    center_keV = energy_keV[peak_idx]
    lo_keV = center_keV - width_keV / 2
    hi_keV = center_keV + width_keV / 2
    lo_idx = np.searchsorted(energy_keV, lo_keV, side="left")
    hi_idx = np.searchsorted(energy_keV, hi_keV, side="right")
    return slice(max(0, lo_idx), min(len(energy_keV), hi_idx))
