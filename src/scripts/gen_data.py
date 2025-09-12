"""
Synthetic data helpers for WBKC.

- synth_spectrum(): generate a single synthetic spectrum (E, counts) for a given TBK.
- make_phantoms(): generate a synthetic phantom table (TBK_true, cps_measured, weight_kg, height_cm).
- make_pregnancy(): generate synthetic TBK trajectories across 3 trimesters.

All outputs are synthetic; no real subject or phantom data are included.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SynthParams:
    mu_keV: float = 1461.0      # K-40 peak center
    sigma_keV: float = 15.0     # peak sigma
    bg_cps_per_bin: float = 5e-5  # background cps per energy bin

def synth_spectrum(
    tbk: float,
    cps_per_TBK: float,
    live_time_s: float,
    E_min: float = 0.0,
    E_max: float = 3000.0,
    dE: float = 1.0,
    params: Optional[SynthParams] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (energy_keV, counts) for a synthetic spectrum."""
    if rng is None:
        rng = np.random.default_rng()
    if params is None:
        params = SynthParams()

    E = np.arange(E_min, E_max + dE, dE, dtype=float)
    # background counts
    lam_bg_bin = params.bg_cps_per_bin * live_time_s
    bg = rng.poisson(lam_bg_bin, size=E.size).astype(float)

    # Gaussian peak normalized to 1.0 area, scale by signal counts
    mu, sig = params.mu_keV, params.sigma_keV
    gauss = np.exp(-0.5 * ((E - mu) / sig) ** 2)
    gauss /= gauss.sum()

    signal_counts = tbk * cps_per_TBK * live_time_s
    sig = rng.poisson(signal_counts * gauss)

    y = bg + sig
    return E, y.astype(float)

def make_phantoms(
    n: int = 12,
    true_cps_per_tbk: float = 100.0,
    a: float = 0.30,
    b: float = 0.70,
    noise_rel: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic phantom table for calibration fitting (no spectra)."""
    rng = np.random.default_rng(seed)
    weight = rng.uniform(50, 95, size=n)       # kg
    height = rng.uniform(155, 185, size=n)     # cm
    tbk_true = rng.uniform(2.0, 4.0, size=n)   # arbitrary TBK units
    ratio = weight / height
    eff = 1.0 / (a * ratio + b)
    cps_eff = true_cps_per_tbk * eff
    cps_meas = tbk_true * cps_eff * rng.normal(1.0, noise_rel, size=n)
    return pd.DataFrame({
        "TBK_true": tbk_true,
        "cps_measured": cps_meas,
        "weight_kg": weight,
        "height_cm": height
    })

def make_pregnancy(
    n_subjects: int = 6,
    baseline_tbk: float = 2.2,
    trimester_delta: float = 0.18,
    noise_rel: float = 0.03,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Synthetic pregnancy TBK trajectories for 3 trimesters:
      TBK_T1 = baseline_tbk + eps
      TBK_T2 = TBK_T1 + trimester_delta + eps
      TBK_T3 = TBK_T2 + trimester_delta + eps
    """
    rng = np.random.default_rng(seed)
    subj = np.arange(1, n_subjects + 1)
    t1 = baseline_tbk * rng.normal(1.0, noise_rel, size=n_subjects)
    t2 = (t1 + trimester_delta) * rng.normal(1.0, noise_rel, size=n_subjects)
    t3 = (t2 + trimester_delta) * rng.normal(1.0, noise_rel, size=n_subjects)
    df = pd.DataFrame({
        "subject": np.repeat(subj, 3),
        "trimester": np.tile([1, 2, 3], n_subjects),
        "TBK_true": np.concatenate([t1, t2, t3]),
    })
    return df

if __name__ == "__main__":
    # Optional CLI demo (writes CSVs under docs/examples); commented by default for confidentiality
    # import pathlib
    # outdir = pathlib.Path("docs/examples"); outdir.mkdir(parents=True, exist_ok=True)
    # make_phantoms().to_csv(outdir/"synthetic_phantoms.csv", index=False)
    # make_pregnancy().to_csv(outdir/"synthetic_pregnancy.csv", index=False)
    print("Synthetic helpers ready. Use from tests or demos; no files written by default.")
