from __future__ import annotations

import numpy as np


def synth_spectrum(tbk_true: float, cps_per: float, live_time_s: int, rng=None):
    """
    Make a toy gamma spectrum with a Gaussian line at 1461 keV.
    Returns (E, Y) where:
      - E is energy array [0..3000] keV (step 1 keV)
      - Y is counts (float) with Poisson fluctuations for background and signal
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Energy grid
    E = np.arange(0, 3001, dtype=float)

    # Peak shape (K-40 line, for example)
    mu, sig = 1461.0, 15.0
    gauss = np.exp(-0.5 * ((E - mu) / sig) ** 2)
    gauss /= gauss.sum()

    # Very light, flat background (cps ~ 5e-5/keV)
    bg_rate_per_keV = 5e-5  # counts per second per keV (toy)
    bg_counts = rng.poisson(bg_rate_per_keV * live_time_s, size=E.size).astype(float)

    # Signal scale: tbk_true * cps_per is cps in the peak; distribute by gauss
    lam = (tbk_true * cps_per * live_time_s) * gauss
    sig_counts = rng.poisson(lam)

    Y = bg_counts + sig_counts
    return E, Y


# Back-compat convenience used by some tests earlier
def make_synth(true_tbk: float, cps_per_TBK: float, live_time_s: int, rng=None):
    return synth_spectrum(true_tbk, cps_per_TBK, live_time_s, rng=rng)
