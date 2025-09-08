"""
Generate synthetic spectra and subject metadata for WBKC Monte Carlo experiments.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def synth_spectrum(true_tbk=2.5, cps_per_TBK=100.0, live_time_s=900, n=3001, keV_max=3000.0, seed=None):
    rng = np.random.default_rng(seed)
    energy = np.linspace(0, keV_max, n)
    bg_cps_per_bin = 0.00005
    bg = rng.poisson(lam=bg_cps_per_bin * live_time_s, size=n).astype(float)
    mu_keV = 1460.0
    sigma_keV = 15.0
    amp_cps_total = cps_per_TBK * true_tbk
    gauss = np.exp(-0.5 * ((energy - mu_keV) / sigma_keV) ** 2)
    gauss /= gauss.sum()
    peak_counts = rng.poisson(lam=(amp_cps_total * live_time_s) * gauss)
    counts = bg + peak_counts
    return pd.DataFrame({"energy_keV": energy, "counts": counts})

def main(outdir="data", n_subjects=5, seed=123):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    subjects = []
    for i in range(n_subjects):
        subj = {
            "id": f"S{i+1:02d}",
            "true_tbk": float(rng.uniform(1.5, 4.0)),
            "live_time_s": int(rng.integers(600, 1500)),
            "cps_per_TBK": float(rng.uniform(80.0, 120.0)),
        }
        df = synth_spectrum(true_tbk=subj["true_tbk"], cps_per_TBK=subj["cps_per_TBK"], live_time_s=subj["live_time_s"], seed=seed+i)
        csv_path = os.path.join(outdir, f"spectrum_{subj['id']}.csv")
        df.to_csv(csv_path, index=False)
        subjects.append({**subj, "spectrum_csv": csv_path})
    with open(os.path.join(outdir, "subjects.json"), "w") as f:
        json.dump(subjects, f, indent=2)
    print(f"Wrote {n_subjects} spectra & subjects to '{outdir}'")

if __name__ == "__main__":
    main()
