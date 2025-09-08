
# Literature notes (alignment to repo)

This document summarizes how the repo aligns with three core papers and lists TODOs for deeper integration.

## 1) Kuriyan et al., 2020 — Thin but Fat phenotype & neonatal WBKC use

- Uses WBKC (K-40 1461 keV) to measure TBK and derive BCM in neonates; describes calibration with KCl phantoms and Monte Carlo geometry correction. Highlights: instrument precision around TBK, neonatal counting error, and ROI around 1460 keV.

**Repo alignment**
- `src/wbkc/model.py`: ROI defaults center near 1460 keV; Poisson + attenuation Monte Carlo for uncertainty.
- `src/wbkc/signal.py`: simple detrend + Savitzky–Golay smoothing + peak/ROI helpers.
- `app/streamlit_app.py`: upload spectrum, estimate TBK with 95% CI.

**TODOs**
- Geometry correction factor as a function of body size for static-geometry counters.
- Sideband background model refinement (e.g., linear or polynomial continuum under the K-40 peak).

## 2) Naqvi et al., 2018 — WBKC construction & validation in adults

- Shadow-shield WBKC design with four NaI(Tl) detectors and moving bed; calibration across BOMAB-like phantoms; Monte Carlo efficiency vs body geometry. Provides accuracy (~few %) and precision (~2%) with phantoms and gives formulae and workflow for peak area extraction and CPS-to-TBK mapping.

**Repo alignment**
- `src/wbkc/model.py`: current scaffold assumes a provided cps_per_TBK calibration constant; replace with a physics/geometry-aware calibration chain.
- `scripts/gen_data.py`: synthetic spectra generator for K-40 peak + background; can be extended to simulate detector efficiency vs size.

**TODOs**
- Implement geometry-aware efficiency model (`src/wbkc/calibration.py` stub added).
- Support inputs for detector response and BOMAB-equivalent size features (height, weight) to correct cps_per_TBK.

## 3) Kuriyan et al., 2019 — Protein requirements in pregnancy via TBK

- Uses trimester-wise TBK accretion to estimate protein accretion/EAR; supports the independence of TBK from hydration changes during pregnancy.

**Repo alignment**
- `scripts/gen_data.py`: can be extended to create trimester-linked synthetic spectra per subject to emulate TBK accretion series.
- Potential future module to compute BCM and protein accretion from longitudinal TBK (not implemented).

**General next steps**
- Replace simple ROI integration with Gaussian peak fit + linear background windowing.
- Expose calibration API: `counts_to_tbk(counts, live_time, geom_features, calib_table)`.
- Allow entering phantom-derived look-up tables or regression for CPS↔TBK.
