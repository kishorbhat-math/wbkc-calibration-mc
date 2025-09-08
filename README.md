# wbkc-calibration-mc

Monte Carlo calibration scaffold for Whole-Body K Counting (WBKC) — **Python 3.11**.

This repository provides:
- `src/wbkc/model.py` exposing `simulate()` to estimate **TBK** (Total Body Potassium) with uncertainty.
- Signal pipeline **stubs** in `src/wbkc/signal.py`: `detrend`, `sg_smooth`, `detect_peak`, `find_roi`.
- **Poisson + attenuation** Monte Carlo uncertainty.
- **Streamlit app** (`app/streamlit_app.py`): upload spectrum CSV, view TBK + 95% CI and precision vs time.
- `scripts/gen_data.py`: generate **synthetic spectra and subjects**.
- `tests`: minimal tests for model behavior.
- `environment.yml`, `pyproject.toml`, and CI workflow.
- MIT License.

> ⚠️ This is a **starter/scaffold** intended for extension. Calibration physics and numerics are simplified.

## Install

Using conda/mamba:

```bash
mamba env create -f environment.yml
mamba activate wbkc-calibration-mc
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
pip install -r <(python -c "import tomllib,sys;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['dev']))")
```

## Run tests

```bash
pytest -q
```

## Run Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Upload a CSV with columns: `energy_keV,counts`.
If no file is provided, the app can generate a synthetic spectrum.

## API sketch

```python
from wbkc.model import simulate

result = simulate(
    energy_keV, counts, live_time_s=900,
    calib=dict(cps_per_TBK=100.0,  # counts/s per unit TBK (arbitrary unit)
               bg_cps=0.2,
               attn_mean=1.0, attn_rel_sigma=0.05),
    n_mc=5000,
    roi_keV=(1410, 1510)
)
print(result["tbk_mean"], result["ci_95"])
```

## CSV format

Required headers:
- `energy_keV`: bin center energies (keV)
- `counts`: integer counts per bin for the acquisition

## Notes

- **Precision vs time** plot assumes Poisson-dominated statistics and scales by \(t^{-1/2}\).
- Replace `cps_per_TBK` with a lab-calibrated constant or full physics-based model as needed.


## References
- Kuriyan R. et al., 2020. *The Thin But Fat Phenotype is Uncommon at Birth in Indian Babies* (WBKC + neonatal body composition).
- Naqvi S. et al., 2018. *The development of a whole-body potassium counter for the measurement of body cell mass in adult humans* (WBKC construction, Monte Carlo geometry).
- Kuriyan R. et al., 2019. *Estimation of protein requirements in Indian pregnant women using a whole-body potassium counter* (TBK accretion in pregnancy).
