import io
import numpy as np
import pandas as pd
import streamlit as st

from wbkc.model import simulate
from wbkc.signal import detect_peak, find_roi, sg_smooth, detrend

st.set_page_config(page_title="WBKC TBK Monte Carlo", layout="wide")
st.title("WBKC TBK Estimator (Monte Carlo)")

st.sidebar.header("Inputs")

uploaded = st.sidebar.file_uploader("Spectrum CSV (energy_keV,counts)", type=["csv"])
live_time_s = st.sidebar.number_input("Live time (s)", min_value=10, max_value=24*3600, value=900, step=10)

st.sidebar.subheader("Calibration")
cps_per_TBK = st.sidebar.number_input("cps_per_TBK", min_value=1e-6, value=100.0, step=1.0, format="%.6f")
bg_cps = st.sidebar.number_input("Background cps in ROI", min_value=0.0, value=0.1, step=0.05)
attn_mean = st.sidebar.number_input("Attenuation mean", min_value=0.01, value=1.0, step=0.01)
attn_rel_sigma = st.sidebar.number_input("Attenuation relative sigma", min_value=0.0, value=0.05, step=0.01, format="%.4f")

n_mc = st.sidebar.number_input("MC samples", min_value=500, max_value=100000, value=5000, step=500)

st.sidebar.subheader("ROI")
use_auto_roi = st.sidebar.checkbox("Auto ROI around 1460 keV", value=True)
roi_lo = st.sidebar.number_input("ROI low (keV)", value=1410.0, step=5.0)
roi_hi = st.sidebar.number_input("ROI high (keV)", value=1510.0, step=5.0)

def gen_synth(n=3001, keV_max=3000.0, true_tbk=2.5, cps_per_TBK=100.0, live_time_s=900, seed=42):
    rng = np.random.default_rng(seed)
    energy = np.linspace(0, keV_max, n)
    # background ~ 0.1 cps/bin within ROI width ~100 bins -> ~10 cps total -> *live_time
    bg_cps_per_bin = 0.00005
    bg = rng.poisson(lam=bg_cps_per_bin * live_time_s, size=n).astype(float)
    # Gaussian peak near 1460 keV
    mu_keV = 1460.0
    sigma_keV = 15.0
    amp_cps_total = cps_per_TBK * true_tbk  # cps integrated over peak area (approx)
    # approximate discrete Gaussian area normalization
    gauss = np.exp(-0.5 * ((energy - mu_keV) / sigma_keV) ** 2)
    gauss /= gauss.sum()
    peak_counts = rng.poisson(lam=(amp_cps_total * live_time_s) * gauss)
    counts = bg + peak_counts
    df = pd.DataFrame({"energy_keV": energy, "counts": counts})
    return df

if uploaded is None:
    st.info("No file uploaded — generating a synthetic spectrum.")
    df = gen_synth(true_tbk=2.5, cps_per_TBK=cps_per_TBK, live_time_s=live_time_s)
else:
    df = pd.read_csv(uploaded)

energy = df["energy_keV"].to_numpy()
counts = df["counts"].to_numpy()

# Optional conditioning preview
y = detrend(energy, counts, order=1)
y = sg_smooth(y, window_length=31, polyorder=3)

if use_auto_roi:
    pidx = detect_peak(energy, y, guess_keV=1460.0, window_keV=150.0)
    roi_slice = find_roi(energy, pidx, width_keV=100.0)
    roi = (energy[roi_slice.start], energy[roi_slice.stop-1])
else:
    roi = (roi_lo, roi_hi)

res = simulate(
    energy, counts, live_time_s,
    calib=dict(cps_per_TBK=cps_per_TBK, bg_cps=bg_cps, attn_mean=attn_mean, attn_rel_sigma=attn_rel_sigma),
    n_mc=int(n_mc),
    roi_keV=None if use_auto_roi else roi
)

c1, c2, c3 = st.columns(3)
c1.metric("TBK (mean, a.u.)", f"{res['tbk_mean']:.3f}")
c2.metric("95% CI low", f"{res['ci_95'][0]:.3f}")
c3.metric("95% CI high", f"{res['ci_95'][1]:.3f}")
st.caption(f"Relative precision (95% half-width / mean): {res['precision']:.2%}")

# Precision vs time (t^{-1/2} scaling assumption using current estimate as baseline)
t_grid = np.linspace(60, 3600, 20)
# Approximate: std ~ k / sqrt(t); fit k from current MC
samples = np.array(res["samples"])
std_now = samples.std()
t_now = res["meta"]["live_time_s"]
k = std_now * np.sqrt(t_now)
pred_std = k / np.sqrt(t_grid)
pred_hw95 = 1.96 * pred_std  # ~95% half width for normal approx
with st.expander("Precision vs. time"):
    chart_df = pd.DataFrame({"time_s": t_grid, "rel_precision": pred_hw95 / max(res["tbk_mean"],1e-12)})
    st.line_chart(chart_df.set_index("time_s"))

with st.expander("Download TBK samples (CSV)"):
    csv = pd.DataFrame({"tbk_samples": samples}).to_csv(index=False).encode()
    st.download_button("Download", data=csv, file_name="tbk_samples.csv", mime="text/csv")
