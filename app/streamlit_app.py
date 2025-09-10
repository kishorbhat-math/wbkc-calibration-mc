import io, sys, pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Ensure repo root is importable (so "scripts" & "src" are visible)
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wbkc.model import simulate
from wbkc.calib_fit import fit_params
from scripts.gen_data import make_pregnancy, synth_spectrum, make_phantoms

st.set_page_config(page_title="WBKC Calibration MC", layout="wide")
st.title("WBKC Calibration MC")
st.caption("Upload data or generate synthetic examples; estimate TBK with Monte Carlo uncertainty.")

def _sanitize_state():
    for key in ["single_demo_data", "long_df", "phantom_df"]:
        val = st.session_state.get(key, None)
        if isinstance(val, bool):
            st.session_state.pop(key, None)
_sanitize_state()

# --- Sidebar: shared controls ---
st.sidebar.header("Monte Carlo & Calibration")
live_time_s = st.sidebar.number_input("Live time (s)", min_value=1, value=900, step=30)
n_mc        = st.sidebar.slider("Monte Carlo samples", min_value=1000, max_value=20000, value=5000, step=1000)

st.sidebar.subheader("Calibration (central values)")
cps_per_TBK = st.sidebar.number_input("cps_per_TBK", min_value=1.0, value=100.0, step=1.0)
attn_mean   = st.sidebar.number_input("attenuation mean", min_value=0.1, value=1.0, step=0.1)
a           = st.sidebar.number_input("geom a", value=0.30, step=0.01, format="%.2f")
b           = st.sidebar.number_input("geom b", value=0.70, step=0.01, format="%.2f")

st.sidebar.subheader("Uncertainty (1σ)")
attn_rel_sigma = st.sidebar.slider("attenuation rel σ", 0.0, 0.2, 0.05, 0.01)
geom_rel_sigma = st.sidebar.slider("geometry rel σ", 0.0, 0.2, 0.04, 0.01)
cps_rel_sigma  = st.sidebar.slider("cps_per_TBK rel σ", 0.0, 0.2, 0.03, 0.01)
a_sigma        = st.sidebar.slider("a σ (abs)", 0.0, 0.5, 0.00, 0.01)
b_sigma        = st.sidebar.slider("b σ (abs)", 0.0, 0.5, 0.00, 0.01)

st.sidebar.subheader("Geometry (optional)")
w_kg = st.sidebar.number_input("weight_kg", min_value=0.0, value=80.0, step=1.0)
h_cm = st.sidebar.number_input("height_cm", min_value=0.0, value=170.0, step=1.0)
use_geom = st.sidebar.checkbox("Use geometry", value=True)

calib = dict(
    cps_per_TBK=float(cps_per_TBK),
    attn_mean=float(attn_mean),
    attn_rel_sigma=float(attn_rel_sigma),
    geom_rel_sigma=float(geom_rel_sigma),
    cps_rel_sigma=float(cps_rel_sigma),
    a=float(a), b=float(b),
    a_sigma=float(a_sigma), b_sigma=float(b_sigma),
)
geom = dict(weight_kg=float(w_kg), height_cm=float(h_cm)) if use_geom else None

# Tabs
tab_single, tab_long, tab_phant = st.tabs(["Single Spectrum", "Longitudinal Series", "Phantom Calibration"])

# ---------- Tab 1: Single Spectrum ----------
with tab_single:
    st.subheader("Single Spectrum")
    with st.expander("Input format", expanded=False):
        st.markdown("CSV columns: `energy_keV` (or `keV`) and `counts` (or `cps`).")

    peak_method = st.selectbox("Peak/Background Method", ["gauss_linear", "sidebands_linearbg"], index=0)
    st.sidebar.subheader("ROI (optional)")
    roi_on = st.sidebar.checkbox("Specify ROI manually", value=False)
    roi_lo = st.sidebar.number_input("ROI lo (keV)", value=1400.0, step=1.0)
    roi_hi = st.sidebar.number_input("ROI hi (keV)", value=1520.0, step=1.0)
    uploaded = st.file_uploader("Upload spectrum CSV", type=["csv"], key="single_csv")

    def _read_csv_guess(file: io.BytesIO) -> tuple[np.ndarray, np.ndarray]:
        raw = file.read()
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:
            df = None
            for sep in [";", "\t", "|", " "]:
                try: df = pd.read_csv(io.BytesIO(raw), sep=sep); break
                except Exception: pass
            if df is None: raise
        cols = {c.lower(): c for c in df.columns}
        e_key = next((cols[c] for c in ["energy_keV","energy","keV","e"] if c in cols), None)
        if e_key is None: raise ValueError("Missing energy column (energy_keV/keV).")
        y_key = next((cols[c] for c in ["counts","cps","count","y"] if c in cols), None)
        if y_key is None: raise ValueError("Missing counts column (counts/cps).")
        x = pd.to_numeric(df[e_key], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[y_key], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    left, right = st.columns([7, 5])
    if uploaded is None:
        st.info("Upload a spectrum CSV or generate a synthetic example.")
        if st.button("Generate synthetic example (in-memory)", key="btn_demo_single"):
            E = np.linspace(0, 3000, 3001)
            mu, sig = 1461.0, 15.0
            rng = np.random.default_rng(11)
            bg = rng.poisson(0.00005 * live_time_s, size=E.size).astype(float)
            gauss = np.exp(-0.5*((E-mu)/sig)**2); gauss /= gauss.sum()
            true_tbk = 2.5
            y = bg + rng.poisson((true_tbk * cps_per_TBK * live_time_s) * gauss)
            st.session_state["single_demo_data"] = (E, y)
    else:
        st.session_state.pop("single_demo_data", None)

    E = y = None
    if "single_demo_data" in st.session_state:
        data = st.session_state["single_demo_data"]
        if isinstance(data, (tuple, list)) and len(data) == 2:
            E, y = data

    if uploaded is not None:
        try:
            E, y = _read_csv_guess(uploaded)
        except Exception as ex:
            st.error(f"Could not parse CSV: {ex}")

    if E is not None and y is not None:
        roi = (float(roi_lo), float(roi_hi)) if roi_on else None
        with st.spinner("Running Monte Carlo..."):
            res = simulate(E, y, live_time_s=float(live_time_s),
                           calib=calib, n_mc=int(n_mc),
                           roi_keV=roi, peak_method=peak_method,
                           geom=geom)
        c1, c2, c3 = st.columns(3)
        c1.metric("TBK (mean)", f"{res['tbk_mean']:.3f}")
        c2.metric("95% CI", f"[{res['ci_95'][0]:.3f}, {res['ci_95'][1]:.3f}]")
        c3.metric("Precision", f"{res['precision']:.3f}")
        fig = px.line(x=E, y=y, labels={"x":"Energy (keV)", "y":"Counts"})
        fig.update_traces(mode="lines", line=dict(width=1))
        if roi is not None:
            fig.add_vrect(x0=roi[0], x1=roi[1], fillcolor="LightGreen", opacity=0.2, line_width=0)
        right.plotly_chart(fig, use_container_width=True)

# ---------- Tab 2: Longitudinal Series ----------
with tab_long:
    st.subheader("Longitudinal Series (synthetic spectra from TBK_true)")
    st.markdown("Upload CSV with `subject, trimester, TBK_true`, or generate a synthetic table.")
    up = st.file_uploader("Upload longitudinal CSV", type=["csv"], key="long_csv")
    colA, colB = st.columns([1,1])
    if colA.button("Generate synthetic table (n_subjects=6)", key="gen_long"):
        df_syn = make_pregnancy(n_subjects=6, baseline_tbk=2.2, trimester_delta=0.18, noise_rel=0.03, seed=123)
        st.session_state["long_df"] = df_syn

    if up is not None:
        try:
            df_long = pd.read_csv(up)
            need = {"subject","trimester","TBK_true"}
            if not need.issubset({c.strip() for c in df_long.columns}):
                raise ValueError(f"CSV must include columns: {sorted(need)}")
            st.session_state["long_df"] = df_long
        except Exception as ex:
            st.error(f"Could not parse longitudinal CSV: {ex}")

    if "long_df" in st.session_state:
        df_in = st.session_state["long_df"].copy()
        df_in["subject"] = df_in["subject"].astype(str)
        df_in["trimester"] = df_in["trimester"].astype(int)
        df_in = df_in.sort_values(["subject","trimester"])

        results = []
        for subj, df_subj in df_in.groupby("subject"):
            rng = np.random.default_rng(abs(hash(subj)) % (2**32))
            for _, row in df_subj.iterrows():
                tbk_true = float(row["TBK_true"])
                E, Y = synth_spectrum(tbk_true, cps_per_TBK=cps_per_TBK, live_time_s=live_time_s, rng=rng)
                out = simulate(E, Y, live_time_s=live_time_s, calib=calib, geom=geom, n_mc=int(n_mc))
                ci_lo, ci_hi = out["ci_95"]
                results.append({
                    "subject": subj,
                    "trimester": int(row["trimester"]),
                    "TBK_est": float(out["tbk_mean"]),
                    "CI_lo": float(ci_lo),
                    "CI_hi": float(ci_hi),
                    "precision": float(out["precision"]),
                })

        dfr = pd.DataFrame(results).sort_values(["subject","trimester"]).reset_index(drop=True)
        dfr["err"] = (dfr["CI_hi"] - dfr["CI_lo"]) / 2.0
        fig = px.line(dfr, x="trimester", y="TBK_est", color="subject", markers=True, error_y="err",
                      labels={"trimester":"Trimester","TBK_est":"Estimated TBK"})
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Estimates table**")
        st.dataframe(dfr[["subject","trimester","TBK_est","CI_lo","CI_hi","precision"]], use_container_width=True)

# ---------- Tab 3: Phantom Calibration ----------
with tab_phant:
    st.subheader("Phantom Calibration (synthetic-only)")
    st.markdown(
        "Upload `TBK_true, cps_measured, weight_kg, height_cm` or generate synthetic phantoms. "
        "Fit `(cps_per_TBK, a, b)` and visualize measured vs fitted cps and residuals."
    )
    up_ph = st.file_uploader("Upload phantom CSV", type=["csv"], key="phantom_csv")
    col1, col2 = st.columns([1,1])
    if col1.button("Generate synthetic phantoms (n=12)", key="btn_make_phantoms"):
        dfp = make_phantoms(n=12, true_cps_per_tbk=float(cps_per_TBK), a=float(a), b=float(b),
                            noise_rel=0.02, seed=123)
        st.session_state["phantom_df"] = dfp

    df_ph = None
    if up_ph is not None:
        try:
            df_ph = pd.read_csv(up_ph)
            need = {"TBK_true","cps_measured","weight_kg","height_cm"}
            if not need.issubset({c.strip() for c in df_ph.columns}):
                raise ValueError(f"CSV must include columns: {sorted(need)}")
            st.session_state["phantom_df"] = df_ph
        except Exception as ex:
            st.error(f"Could not parse phantom CSV: {ex}")

    if "phantom_df" in st.session_state:
        df_ph = st.session_state["phantom_df"].copy()
        st.write("**Phantom table (first 10 rows)**")
        st.dataframe(df_ph.head(10), use_container_width=True)

        with st.spinner("Fitting calibration parameters (cps_per_TBK, a, b)..."):
            est = fit_params(df_ph, init=(float(cps_per_TBK), float(a), float(b)))

        c1, c2, c3 = st.columns(3)
        c1.metric("cps_per_TBK", f"{est['cps_per_TBK']:.3f}")
        c2.metric("a (geom)", f"{est['a']:.4f}")
        c3.metric("b (geom)", f"{est['b']:.4f}")

        r = df_ph["weight_kg"].to_numpy(float) / np.maximum(df_ph["height_cm"].to_numpy(float), 1e-6)
        eff = 1.0 / np.maximum(est["a"] * r + est["b"], 1e-9)
        cps_fit = df_ph["TBK_true"].to_numpy(float) * est["cps_per_TBK"] * eff
        df_plot = pd.DataFrame({"cps_measured": df_ph["cps_measured"].to_numpy(float), "cps_fitted": cps_fit})

        fig_sc = px.scatter(df_plot, x="cps_measured", y="cps_fitted",
                            labels={"cps_measured":"Measured cps", "cps_fitted":"Fitted cps"},
                            title="Measured vs Fitted cps")
        xy_max = float(df_plot[["cps_measured","cps_fitted"]].to_numpy().max()) * 1.05
        xy = np.linspace(0, xy_max, 50)
        fig_sc.add_scatter(x=xy, y=xy, mode="lines", name="y = x", line=dict(width=1, dash="dash"))
        st.plotly_chart(fig_sc, use_container_width=True)

        resid = df_plot["cps_measured"] - df_plot["cps_fitted"]
        fig_hist = px.histogram(pd.DataFrame({"residual":resid}), x="residual", nbins=20, title="Residuals (measured - fitted)")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.write("**Residual summary**")
        st.json({"mean": float(resid.mean()), "std": float(resid.std(ddof=1) if len(resid)>1 else 0.0),
                 "max_abs": float(np.max(np.abs(resid)))})
    else:
        st.info("Upload or generate a phantom table to fit calibration.")
