import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from wbkc.model import simulate

st.set_page_config(page_title="WBKC: Single Spectrum", layout="wide")

st.title("WBKC Calibration MC — Single Spectrum")
st.caption("Upload a spectrum CSV and estimate TBK with Monte Carlo uncertainty.")

with st.expander("Input format", expanded=False):
    st.markdown(
        """
        **CSV columns (auto-detected):**
        - `energy_keV` or `Energy_keV` or `keV`
        - `counts` or `Counts` or `cps` (counts per bin are fine; live time is specified below)

        The file can include headers; delimiter will be auto-sniffed.
        """
    )

# --- Sidebar controls ---
st.sidebar.header("Analysis controls")
peak_method = st.sidebar.selectbox(
    "Peak/Background Method",
    options=["gauss_linear", "sidebands_linearbg"],
    index=0,
    help="Gaussian peak + linear background fit vs. sideband linear background subtraction"
)

live_time_s = st.sidebar.number_input("Live time (s)", min_value=1, value=900, step=30)
n_mc = st.sidebar.slider("Monte Carlo samples", min_value=1000, max_value=20000, value=5000, step=1000)

st.sidebar.subheader("Calibration (central values)")
cps_per_TBK = st.sidebar.number_input("cps_per_TBK", min_value=1.0, value=100.0, step=1.0)
attn_mean = st.sidebar.number_input("attenuation mean", min_value=0.1, value=1.0, step=0.1)
a = st.sidebar.number_input("geom a", value=0.30, step=0.01, format="%.2f")
b = st.sidebar.number_input("geom b", value=0.70, step=0.01, format="%.2f")

st.sidebar.subheader("Uncertainty (1σ)")
attn_rel_sigma = st.sidebar.slider("attenuation rel σ", 0.0, 0.2, 0.05, 0.01)
geom_rel_sigma = st.sidebar.slider("geometry rel σ", 0.0, 0.2, 0.04, 0.01)
cps_rel_sigma = st.sidebar.slider("cps_per_TBK rel σ", 0.0, 0.2, 0.03, 0.01)
a_sigma = st.sidebar.slider("a σ (abs)", 0.0, 0.5, 0.00, 0.01)
b_sigma = st.sidebar.slider("b σ (abs)", 0.0, 0.5, 0.00, 0.01)

st.sidebar.subheader("Geometry (optional)")
w_kg = st.sidebar.number_input("weight_kg", min_value=0.0, value=80.0, step=1.0)
h_cm = st.sidebar.number_input("height_cm", min_value=0.0, value=170.0, step=1.0)
use_geom = st.sidebar.checkbox("Use geometry", value=True)

st.sidebar.subheader("ROI (optional)")
roi_on = st.sidebar.checkbox("Specify ROI manually", value=False)
roi_lo = st.sidebar.number_input("ROI lo (keV)", value=1400.0, step=1.0)
roi_hi = st.sidebar.number_input("ROI hi (keV)", value=1520.0, step=1.0)

uploaded = st.file_uploader("Upload spectrum CSV", type=["csv"])

def _read_csv_guess(file: io.BytesIO) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    raw = file.read()
    try:
        # Try pandas default
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        # Attempt semicolon / tab sniffing
        for sep in [";", "\t", "|", " "]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep)
                break
            except Exception:
                df = None
        if df is None:
            raise
    cols = {c.lower(): c for c in df.columns}

    # energy
    e_key = None
    for cand in ["energy_keV", "energykev", "energy", "keV", "e"]:
        if cand in cols:
            e_key = cols[cand]; break
        # try case-insensitive variants
        matches = [c for c in df.columns if c.lower() == cand]
        if matches:
            e_key = matches[0]; break
    if e_key is None:
        raise ValueError("Could not find energy column (energy_keV/keV).")

    # counts
    y_key = None
    for cand in ["counts", "cps", "count", "y"]:
        if cand in cols:
            y_key = cols[cand]; break
        matches = [c for c in df.columns if c.lower() == cand]
        if matches:
            y_key = matches[0]; break
    if y_key is None:
        raise ValueError("Could not find counts column (counts/cps).")

    x = pd.to_numeric(df[e_key], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_key], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    return x, y, df

# Main layout
left, right = st.columns([7, 5])

if uploaded is None:
    st.info("Upload a spectrum CSV to begin. Or generate a quick synthetic example with the button below.")
    if st.button("Generate synthetic example (in-memory)"):
        # Minimal synthetic spectrum using the same model as tests
        E = np.linspace(0, 3000, 3001)
        mu, sig = 1461.0, 15.0
        rng = np.random.default_rng(11)
        bg = rng.poisson(0.00005 * live_time_s, size=E.size).astype(float)
        gauss = np.exp(-0.5*((E-mu)/sig)**2); gauss /= gauss.sum()
        true_tbk = 2.5
        y = bg + rng.poisson((true_tbk * cps_per_TBK * live_time_s) * gauss)
        st.session_state["demo"] = (E, y)
else:
    st.session_state.pop("demo", None)

E = y = None
if "demo" in st.session_state:
    E, y = st.session_state["demo"]

if uploaded is not None:
    try:
        E, y, df = _read_csv_guess(uploaded)
    except Exception as ex:
        st.error(f"Could not parse CSV: {ex}")

if E is not None and y is not None:
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
    roi = (float(roi_lo), float(roi_hi)) if roi_on else None

    with st.spinner("Running Monte Carlo..."):
        res = simulate(E, y, live_time_s=float(live_time_s),
                       calib=calib, n_mc=int(n_mc),
                       roi_keV=roi, peak_method=peak_method,
                       geom=geom)

    # --- Results cards ---
    c1, c2, c3 = st.columns(3)
    c1.metric("TBK (mean)", f"{res['tbk_mean']:.3f}")
    c2.metric("95% CI", f"[{res['ci_95'][0]:.3f}, {res['ci_95'][1]:.3f}]")
    c3.metric("Precision (half-CI / mean)", f"{res['precision']:.3f}")

    # --- Spectrum plot ---
    fig = px.line(x=E, y=y, labels={"x": "Energy (keV)", "y": "Counts"})
    fig.update_traces(mode="lines", line=dict(width=1))
    if roi is not None:
        fig.add_vrect(x0=roi[0], x1=roi[1], fillcolor="LightGreen", opacity=0.2, line_width=0)
    right.plotly_chart(fig, use_container_width=True)

    # --- Uncertainty breakdown ---
    u = res.get("meta", {}).get("uncertainty", {})
    comps = u.get("components", {})
    if comps:
        rows = []
        for name, d in comps.items():
            rows.append({
                "component": name,
                "fraction_of_explained": float(d.get("fraction_of_explained", 0.0)),
                "var_drop_if_clamped": float(d.get("var_drop_if_clamped", 0.0)),
            })
        dfu = pd.DataFrame(rows).sort_values("fraction_of_explained", ascending=False)
        left.subheader("Uncertainty breakdown")
        left.dataframe(dfu, use_container_width=True)
else:
    st.stop()
