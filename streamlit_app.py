import numpy as np
import plotly.graph_objects as go
import streamlit as st

from scripts.gen_data import synth_spectrum
from wbkc.model import simulate

st.set_page_config(page_title="WBKC Demo", layout="centered")
st.title("WBKC Monte Carlo — Demo")

tbk = st.slider("True TBK", 1.0, 4.0, 2.5, 0.1)
cps = st.slider("Counts per TBK (cps_per_TBK)", 50.0, 200.0, 100.0, 5.0)
live = st.select_slider("Live time (s)", options=[300, 600, 900, 1800], value=900)
nmc = st.select_slider(
    "Monte Carlo draws", options=[1500, 3000, 6000, 10000], value=6000
)

rng = np.random.default_rng(7)
E, Y = synth_spectrum(tbk, cps, live, rng=rng)
res = simulate(E, Y, live_time_s=live, calib={"cps_per_TBK": cps}, n_mc=nmc)

st.subheader("Results")
st.write(res)

fig = go.Figure()
fig.add_scatter(x=E, y=Y, mode="lines", name="Counts")
fig.update_layout(
    title="Synthetic Spectrum", xaxis_title="Energy (keV)", yaxis_title="Counts"
)
st.plotly_chart(fig, use_container_width=True)
