import numpy as np
import plotly.graph_objects as go

from scripts.gen_data import synth_spectrum
from wbkc.model import simulate

# make a synthetic spectrum and run the MC simulate
E, Y = synth_spectrum(2.5, 100.0, 900, rng=np.random.default_rng(7))
res = simulate(E, Y, live_time_s=900, calib={"cps_per_TBK": 100.0}, n_mc=6000)

print("=== WBKC Monte Carlo demo ===")
for k in ("tbk_mean", "precision", "ci_95"):
    print(f"{k}: {res[k]}")

# quick plot and save to HTML
fig = go.Figure()
fig.add_scatter(x=E, y=Y, mode="lines", name="Counts")
fig.update_layout(
    title="Synthetic Spectrum (demo)", xaxis_title="Energy (keV)", yaxis_title="Counts"
)
fig.write_html("demo_output.html", include_plotlyjs="cdn")
print("\nWrote demo_output.html (open in browser).")
