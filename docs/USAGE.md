# Usage

## Installation (dev)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
import numpy as np
from wbkc.model import simulate

# Toy example spectrum
rng = np.random.default_rng(7)
E = np.linspace(0, 3000, 3001)
mu, sig = 1461.0, 15.0
gauss = np.exp(-0.5 * ((E - mu)/sig)**2)
gauss /= gauss.sum()
bg = rng.poisson(0.00005 * 900, size=E.size).astype(float)
Y = bg + rng.poisson((2.5 * 100.0 * 900) * gauss)

res = simulate(E, Y, live_time_s=900, calib={"cps_per_TBK": 100.0}, n_mc=2000)
print(res)
.\scripts\run_tests.ps1
.\scripts\run_app.ps1
