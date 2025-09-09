from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REQUIRED_COLS = ["TBK_true", "cps_measured", "weight_kg", "height_cm"]

def load_phantoms(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Phantom file not found: {p}")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in {".json", ".ndjson"}:
        df = pd.read_json(p, lines=(p.suffix.lower()==".ndjson"))
    else:
        raise ValueError("Provide CSV or JSON with columns: " + ", ".join(REQUIRED_COLS))
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in phantom file: {missing}")
    return df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)

def _ratio(weight_kg: np.ndarray, height_cm: np.ndarray) -> np.ndarray:
    return weight_kg / np.maximum(height_cm, 1e-6)

def fit_params(df: pd.DataFrame, init: Tuple[float, float, float] = (100.0, 0.30, 0.70)) -> Dict[str, float]:
    """
    Identifiable two-stage fit:
      1) Fit a,b by flattening log(y*(a*r + b)) across samples (shape fit, scale-free)
      2) Recover cps_per_TBK as exp(mean(log(y*(a*r + b))))
    where y = cps_measured / TBK_true, r = weight/height.

    This breaks the scale degeneracy between cps_per_TBK and (a,b) and is robust to multiplicative noise.
    """
    tbk = df["TBK_true"].to_numpy(float)
    cps = df["cps_measured"].to_numpy(float)
    wkg = df["weight_kg"].to_numpy(float)
    hcm = df["height_cm"].to_numpy(float)

    y = cps / np.maximum(tbk, 1e-12)
    r = _ratio(wkg, hcm)

    # Stage 1: fit (a,b) by minimizing variance of log(y*(a*r+b))
    eps = 1e-12
    def resid_ab(theta_ab):
        a, b = theta_ab
        denom = np.maximum(a * r + b, 1e-6)
        z = np.log(np.maximum(y * denom, eps))
        zc = z - z.mean()          # remove scale; only shape remains
        return zc

    # init from provided init tuple
    a0, b0 = float(init[1]), float(init[2])
    lb = [-5.0, 1e-3]             # allow a slightly negative; b positive
    ub = [ 5.0, 10.0]
    res_ab = least_squares(resid_ab, x0=np.array([a0, b0], float), bounds=(lb, ub), max_nfev=20000)
    a_hat, b_hat = map(float, res_ab.x)

    # Stage 2: recover cps_per_TBK as the geometric mean of y*(a*r+b)
    denom = np.maximum(a_hat * r + b_hat, 1e-6)
    z = np.log(np.maximum(y * denom, eps))
    cps_per_tbk_hat = float(np.exp(z.mean()))

    return {"cps_per_TBK": cps_per_tbk_hat, "a": a_hat, "b": b_hat}

def save_params(params: Dict[str, float], path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

def load_params(path: str | Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fit_from_file(phantom_path: str | Path, out_json: str | Path = "docs/calibration/calib_params.json",
                  init: Tuple[float, float, float] = (100.0, 0.30, 0.70)) -> Dict[str, float]:
    df = load_phantoms(phantom_path)
    params = fit_params(df, init=init)
    save_params(params, out_json)
    return params
