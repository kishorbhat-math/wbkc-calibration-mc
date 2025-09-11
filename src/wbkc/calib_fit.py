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

def _two_stage_init(y: np.ndarray, r: np.ndarray, init_ab: Tuple[float, float]) -> Tuple[float, float, float]:
    """Stage-1: pick (a,b) to flatten log(y*(a*r+b)); Stage-2: cps := geometric mean of y*(a*r+b)."""
    eps = 1e-12
    def resid_ab(theta_ab):
        a, b = theta_ab
        denom = np.maximum(a * r + b, 1e-6)
        z = np.log(np.maximum(y * denom, eps))
        zc = z - z.mean()
        return zc
    a0, b0 = float(init_ab[0]), float(init_ab[1])
    lb = [-5.0, 1e-3]   # a can be slightly negative; b positive
    ub = [ 5.0, 10.0]
    res_ab = least_squares(resid_ab, x0=np.array([a0, b0], float), bounds=(lb, ub), max_nfev=30000)
    a_hat, b_hat = map(float, res_ab.x)

    denom = np.maximum(a_hat * r + b_hat, 1e-6)
    z = np.log(np.maximum(y * denom, eps))
    cps_per_tbk_hat = float(np.exp(z.mean()))
    return cps_per_tbk_hat, a_hat, b_hat

def fit_params(df: pd.DataFrame, init: Tuple[float, float, float] = (100.0, 0.30, 0.70)) -> Dict[str, float]:
    """
    Robust identifiable fit for (cps_per_TBK, a, b):
      - Two-stage shape init → (c_init, a_init, b_init)
      - Joint log-residual least-squares on (c, a, b) with a tiny Tikhonov regularization on (a,b)
        Model: log(y) ≈ log(c) - log(a*r + b), y = cps_measured / TBK_true, r = weight/height
    """
    tbk = df["TBK_true"].to_numpy(float)
    cps = df["cps_measured"].to_numpy(float)
    wkg = df["weight_kg"].to_numpy(float)
    hcm = df["height_cm"].to_numpy(float)

    y = cps / np.maximum(tbk, 1e-12)
    r = _ratio(wkg, hcm)

    # Two-stage initializer
    c0, a0, b0 = _two_stage_init(y, r, init_ab=(float(init[1]), float(init[2])))

    # Joint optimization (on c, a, b). Keep them in natural space with positivity on c and b.
    eps = 1e-12
    prior_a, prior_b = 0.30, 0.70
    lam = 1e-3  # small regularization strength

    def residuals(theta):
        c, a, b = theta
        c = float(np.maximum(c, 1e-6))
        b = float(np.maximum(b, 1e-6))
        denom = np.maximum(a * r + b, 1e-6)
        pred = np.log(c) - np.log(denom)
        data_resid = np.log(np.maximum(y, eps)) - pred
        # mild Tikhonov on (a,b)
        reg = np.sqrt(lam) * np.array([a - prior_a, b - prior_b], dtype=float)
        return np.concatenate([data_resid, reg])

    lb = [1e-6, -5.0, 1e-6]
    ub = [1e6,   5.0, 10.0]
    theta0 = np.array([c0, a0, b0], float)
    res = least_squares(residuals, x0=theta0, bounds=(lb, ub), max_nfev=50000)
    c_hat, a_hat, b_hat = map(float, res.x)
    return {"cps_per_TBK": c_hat, "a": a_hat, "b": b_hat}

def save_params(params: Dict[str, float], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
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
