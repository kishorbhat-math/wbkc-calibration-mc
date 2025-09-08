from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .calibration import CPS2TBKCalib

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

def _predict_cps(tbk: np.ndarray, wkg: np.ndarray, hcm: np.ndarray, cps_per_tbk: float, a: float, b: float) -> np.ndarray:
    ratio = wkg / np.maximum(hcm, 1e-6)
    eff_corr = 1.0 / np.maximum(a * ratio + b, 1e-3)   # dimensionless
    cps_eff  = cps_per_tbk * eff_corr
    return tbk * cps_eff

def fit_params(df: pd.DataFrame, init: Tuple[float, float, float] = (100.0, 0.30, 0.70)) -> Dict[str, float]:
    """
    Fit cps_per_TBK, a, b by minimizing LOG residuals (robust to multiplicative noise):
        log(cps_measured) ~= log( TBK_true * cps_per_TBK / (a*(w/h)+b) )
    """
    tbk = df["TBK_true"].to_numpy(float)
    cps = df["cps_measured"].to_numpy(float)
    wkg = df["weight_kg"].to_numpy(float)
    hcm = df["height_cm"].to_numpy(float)

    eps = 1e-12
    def residuals(theta):
        cps_per_tbk, a, b = theta
        pred = _predict_cps(tbk, wkg, hcm, cps_per_tbk, a, b)
        # log-residuals; guard small values
        return np.log(np.maximum(cps, eps)) - np.log(np.maximum(pred, eps))

    # Keep cps_per_tbk positive; b positive; a can be slightly negative if data suggest
    lb = [1e-6, -5.0, 1e-3]
    ub = [1e6,   5.0, 10.0]
    res = least_squares(residuals, x0=np.array(init, float), bounds=(lb, ub), max_nfev=20000)
    cps_per_tbk, a, b = map(float, res.x)
    return {"cps_per_TBK": cps_per_tbk, "a": a, "b": b}

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
