from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class GeometryFeatures:
    """Simple anthropometry for geometry correction."""
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return None
        return cls(**{k: d.get(k) for k in ("weight_kg", "height_cm")})

@dataclass
class CPS2TBKCalib:
    """
    Geometry-aware CPS↔TBK mapping.
    - Base 'cps_per_TBK' comes from lab calibration.
    - Efficiency correction is a multiplicative factor based on geometry (BOMAB-like trend).
    - Uncertainty on geometry correction can be sampled as lognormal with rel sigma 'geom_rel_sigma'.
    """
    cps_per_TBK: float = 100.0    # counts/s per unit TBK (lab base)
    # Linear-in-ratio model for efficiency correction:
    # eff_corr = 1 / (a * (weight_kg/height_cm) + b)
    # Typical behaviour: heavier (for same height) => lower efficiency => larger TBK for same cps.
    a: float = 0.30
    b: float = 0.70
    geom_rel_sigma: float = 0.03  # relative sigma for geometry correction (lognormal)

    def efficiency_correction_mean(self, geom: GeometryFeatures | None) -> float:
        if geom is None or geom.weight_kg is None or geom.height_cm is None:
            return 1.0
        ratio = float(geom.weight_kg) / max(float(geom.height_cm), 1e-6)
        denom = max(self.a * ratio + self.b, 1e-3)
        return 1.0 / denom

    def draw_efficiency_correction(self, rng: np.random.Generator, geom: GeometryFeatures | None, size: int) -> np.ndarray:
        """Lognormal draws centered at mean efficiency correction with rel sigma 'geom_rel_sigma'."""
        mean = self.efficiency_correction_mean(geom)
        rel = max(self.geom_rel_sigma, 1e-8)
        var = (rel * mean) ** 2
        sigma2 = np.log(1.0 + var / (mean ** 2))
        mu = np.log(mean) - 0.5 * sigma2
        return rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=size)
