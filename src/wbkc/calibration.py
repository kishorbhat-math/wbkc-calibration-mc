
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class GeometryFeatures:
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

@dataclass
class CPS2TBKCalib:
    """Placeholder for CPS↔TBK calibration using geometry-aware efficiency.

    This scaffold mimics WBKC practices where counts-per-second near 1461 keV are mapped to TBK via
    phantom-derived calibrations plus Monte Carlo geometry correction.
    """
    slope: float = 1.0      # TBK per CPS (arbitrary)
    intercept: float = 0.0  # TBK offset (arbitrary)

    def efficiency_correction(self, geom: GeometryFeatures | None) -> float:
        # Simple linear correction by (weight/height). Replace with regression from BOMAB-like phantoms.
        if geom and geom.weight_kg and geom.height_cm:
            ratio = geom.weight_kg / max(geom.height_cm, 1e-6)
            return 1.0 / max(0.3 * ratio + 0.7, 1e-3)
        return 1.0

    def cps_to_tbk(self, cps: float, geom: GeometryFeatures | None = None) -> float:
        corr = self.efficiency_correction(geom)
        return self.slope * cps * corr + self.intercept
