"""Measurements with confidence intervals, shared by every tier.

Every number in the output is a `Measurement`: value, 1-sigma, 95% interval and the error
model that produced it. Error models are explicit sums of independent terms so the
calibration step (benchmark) can scale each tier's terms with one factor per tier.

LiDAR-tier error terms (all 1-sigma):
  face position     : robust spread of the face inliers / sqrt(n_eff), plus
                      SIGMA_FACE_SYS (ARKit depth bias + residual drift across the face)
  length scale      : SCALE_REL * length (iPhone LiDAR depth scale error)
  unsnapped edge    : SIGMA_UNSNAPPED (edge bounded by free space only, no wall face)
  opening jamb      : SIGMA_JAMB if snapped to a wall-face run end, else SIGMA_JAMB_RAW
  floor / ceiling   : spread of plane inliers / sqrt(n_eff) + SIGMA_LEVEL_SYS
Calibration multiplies the whole sigma by tier factor k (TIER_K), fitted on repeatability
pairs and ground truth when available (see benchmark/calibrate.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z95 = 1.959964

# LiDAR-tier constants (metres / relative); see docstring
SIGMA_FACE_SYS = 0.004
# iPhone/iPad LiDAR depth scale: 0.7% short vs laser on ARKitScenes visit 422382 (ceiling
# pred/GT 0.990-0.997, benchmark/arkitscenes/report_422382.md); one visit is not enough to
# correct the bias, so it enters as 1-sigma uncertainty. Held-out check: visit 422386.
SCALE_REL = 0.007
SIGMA_UNSNAPPED = 0.04
SIGMA_JAMB = 0.006
SIGMA_JAMB_RAW = 0.03
SIGMA_LEVEL_SYS = 0.003

# Per-tier calibration multipliers on sigma (updated by benchmark/calibrate.py)
TIER_K = {"lidar": 1.0, "video": 1.0, "photo": 1.0}


@dataclass
class Measurement:
    value: float
    sigma: float
    method: str
    unit: str = "m"
    observed: bool = True

    def to_json(self, k: float = 1.0, nd: int = 4) -> dict:
        s = self.sigma * k
        return {
            "value": round(float(self.value), nd),
            "unit": self.unit,
            "sigma": round(float(s), nd),
            "ci95": [round(float(self.value - Z95 * s), nd), round(float(self.value + Z95 * s), nd)],
            "observed": self.observed,
            "method": self.method,
        }


def unobserved(prior: float, sigma: float, method: str, unit: str = "m") -> Measurement:
    """A quantity the capture did not observe: report a prior with an honest (wide) interval."""
    return Measurement(prior, sigma, method, unit, observed=False)


def face_sigma(fit_sigma: float, support: int) -> float:
    # neighbouring 1.5 cm voxels are correlated; effective samples ~ support / 20
    n_eff = max(support / 20.0, 1.0)
    return float(np.hypot(fit_sigma / np.sqrt(n_eff), SIGMA_FACE_SYS))


def length_sigma(length: float, s_a: float, s_b: float) -> float:
    return float(np.sqrt(s_a**2 + s_b**2 + (SCALE_REL * length) ** 2))
