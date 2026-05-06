"""Path simulators for the stationary CIR variance and Heston spot."""

from __future__ import annotations
from enum import Enum


import numpy as np

class InitialVarianceStrategy(str, Enum):
    GAMMA = "GAMMA"
    LAST_VALUE = "LAST_VALUE"
    MEAN = "MEAN"

def _boosted_milstein_step(
    y: np.ndarray,
    t: float,
    h: float,
    z: np.ndarray,
    kappa: float,
    theta: float,
    xi: float
) -> np.ndarray:
    """Milstein step for Y_t = exp(kappa t) v_t.

    Under the Feller condition, the paper writes this as a positive square plus
    a non-negative drift correction. We still clip tiny negative round-off noise.
    """

    exp_kt = np.exp(kappa * t)
    exp_half_kt = np.exp(0.5 * kappa * t)
    square_shift = 2.0 * np.sqrt(np.maximum(y, 0.0)) / (
        np.sqrt(h) * xi * exp_half_kt
    )
    next_y = (
        h * exp_kt * (kappa * theta - 0.25 * xi**2)
        + 0.25 * h * xi**2 * exp_kt * (z + square_shift) ** 2
    )
    return np.maximum(next_y, 0.0)