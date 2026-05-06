"""Path simulators for the stationary CIR variance"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._utils import InitialVarianceStrategy, _boosted_milstein_step

class CIRStationarySimulator:
    """CIR helper with stationary Gamma initialisation."""

    def __init__(self, 
        theta: float,
        kappa: float,
        stationary_gamma_shape: float,
        stationary_gamma_scale: float,
        rng: np.random.Generator | None = None
    ) -> None:
        
        self.theta = theta
        self.kappa = kappa
        self.stationary_gamma_shape = stationary_gamma_shape
        self.stationary_gamma_scale = stationary_gamma_scale
        self.rng = rng or np.random.default_rng() 

    def initial_variance(
        self,
        strategy: InitialVarianceStrategy,
        n_paths: int,
        last_variance: float | ArrayLike | None = None,
    ) -> np.ndarray:
        """Build a vector of initial variances from the requested strategy."""

        if strategy == "GAMMA":
            return self.rng.gamma(
                shape = self.stationary_gamma_shape,
                scale = self.stationary_gamma_scale,
                size = n_paths,
            )
        if strategy == "MEAN":
            return np.full(n_paths, self.theta, dtype = float)
        if strategy == "LAST_VALUE":
            if last_variance is None:
                raise ValueError("last_variance is required for LAST_VALUE.")
            values = np.asarray(last_variance, dtype = float)
            if values.ndim == 0:
                return np.full(n_paths, float(values), dtype = float)
            if values.shape != (n_paths):
                raise ValueError("last_variance must be scalar or have shape (n_paths,).")
            return values.copy()
        raise ValueError(f"Unsupported initial variance strategy: {strategy!r}")

    def simulate(
        self,
        maturity: float,
        n_steps: int,
        n_paths: int,
        strategy: InitialVarianceStrategy = InitialVarianceStrategy.GAMMA,
        last_variance: float | ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the standalone CIR variance with the same Milstein scheme.

        This returns (times, variance). The scheme mirrors the Heston variance
        leg, i.e. Milstein on Y_t = exp(kappa t) v_t and back-transforming to v_t.
        """

        times = np.linspace(0.0, maturity, n_steps + 1)
        h = maturity / n_steps
        variance = np.empty((n_paths, n_steps + 1), dtype=float)
        variance[:, 0] = self.initial_variance(strategy, n_paths, last_variance)

        boosted = variance[:, 0].copy()
        for k, t in enumerate(times[:-1]):
            z = self.rng.standard_normal(n_paths)
            boosted = _boosted_milstein_step(boosted, t, h, z)
            variance[:, k + 1] = np.exp(-self.kappa * times[k + 1]) * boosted

        return times, variance