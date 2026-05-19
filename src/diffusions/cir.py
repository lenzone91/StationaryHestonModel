"""Path simulators for the stationary CIR variance"""

from __future__ import annotations
from warnings import warn
from enum import Enum

import numpy as np

class InitialVarianceStrategy(str, Enum):
    GAMMA = "GAMMA"
    LAST_VALUE = "LAST_VALUE"
    MEAN = "MEAN"

class CIRStationarySimulator:
    """CIR helper with stationary Gamma initialisation."""

    def __init__(self, 
        theta: float,
        kappa: float,
        xi: float,
        rng: np.random.Generator | None = None
    ) -> None:
        """
        Initialize a stationary CIR variance simulator.

        Parameters
        ----------
        theta : float
            Long-run mean level of the CIR variance process.
        kappa : float
            Mean-reversion speed of the CIR variance process.
        xi : float
            Volatility of volatility of the CIR variance process.
        rng : np.random.Generator | None, optional
            Random number generator used for path simulation.

        Returns
        -------
        None
        """
        
        if not xi**2 <= 2.0 * kappa * theta:
            warn("The Feller condition is not satisfied.")
        
        self.theta = theta
        self.kappa = kappa
        self.xi = xi
        self.rng = rng or np.random.default_rng() 
        
        self.stationary_gamma_shape = 2.0 * self.kappa * self.theta / self.xi**2
        self.stationary_gamma_rate = 2.0 * self.kappa / self.xi**2
        self.stationary_gamma_scale = 1.0 / self.stationary_gamma_rate

    def initial_variance(
        self,
        strategy: InitialVarianceStrategy,
        n_paths: int,
        last_variance: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build a vector of initial variances from the requested strategy.

        Parameters
        ----------
        strategy : InitialVarianceStrategy
            Initial variance strategy among ``"GAMMA"``, ``"LAST_VALUE"``,
            and ``"MEAN"``.
        n_paths : int
            Number of simulated paths.
        last_variance : float | np.ndarray | None, optional
            Scalar variance value or array of shape ``(n_paths,)`` used when
            ``strategy == "LAST_VALUE"``.

        Returns
        -------
        np.ndarray
            Initial variance values with shape ``(n_paths,)``.
        """

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
            return values
        raise ValueError(f"Unsupported initial variance strategy: {strategy!r}")

    def simulate(
        self,
        maturity: float,
        n_steps: int,
        n_paths: int,
        strategy: InitialVarianceStrategy = InitialVarianceStrategy.GAMMA,
        last_variance: float | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate CIR variance paths with the boosted Milstein scheme.

        Parameters
        ----------
        maturity : float
            Time horizon of the simulation.
        n_steps : int
            Number of time steps.
        n_paths : int
            Number of simulated paths.
        strategy : InitialVarianceStrategy, optional
            Initial variance strategy among ``"GAMMA"``, ``"LAST_VALUE"``,
            and ``"MEAN"``.
        last_variance : float | np.ndarray | None, optional
            Scalar variance value or array of shape ``(n_paths,)`` used when
            ``strategy == "LAST_VALUE"``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pair ``(times, variance)`` where ``times`` has shape
            ``(n_steps + 1,)`` and ``variance`` has shape
            ``(n_paths, n_steps + 1)``.
        """

        times = np.linspace(0.0, maturity, n_steps + 1)
        h = maturity / n_steps
        variance = np.empty((n_paths, n_steps + 1), dtype=float)
        variance[:, 0] = self.initial_variance(strategy, n_paths, last_variance)

        boosted = variance[:, 0].copy()
        for k, t in enumerate(times[:-1]):
            z = self.rng.standard_normal(n_paths)
            boosted = self.boosted_milstein_step(boosted, t, h, z)
            variance[:, k + 1] = np.exp(-self.kappa * times[k + 1]) * boosted

        return times, variance
    
    def boosted_milstein_step(
        self,
        y: np.ndarray,
        t: float,
        h: float,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Perform one boosted Milstein step for the transformed CIR variance
        process Y_t = exp(kappa * t) * v_t.

        Parameters
        ----------
        y : np.ndarray
            Current value of the transformed variance process Y_t
        t : float
            Current time
        h : float
            Timestep size
        z : np.ndarray
            Standard normal random draws, one per path

        Returns
        -------
        np.ndarray
            Next value Y_{t+h} of the transformed variance process,
            floored at 1e-6 for numerical stability.
        """

        exp_kt = np.exp(self.kappa * t)
        exp_half_kt = np.exp(0.5 * self.kappa * t)
        square_shift = 2.0 * np.sqrt(y) / (
            np.sqrt(h) * self.xi * exp_half_kt
        )
        next_y = (
            h * exp_kt * (self.kappa * self.theta - 0.25 * self.xi**2)
            + 0.25 * h * self.xi**2 * exp_kt * (z + square_shift) ** 2
        )
        return np.maximum(next_y, 1e-6)
