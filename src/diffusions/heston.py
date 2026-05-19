"""Path simulators for the Heston"""

from __future__ import annotations
from typing import Tuple

import numpy as np

from .cir import CIRStationarySimulator, InitialVarianceStrategy

class HestonPathSimulator:
    """
    Stationary Heston simulator.
    """

    def __init__(self, s0: float, r: float, q: float,
                 kappa: float, theta: float, xi: float,
                 rho: float, rng: np.random.Generator | None = None) -> None:
        """
        Parameters:
        -----------
        s0: float
            Initial spot.
        r: float
            Risk-free rate.
        q: float
            Continuous dividend yield.
        kappa: float
            CIR mean-reversion speed.
        theta: float
            CIR long-run variance mean.
        xi: float
            Vol-of-vol.
        rho: float
            Correlation between spot and variance Brownian motions.
        rng: np.random.Generator | None
        """
        if s0 <= 0.0:
            raise ValueError("s0 must be positive.")
        if kappa <= 0.0:
            raise ValueError("kappa must be positive.")
        if theta <= 0.0:
            raise ValueError("theta must be positive.")
        if xi <= 0.0:
            raise ValueError("xi must be positive.")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must be in [-1, 1].")

        self.s0 = s0
        self.r = r
        self.q = q
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

        self.rng = rng or np.random.default_rng()
        self.cir = CIRStationarySimulator(self.theta,
                                          self.kappa,
                                          self.xi,
                                          self.rng)

    def simulate(
        self,
        maturity: float,
        n_steps: int,
        n_paths: int,
        strategy: InitialVarianceStrategy = InitialVarianceStrategy.GAMMA,
        last_variance: float | np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
        -----------
        maturity: float
            Simulation horizon
        n_steps: int
            Number of steps
        n_paths: int
            Number of paths
        strategy: InitialVarianceStrategy
            Initial variance strategy ["GAMMA", "LAST_VALUE", "MEAN"] (default = "GAMMA")
        last_variance: float | np.ndarray 
            Value of the last variance (not useful for strategy == "GAMMA")
        """

        if maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")
        if n_paths < 1:
            raise ValueError("n_paths must be at least 1.")

        times = np.linspace(0.0, maturity, n_steps + 1)
        h = maturity / n_steps
        sqrt_h = np.sqrt(h)
        orthogonal_weight = np.sqrt(1.0 - self.rho**2)

        boosted_variance = self.cir.initial_variance(strategy, n_paths, last_variance)
        log_spot = np.empty((n_paths, n_steps + 1), dtype=float)
        variance = np.empty((n_paths, n_steps + 1), dtype=float)
        log_spot[:, 0] = np.log(self.s0)
        variance[:, 0] = boosted_variance

        for k, t in enumerate(times[:-1]):
            z_variance = self.rng.standard_normal(n_paths) # the variance brownian
            z_independent = self.rng.standard_normal(n_paths) # the heston brownian
            z_spot = self.rho * z_variance + orthogonal_weight * z_independent

            v_t = np.exp(-self.kappa * t) * boosted_variance
            v_t = np.maximum(v_t, 1e-6)
            log_spot[:, k + 1] = (
                log_spot[:, k]
                + (self.r - self.q - 0.5 * v_t) * h
                + np.sqrt(v_t) * sqrt_h * z_spot
            )

            boosted_variance = self.cir.boosted_milstein_step(
                boosted_variance,
                t,
                h,
                z_variance,
            )
            variance[:, k + 1] = (
                np.exp(-self.kappa * times[k + 1]) * boosted_variance
            )

        return np.exp(log_spot), variance
    


