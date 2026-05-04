"""Path simulators for the stationary CIR variance and Heston spot."""

from __future__ import annotations
from enum import Enum

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


class InitialVarianceStrategy(str, Enum):
    GAMMA = "GAMMA"
    LAST_VALUE = "LAST_VALUE"
    MEAN = "MEAN"

@dataclass(frozen = True)
class HestonPaths:
    """Container returned by the Heston simulator."""

    times: np.ndarray
    spot: np.ndarray
    variance: np.ndarray
    log_spot: np.ndarray
    initial_variance: np.ndarray

    @property
    def terminal_spot(self) -> np.ndarray:
        return self.spot[:, -1]


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


class HestonPathSimulator:
    """Hybrid Stationary Heston simulator.

    The paper uses Milstein on the boosted variance Y_t = exp(kappa t) v_t and
    Euler-Maruyama on X_t = log(S_t). This class implements that time scheme
    pathwise for Monte Carlo pricing.
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

        self.feller_lhs = self.xi**2
        self.feller_rhs = 2.0 * self.kappa * self.theta
        self.satisfies_feller = self.feller_lhs <= self.feller_rhs

        self.stationary_gamma_shape = 2.0 * self.kappa * self.theta / self.xi**2
        self.stationary_gamma_rate = 2.0 * self.kappa / self.xi**2
        self.stationary_gamma_scale = 1.0 / self.stationary_gamma_rate
        self.rng = rng or np.random.default_rng()
        self.cir = CIRStationarySimulator(self.theta,
                                          self.kappa,
                                          self.stationary_gamma_shape,
                                          self.stationary_gamma_scale,
                                          self.rng)

    def simulate(
        self,
        maturity: float,
        n_steps: int,
        n_paths: int,
        strategy: InitialVarianceStrategy = InitialVarianceStrategy.GAMMA,
        last_variance: float | ArrayLike | None = None,
    ) -> HestonPaths:
        if maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")
        if n_paths < 1:
            raise ValueError("n_paths must be at least 1.")

        times = np.linspace(0.0, maturity, n_steps + 1)
        h = maturity / n_steps
        sqrt_h = np.sqrt(h)
        rho = self.rho
        orthogonal_weight = np.sqrt(max(0.0, 1.0 - rho**2))

        initial_variance = self.cir.initial_variance(strategy, n_paths, last_variance)
        log_spot = np.empty((n_paths, n_steps + 1), dtype=float)
        variance = np.empty((n_paths, n_steps + 1), dtype=float)
        log_spot[:, 0] = np.log(self.s0)
        variance[:, 0] = initial_variance

        boosted_variance = initial_variance.copy()
        for k, t in enumerate(times[:-1]):
            z_variance = self.rng.standard_normal(n_paths)
            z_independent = self.rng.standard_normal(n_paths)
            z_spot = rho * z_variance + orthogonal_weight * z_independent

            v_t = np.exp(-self.kappa * t) * boosted_variance
            v_t = np.maximum(v_t, 0.0)
            log_spot[:, k + 1] = (
                log_spot[:, k]
                + (self.r - self.q - 0.5 * v_t) * h
                + np.sqrt(v_t) * sqrt_h * z_spot
            )

            boosted_variance = _boosted_milstein_step(
                boosted_variance,
                t,
                h,
                z_variance,
                self.kappa,
                self.theta,
                self.xi
            )
            variance[:, k + 1] = (
                np.exp(-self.kappa * times[k + 1]) * boosted_variance
            )

        return HestonPaths(
            times = times,
            spot = np.exp(log_spot),
            variance = variance,
            log_spot = log_spot,
            initial_variance = initial_variance,
        )


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
