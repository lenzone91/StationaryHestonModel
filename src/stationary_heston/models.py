"""Model parameters and variance initialisation policies.

The Stationary Heston paper replaces the deterministic CIR variance initial
condition by the invariant distribution of the CIR process. With the paper's
Gamma(shape, rate) convention:

    shape = 2 * kappa * theta / xi**2
    rate  = 2 * kappa / xi**2

NumPy uses Gamma(shape, scale), therefore scale = 1 / rate.
"""

from __future__ import annotations

from enum import Enum


class InitialVarianceStrategy(str, Enum):
    """Available initial variance choices for short-maturity comparisons."""

    GAMMA = "gamma"
    LAST_VALUE = "last_value"
    MEAN = "mean"


class HestonParameters:
    """Risk-neutral Heston parameters.

    Attributes:
        s0: Initial spot.
        r: Risk-free rate.
        q: Continuous dividend yield.
        kappa: CIR mean-reversion speed.
        theta: CIR long-run variance mean.
        xi: Vol-of-vol.
        rho: Correlation between spot and variance Brownian motions.
    """

    def __init__(
        self,
        s0: float,
        r: float,
        q: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
    ) -> None:
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

        self.feller_lhs = xi**2
        self.feller_rhs = 2.0 * kappa * theta
        self.satisfies_feller = self.feller_lhs <= self.feller_rhs

        self.stationary_gamma_shape = 2.0 * kappa * theta / xi**2
        self.stationary_gamma_rate = 2.0 * kappa / xi**2
        self.stationary_gamma_scale = 1.0 / self.stationary_gamma_rate

    def __repr__(self) -> str:
        return (
            "HestonParameters("
            f"s0={self.s0}, r={self.r}, q={self.q}, "
            f"kappa={self.kappa}, theta={self.theta}, "
            f"xi={self.xi}, rho={self.rho})"
        )
