"""Burmudean product definition."""
from __future__ import annotations
from enum import Enum

import numpy as np

from ._utils import OptionType
from diffusions.cir import InitialVarianceStrategy
from diffusions.heston import HestonPathSimulator


class BarrierDirection(str, Enum):
    UP_AND_OUT = "up-and-out"
    DOWN_AND_OUT = "down-and-out"

class BarrierOption:
    """Barrier pricer with optional Brownian-bridge survival correction."""

    def __init__(self, 
        simulator: HestonPathSimulator,
        strike: float,
        maturity: float,
        barrier: float,
        direction: BarrierDirection = BarrierDirection.UP_AND_OUT,
        option_type: OptionType = OptionType.CALL,
        use_brownian_bridge: bool = True,
        n_paths: int = 10000,
        n_steps: int = 10000,
        last_variance: float | np.ndarray | None = None,
        strategy: InitialVarianceStrategy = InitialVarianceStrategy.GAMMA
    ) -> None:
        
        if strike <= 0.0:
            raise ValueError("strike must be positive.")
        if maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        if barrier <= 0.0:
            raise ValueError("barrier must be positive.")
        
        self.simulator = simulator
        self.strike = strike
        self.maturity = maturity
        self.barrier = barrier
        self.direction = direction
        self.option_type = option_type
        self.use_brownian_bridge = use_brownian_bridge
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.last_variance = last_variance
        self.strategy = strategy
        self.times = np.linspace(0.0, self.maturity, self.n_steps + 1)


    def payoff(self, spot: np.ndarray) -> np.ndarray:
        if self.option_type == OptionType.CALL:
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)
    
    def price(self) -> tuple[float, float]:
        
        spots, variance = self.simulator.simulate(
            self.maturity,
            self.n_steps,
            self.n_paths,
            self.strategy,
            self.last_variance,
        )
        
        payoff = self.payoff(spots[:, -1])
        if self.use_brownian_bridge:
            survival = self._bridge_survival_probability(np.log(spots), variance)
            discounted = np.exp(-self.simulator.r * self.maturity) * payoff * survival
        else:
            alive = self._discrete_barrier_survival(spots)
            discounted = np.exp(-self.simulator.r * self.maturity) * payoff * alive
        
        price = float(np.mean(discounted))
        standard_error = float(np.std(discounted, ddof=1) / np.sqrt(self.n_paths))

        return price, standard_error

    def _bridge_survival_probability(self, log_spot: np.ndarray, variance: np.ndarray) -> np.ndarray:
        log_barrier = np.log(self.barrier)
        x0 = log_spot[:, :-1]
        x1 = log_spot[:, 1:]
        dt = np.diff(self.times)[None, :]
        variance_time = variance[:, :-1] * dt

        if self.direction == BarrierDirection.UP_AND_OUT:
            impossible = (x0 >= log_barrier) | (x1 >= log_barrier)
            exponent = -2.0 * (log_barrier - x0) * (log_barrier - x1) / variance_time
        else:
            impossible = (x0 <= log_barrier) | (x1 <= log_barrier)
            exponent = -2.0 * (x0 - log_barrier) * (x1 - log_barrier) / variance_time

        step_survival = 1.0 - np.exp(exponent)
        step_survival[impossible] = 0.0
        return np.prod(step_survival, axis=1)


    def _discrete_barrier_survival(self, spot: np.ndarray) -> np.ndarray:
        if self.direction == BarrierDirection.UP_AND_OUT:
            return np.max(spot, axis=1) <= self.barrier
        return np.min(spot, axis=1) >= self.barrier
