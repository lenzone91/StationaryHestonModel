"""European product definition."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._utils import OptionType
from diffusions._utils import InitialVarianceStrategy
from diffusions.heston import HestonPathSimulator

class EuropeanOption:
    def __init__(self, 
        simulator: HestonPathSimulator,
        strike: float,
        maturity: float,
        option_type: OptionType = OptionType.CALL,
        n_paths: int = 10000,
        n_steps: int = 10000,
        last_variance: float | ArrayLike | None = None,
        strategy: InitialVarianceStrategy = InitialVarianceStrategy.GAMMA
    ) -> None:

        if strike <= 0.0:
            raise ValueError("strike must be positive.")
        if maturity <= 0.0:
            raise ValueError("maturity must be positive.")

        self.simulator = simulator
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.strategy = strategy
        self.last_variance = last_variance

    def payoff(self, spot: ArrayLike) -> ArrayLike:
        if self.option_type == OptionType.CALL:
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)

    def price(self) -> tuple[ArrayLike, ArrayLike]:

        paths = self.simulator.simulate(
            self.maturity,
            self.n_steps,
            self.n_paths,
            self.strategy,
            self.last_variance,
        )
        
        discounted = np.exp(-self.simulator.r * self.maturity) * self.payoff(paths.terminal_spot)
        
        price = float(np.mean(discounted))
        standard_error = float(np.std(discounted, ddof=1) / np.sqrt(self.n_paths))

        return price, standard_error