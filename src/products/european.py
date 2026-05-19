"""European product definition."""
from __future__ import annotations

import numpy as np

from ._utils import OptionType
from diffusions.cir import InitialVarianceStrategy
from diffusions.heston import HestonPathSimulator

class EuropeanOption:
    """
    European option class.
    """

    def __init__(self, 
        simulator: HestonPathSimulator,
        strike: float,
        maturity: float,
        option_type: OptionType = OptionType.CALL,
        n_paths: int = 10000,
        n_steps: int = 10000,
        last_variance: float | np.ndarray | None = None,
        strategy: InitialVarianceStrategy = InitialVarianceStrategy.GAMMA
    ) -> None:
        """
        Parameters:
        -----------
        simulator: HestonPathSimulator
            Heston path simulator 
        strike: float
            Strike
        maturity: float
            Product maturity
        option_type: OptionType
            Type of the option ["CALL", "PUT] ("CALL" by default)
        n_paths: int = 10000
            Number of paths (10000 by default)
        n_steps: int = 10000
            Number of step (10000 by default)
        last_variance: float | np.ndarray | None
            Value of the last variance (not useful for strategy == "GAMMA")
        strategy: InitialVarianceStrategy
            Initial variance strategy ["GAMMA", "LAST_VALUE", "MEAN"] (default = "GAMMA")
        """

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

    def payoff(self, spot: np.ndarray) -> np.ndarray:
        """
        Payoff formula of the product

        Parameters
        ----------
        spot: np.ndarray
            Current value

        Returns
        -------
        np.ndarray
            Payoff values
        """
        if self.option_type == OptionType.CALL:
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)

    def price(self) -> tuple[float, float]:
        """
        Price function of the product

        Returns
        -------
        float
            Monte Carlo price
        float
            Standard error of the price
        """
        spots, _ = self.simulator.simulate(
            self.maturity,
            self.n_steps,
            self.n_paths,
            self.strategy,
            self.last_variance,
        )
        
        discounted = np.exp(-self.simulator.r * self.maturity) * self.payoff(spots[:, -1])
        
        price = float(np.mean(discounted))
        standard_error = float(np.std(discounted, ddof=1) / np.sqrt(self.n_paths))

        return price, standard_error