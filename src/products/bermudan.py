"""Burmudean product definition."""
from __future__ import annotations

import numpy as np

from ._utils import OptionType
from diffusions.cir import InitialVarianceStrategy
from diffusions.heston import HestonPathSimulator

class BermudanOption:
    """Longstaff-Schwartz Bermudan pricer.

    The paper prices Bermudans through quantized backward induction. This first
    version keeps the same optimal-stopping structure, but estimates conditional
    expectations by regression on simulated Heston states.
    """

    def __init__(self,
        simulator: HestonPathSimulator,
        strike: float,
        maturity: float,
        exercise_times: np.ndarray, 
        option_type: OptionType = OptionType.PUT,
        polynomial_degree: int = 2,
        n_paths: int = 10000,
        n_steps: int = 10000,
        last_variance: float | np.ndarray | None = None,
        strategy: InitialVarianceStrategy = InitialVarianceStrategy.GAMMA
    ) -> None:
        
        if strike <= 0.0:
            raise ValueError("strike must be positive.")
        if maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        if exercise_times.size == 0:
            raise ValueError("exercise_times cannot be empty.")
        if min(exercise_times) < 0.0:
            raise ValueError("exercise_times must be non-negative.")
        if max(exercise_times) > maturity:
            raise ValueError("exercise_times cannot exceed maturity.")

        self.simulator = simulator
        self.strike = strike
        self.maturity = maturity
        self.exercise_times = exercise_times
        self.option_type = option_type
        self.polynomial_degree = polynomial_degree
        self.basis_size = (polynomial_degree + 1) * (polynomial_degree + 2) // 2
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.last_variance = last_variance
        self.strategy = strategy
        self.times = np.linspace(0.0, self.maturity, self.n_steps + 1)
        self.exercise_indices = (self.exercise_times / self.maturity * self.n_steps).astype("int")

        if self.exercise_indices not in self.exercise_indices:
            raise ValueError("Exercise times must lie on the simulation grid.")


    def payoff(self, spot: np.ndarray) -> np.ndarray:
        if self.option_type == OptionType.CALL:
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)

    def price(self) -> tuple[np.ndarray, np.ndarray]:

        spots, variance = self.simulator.simulate(
            self.maturity,
            self.n_steps,
            self.n_paths,
            self.strategy,
            self.last_variance,
        )

        # Compute the last payoff 
        values = self.payoff(spots[:, self.exercise_indices[-1]])
        
        # Create the best exercise time vector
        exercise_time_index = np.full(self.n_paths, self.exercise_indices[-1], dtype=int)

        for idx in reversed(self.exercise_indices[:-1]):
            continuation_cashflow = values * np.exp(
                -self.simulator.r
                * (self.times[exercise_time_index] - self.times[idx])
            )
            immediate = self.payoff(spots[:, idx])
            itm = immediate > 0.0

            continuation = np.zeros(self.n_paths, dtype=float)
            
            # Do we have enough values to regress on
            if np.count_nonzero(itm) > self.basis_size:
                basis = self._basis(spots, variance, idx, itm)
                coefficients, *_ = np.linalg.lstsq(
                    basis,
                    continuation_cashflow[itm],
                )
                continuation[itm] = basis @ coefficients

            exercise_now = itm & (immediate >= continuation)
            values[exercise_now] = immediate[exercise_now]
            exercise_time_index[exercise_now] = idx

        discounted_to_zero = values * np.exp(
            -self.simulator.r * self.times[exercise_time_index]
        )

        price = float(np.mean(discounted_to_zero))
        standard_error = float(np.std(discounted_to_zero, ddof=1) / np.sqrt(self.n_paths))

        return price, standard_error
    
    def _basis(self, spots: np.ndarray, variance: np.ndarray, time_index: int, mask: np.ndarray) -> np.ndarray:
        """
        Generates the polynomial basis.
        The order is:
            degree 0: 1
            degree 1: S, v
            degree 2: S^2, S v, v^2
            degree 3: S^3, S^2 v, S v^2, v^3
            etc.        
        """
        spot = spots[mask, time_index] / self.simulator.s0
        var = variance[mask, time_index] / self.simulator.theta
        columns = []

        for degree in range(self.polynomial_degree + 1):
            for power_var in range(degree + 1):
                power_spot = degree - power_var
                columns.append((spot ** power_spot) * (var ** power_var))

        return np.column_stack(columns)