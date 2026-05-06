"""Burmudean product definition."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._utils import OptionType
from diffusions._utils import InitialVarianceStrategy
from diffusions.heston import HestonPathSimulator, HestonPaths

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
        exercise_times: ArrayLike, 
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
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.last_variance = last_variance
        self.strategy = strategy

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

        exercise_indices = self._time_indices(paths.times)
        values = self.payoff(paths.spot[:, exercise_indices[-1]])
        exercise_time_index = np.full(self.n_paths, exercise_indices[-1], dtype=int)

        for idx in reversed(exercise_indices[:-1]):
            continuation_cashflow = values * np.exp(
                -self.simulator.r
                * (paths.times[exercise_time_index] - paths.times[idx])
            )
            immediate = self.payoff(paths.spot[:, idx])
            itm = immediate > 0.0

            continuation = np.zeros(self.n_paths, dtype=float)
            if np.count_nonzero(itm) > self._basis_size:
                basis = self._basis(paths, idx, itm)
                coefficients, *_ = np.linalg.lstsq(
                    basis,
                    continuation_cashflow[itm],
                    rcond=None,
                )
                continuation[itm] = basis @ coefficients

            exercise_now = itm & (immediate >= continuation)
            values[exercise_now] = immediate[exercise_now]
            exercise_time_index[exercise_now] = idx

        discounted_to_zero = values * np.exp(
            -self.simulator.r * paths.times[exercise_time_index]
        )

        price = float(np.mean(discounted_to_zero))
        standard_error = float(np.std(discounted_to_zero, ddof=1) / np.sqrt(self.n_paths))

        return price, standard_error
    
    def _time_indices(self, times: np.ndarray) -> list[int]:
        indices = []
        for t in self.exercise_times:
            idx = int(np.argmin(np.abs(times - t)))
            if not np.isclose(times[idx], t):
                raise ValueError(
                    "Exercise times must lie on the simulation grid. "
                    f"Closest grid point to {t} is {times[idx]}."
                )
            indices.append(idx)
        return sorted(set(indices))

    @property
    def _basis_size(self) -> int:
        # 1, S, v, S^2, S*v, v^2 for degree 2.
        return 6 if self.polynomial_degree >= 2 else 3
    
    def _basis(self, paths: HestonPaths, time_index: int, mask: ArrayLike) -> ArrayLike:
        spot = paths.spot[mask, time_index] / self.simulator.s0
        variance = paths.variance[mask, time_index] / self.simulator.theta
        columns = [np.ones_like(spot), spot, variance]
        if self.polynomial_degree >= 2:
            columns.extend([spot**2, spot * variance, variance**2])
        return np.column_stack(columns)