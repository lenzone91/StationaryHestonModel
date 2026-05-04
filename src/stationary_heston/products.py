"""Option product definitions."""

from __future__ import annotations
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from .simulator import HestonPathSimulator, HestonPaths, InitialVarianceStrategy

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

class BarrierDirection(str, Enum):
    UP_AND_OUT = "up-and-out"
    DOWN_AND_OUT = "down-and-out"


class EuropeanOption:
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
        
        payoff = self.payoff(paths.terminal_spot)
        if self.use_brownian_bridge:
            survival = self._bridge_survival_probability(paths)
            discounted = np.exp(-self.simulator.r * self.maturity) * payoff * survival
        else:
            alive = self._discrete_barrier_survival(paths.spot)
            discounted = np.exp(-self.simulator.r * self.maturity) * payoff * alive
        
        price = float(np.mean(discounted))
        standard_error = float(np.std(discounted, ddof=1) / np.sqrt(self.n_paths))

        return price, standard_error

    def _bridge_survival_probability(self, paths: HestonPaths) -> ArrayLike:
        log_barrier = np.log(self.barrier)
        x0 = paths.log_spot[:, :-1]
        x1 = paths.log_spot[:, 1:]
        dt = np.diff(paths.times)[None, :]
        variance = np.maximum(paths.variance[:, :-1], 1e-16)
        variance_time = variance * dt

        if self.direction == BarrierDirection.UP_AND_OUT:
            impossible = (x0 >= log_barrier) | (x1 >= log_barrier)
            exponent = -2.0 * (log_barrier - x0) * (log_barrier - x1) / variance_time
        else:
            impossible = (x0 <= log_barrier) | (x1 <= log_barrier)
            exponent = -2.0 * (x0 - log_barrier) * (x1 - log_barrier) / variance_time

        step_survival = 1.0 - np.exp(np.minimum(exponent, 0.0))
        step_survival[impossible] = 0.0
        return np.prod(step_survival, axis=1)


    def _discrete_barrier_survival(spot: np.ndarray, option: BarrierOption) -> ArrayLike:
        if option.direction == BarrierDirection.UP_AND_OUT:
            return np.max(spot, axis=1) < option.barrier
        return np.min(spot, axis=1) > option.barrier