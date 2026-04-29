"""Monte Carlo pricers for European, Bermudan and barrier options."""

from __future__ import annotations

import numpy as np

from .models import InitialVarianceStrategy
from .products import BarrierDirection, BarrierOption, BermudanOption, EuropeanOption
from .simulator import HestonPathSimulator, HestonPaths


class PriceResult:
    def __init__(self, price: float, standard_error: float, n_paths: int) -> None:
        self.price = price
        self.standard_error = standard_error
        self.n_paths = n_paths


class EuropeanMonteCarloPricer:
    def __init__(self, simulator: HestonPathSimulator) -> None:
        self.simulator = simulator

    def price(
        self,
        option: EuropeanOption,
        n_paths: int,
        n_steps: int,
        strategy: InitialVarianceStrategy = InitialVarianceStrategy.GAMMA,
        last_variance: float | np.ndarray | None = None,
    ) -> PriceResult:
        paths = self.simulator.simulate(
            option.maturity,
            n_steps,
            n_paths,
            strategy=strategy,
            last_variance=last_variance,
        )
        discounted = np.exp(-self.simulator.params.r * option.maturity) * option.payoff(
            paths.terminal_spot
        )
        return _price_result(discounted)


class BermudanLSMPricer:
    """Longstaff-Schwartz Bermudan pricer.

    The paper prices Bermudans through quantized backward induction. This first
    version keeps the same optimal-stopping structure, but estimates conditional
    expectations by regression on simulated Heston states.
    """

    def __init__(self, simulator: HestonPathSimulator, polynomial_degree: int = 2) -> None:
        self.simulator = simulator
        self.polynomial_degree = polynomial_degree

    def price(
        self,
        option: BermudanOption,
        n_paths: int,
        n_steps: int,
        strategy: InitialVarianceStrategy = InitialVarianceStrategy.GAMMA,
        last_variance: float | np.ndarray | None = None,
    ) -> PriceResult:
        paths = self.simulator.simulate(
            option.maturity,
            n_steps,
            n_paths,
            strategy=strategy,
            last_variance=last_variance,
        )
        exercise_indices = _time_indices(paths.times, option.exercise_times)
        values = option.payoff(paths.spot[:, exercise_indices[-1]])
        exercise_time_index = np.full(n_paths, exercise_indices[-1], dtype=int)

        for idx in reversed(exercise_indices[:-1]):
            continuation_cashflow = values * np.exp(
                -self.simulator.params.r
                * (paths.times[exercise_time_index] - paths.times[idx])
            )
            immediate = option.payoff(paths.spot[:, idx])
            itm = immediate > 0.0

            continuation = np.zeros(n_paths, dtype=float)
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
            -self.simulator.params.r * paths.times[exercise_time_index]
        )
        return _price_result(discounted_to_zero)

    @property
    def _basis_size(self) -> int:
        # 1, S, v, S^2, S*v, v^2 for degree 2.
        return 6 if self.polynomial_degree >= 2 else 3

    def _basis(self, paths: HestonPaths, time_index: int, mask: np.ndarray) -> np.ndarray:
        spot = paths.spot[mask, time_index] / self.simulator.params.s0
        variance = paths.variance[mask, time_index] / self.simulator.params.theta
        columns = [np.ones_like(spot), spot, variance]
        if self.polynomial_degree >= 2:
            columns.extend([spot**2, spot * variance, variance**2])
        return np.column_stack(columns)


class BarrierMonteCarloPricer:
    """Barrier pricer with optional Brownian-bridge survival correction."""

    def __init__(self, simulator: HestonPathSimulator) -> None:
        self.simulator = simulator

    def price(
        self,
        option: BarrierOption,
        n_paths: int,
        n_steps: int,
        strategy: InitialVarianceStrategy = InitialVarianceStrategy.GAMMA,
        last_variance: float | np.ndarray | None = None,
    ) -> PriceResult:
        paths = self.simulator.simulate(
            option.maturity,
            n_steps,
            n_paths,
            strategy=strategy,
            last_variance=last_variance,
        )
        payoff = option.payoff(paths.terminal_spot)
        if option.use_brownian_bridge:
            survival = _bridge_survival_probability(paths, option)
            discounted = np.exp(-self.simulator.params.r * option.maturity) * payoff * survival
        else:
            alive = _discrete_barrier_survival(paths.spot, option)
            discounted = np.exp(-self.simulator.params.r * option.maturity) * payoff * alive
        return _price_result(discounted)


def _time_indices(times: np.ndarray, requested_times: tuple[float, ...]) -> list[int]:
    indices = []
    for t in requested_times:
        idx = int(np.argmin(np.abs(times - t)))
        if not np.isclose(times[idx], t):
            raise ValueError(
                "Exercise times must lie on the simulation grid. "
                f"Closest grid point to {t} is {times[idx]}."
            )
        indices.append(idx)
    return sorted(set(indices))


def _bridge_survival_probability(paths: HestonPaths, option: BarrierOption) -> np.ndarray:
    log_barrier = np.log(option.barrier)
    x0 = paths.log_spot[:, :-1]
    x1 = paths.log_spot[:, 1:]
    dt = np.diff(paths.times)[None, :]
    variance = np.maximum(paths.variance[:, :-1], 1e-16)
    variance_time = variance * dt

    if option.direction == BarrierDirection.UP_AND_OUT:
        impossible = (x0 >= log_barrier) | (x1 >= log_barrier)
        exponent = -2.0 * (log_barrier - x0) * (log_barrier - x1) / variance_time
    else:
        impossible = (x0 <= log_barrier) | (x1 <= log_barrier)
        exponent = -2.0 * (x0 - log_barrier) * (x1 - log_barrier) / variance_time

    step_survival = 1.0 - np.exp(np.minimum(exponent, 0.0))
    step_survival[impossible] = 0.0
    return np.prod(step_survival, axis=1)


def _discrete_barrier_survival(spot: np.ndarray, option: BarrierOption) -> np.ndarray:
    if option.direction == BarrierDirection.UP_AND_OUT:
        return np.max(spot, axis=1) < option.barrier
    return np.min(spot, axis=1) > option.barrier


def _price_result(discounted_payoffs: np.ndarray) -> PriceResult:
    n_paths = discounted_payoffs.size
    price = float(np.mean(discounted_payoffs))
    standard_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths))
    return PriceResult(price=price, standard_error=standard_error, n_paths=n_paths)
