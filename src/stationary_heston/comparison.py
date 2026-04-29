"""Utilities to compare prices across initial variance strategies."""

from __future__ import annotations

from collections.abc import Mapping

from .models import InitialVarianceStrategy


class StrategyPrice:
    def __init__(
        self,
        strategy: InitialVarianceStrategy,
        price: float,
        standard_error: float,
    ) -> None:
        self.strategy = strategy
        self.price = price
        self.standard_error = standard_error


def compare_initialisations(
    pricer,
    option,
    n_paths: int,
    n_steps: int,
    last_variance: float | None = None,
) -> Mapping[InitialVarianceStrategy, StrategyPrice]:
    """Price one product under Gamma, last-value and mean initial variance."""

    results = {}
    for strategy in InitialVarianceStrategy:
        kwargs = {}
        if strategy == InitialVarianceStrategy.LAST_VALUE:
            kwargs["last_variance"] = last_variance
        result = pricer.price(
            option,
            n_paths=n_paths,
            n_steps=n_steps,
            strategy=strategy,
            **kwargs,
        )
        results[strategy] = StrategyPrice(
            strategy=strategy,
            price=result.price,
            standard_error=result.standard_error,
        )
    return results
