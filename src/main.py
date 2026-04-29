"""Minimal example for the Stationary Heston framework."""

from __future__ import annotations

import numpy as np

from stationary_heston import (
    BarrierDirection,
    BarrierMonteCarloPricer,
    BarrierOption,
    BermudanLSMPricer,
    BermudanOption,
    EuropeanMonteCarloPricer,
    EuropeanOption,
    HestonParameters,
    HestonPathSimulator,
    OptionType,
)
from stationary_heston.comparison import compare_initialisations


def main():
    params = HestonParameters(
        s0 = 100.0,
        r = 0.04,
        q = 0.0,
        kappa = 1.15,
        theta = 0.0348,
        xi = 0.39,
        rho = -0.64,
    )
    simulator = HestonPathSimulator(params, rng = np.random.default_rng(42))

    european = EuropeanOption(strike = 100.0, maturity = 0.25, option_type = OptionType.CALL)
    bermudan = BermudanOption(
        strike = 100.0,
        maturity = 0.25,
        exercise_times = tuple(np.linspace(0.05, 0.25, 5)),
        option_type = OptionType.PUT,
    )
    barrier = BarrierOption(
        strike = 100.0,
        maturity = 0.25,
        barrier = 115.0,
        direction = BarrierDirection.UP_AND_OUT,
        option_type = OptionType.CALL,
    )

    n_paths = 20000
    n_steps = 50
    last_variance = 0.04

    for name, pricer, option in [
        ("European call", EuropeanMonteCarloPricer(simulator), european),
        ("Bermudan put", BermudanLSMPricer(simulator), bermudan),
        ("Up-and-out call", BarrierMonteCarloPricer(simulator), barrier),
    ]:
        print(f"\n{name}")
        results = compare_initialisations(
            pricer,
            option,
            n_paths = n_paths,
            n_steps = n_steps,
            last_variance = last_variance,
        )
        for strategy, result in results.items():
            print(
                f"  {strategy.value:>10}: "
                f"{result.price:8.4f} +/- {1.96 * result.standard_error:7.4f}"
            )

if __name__ == "__main__":
    main()
