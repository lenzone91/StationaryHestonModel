"""Stationary Heston framework."""

from __future__ import annotations

import numpy as np

from products import (
    BarrierDirection,
    BarrierOption,
    BermudanOption,
    EuropeanOption,
    OptionType,
)
from diffusions import HestonPathSimulator


def main():

    n_paths = 20000
    n_steps = 6*100
    last_variance = 0.04

    rng = np.random.default_rng(42)

    simulator = HestonPathSimulator(s0 = 100, r = -0.0032, q = 0.00225, kappa = 19.28,
                                    theta = 0.02691, xi = 1.15, rho = -0.99, rng = rng)

    print("European call")
    for strat in ["GAMMA", "LAST_VALUE", "MEAN"]:

        european = EuropeanOption(
            simulator = simulator,
            strike = 100,
            maturity = 0.5, 
            option_type = OptionType.CALL,
            n_paths = n_paths,
            n_steps = n_steps,
            last_variance = last_variance,
            strategy = strat
        )
        price, standard_error = european.price()
        print(f"  {strat}: {price:8.4f} +/- {1.96 * standard_error:7.4f}")

    print("Bermudan put")
    for strat in ["GAMMA", "LAST_VALUE", "MEAN"]:
        bermudan = BermudanOption(
            simulator = simulator,
            strike = 100.0,
            maturity = 0.5,
            exercise_times = np.linspace(0.05, 0.25, 5),
            option_type = OptionType.PUT,
            polynomial_degree = 3,
            n_paths = n_paths,
            n_steps = n_steps,
            last_variance = last_variance,
            strategy = strat
        )
        price, standard_error = bermudan.price()
        print(f"  {strat}: {price:8.4f} +/- {1.96 * standard_error:7.4f}")

    print("Barrier call")
    for strat in ["GAMMA", "LAST_VALUE", "MEAN"]:
        barrier = BarrierOption(
            simulator = simulator,
            strike = 100.0,
            maturity = 0.25,
            barrier = 115.0,
            direction = BarrierDirection.UP_AND_OUT,
            option_type = OptionType.CALL,
            use_brownian_bridge = True,
            n_paths = n_paths,
            n_steps = n_steps,
            last_variance = last_variance,
            strategy = strat
        )
        price, standard_error = barrier.price()
        print(f"  {strat}: {price:8.4f} +/- {1.96 * standard_error:7.4f}")


if __name__ == "__main__":
    main()
