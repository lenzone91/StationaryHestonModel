"""Option product definitions."""

from __future__ import annotations

from enum import Enum

import numpy as np


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class BarrierDirection(str, Enum):
    UP_AND_OUT = "up-and-out"
    DOWN_AND_OUT = "down-and-out"


class EuropeanOption:
    def __init__(
        self,
        strike: float,
        maturity: float,
        option_type: OptionType = OptionType.CALL,
    ) -> None:
        if strike <= 0.0:
            raise ValueError("strike must be positive.")
        if maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type

    def payoff(self, spot: np.ndarray) -> np.ndarray:
        if self.option_type == OptionType.CALL:
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)


class BermudanOption:
    def __init__(
        self,
        strike: float,
        maturity: float,
        exercise_times: tuple[float, ...],
        option_type: OptionType = OptionType.PUT,
    ) -> None:
        if strike <= 0.0:
            raise ValueError("strike must be positive.")
        if maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        if not exercise_times:
            raise ValueError("exercise_times cannot be empty.")
        if min(exercise_times) < 0.0:
            raise ValueError("exercise_times must be non-negative.")
        if max(exercise_times) > maturity:
            raise ValueError("exercise_times cannot exceed maturity.")
        self.strike = strike
        self.maturity = maturity
        self.exercise_times = tuple(exercise_times)
        self.option_type = option_type

    def payoff(self, spot: np.ndarray) -> np.ndarray:
        if self.option_type == OptionType.CALL:
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)


class BarrierOption:
    def __init__(
        self,
        strike: float,
        maturity: float,
        barrier: float,
        direction: BarrierDirection = BarrierDirection.UP_AND_OUT,
        option_type: OptionType = OptionType.CALL,
        use_brownian_bridge: bool = True,
    ) -> None:
        if strike <= 0.0:
            raise ValueError("strike must be positive.")
        if maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        if barrier <= 0.0:
            raise ValueError("barrier must be positive.")
        self.strike = strike
        self.maturity = maturity
        self.barrier = barrier
        self.direction = direction
        self.option_type = option_type
        self.use_brownian_bridge = use_brownian_bridge

    def payoff(self, spot: np.ndarray) -> np.ndarray:
        if self.option_type == OptionType.CALL:
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)
