"""Public package exports for notebook-friendly imports.

This module exposes the main simulation and product classes at package level so
that notebooks can simply do::

    from stationary_heston_model import HestonPathSimulator, EuropeanOption
"""

from __future__ import annotations

import sys

from . import diffusions as _diffusions

# Preserve compatibility with legacy absolute imports used inside product files.
sys.modules.setdefault("diffusions", _diffusions)

from .diffusions import CIRStationarySimulator, HestonPathSimulator, InitialVarianceStrategy
from . import products as _products

# Preserve compatibility with legacy absolute imports used by entry-point code.
sys.modules.setdefault("products", _products)

from .products import (
    BarrierDirection,
    BarrierOption,
    BermudanOption,
    EuropeanOption,
    OptionType,
)

__all__ = [
    "BarrierDirection",
    "BarrierOption",
    "BermudanOption",
    "CIRStationarySimulator",
    "EuropeanOption",
    "HestonPathSimulator",
    "InitialVarianceStrategy",
    "OptionType",
]
