from __future__ import annotations
from enum import Enum

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

