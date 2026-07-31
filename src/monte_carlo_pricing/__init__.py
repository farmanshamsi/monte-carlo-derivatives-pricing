"""Monte Carlo pricing research package."""

from .analytical import analytical_price
from .contracts import CashOrNothingCall, EuropeanCall, EuropeanPut, Market
from .simulation import Estimate, price_option

__all__ = [
    "CashOrNothingCall",
    "Estimate",
    "EuropeanCall",
    "EuropeanPut",
    "Market",
    "analytical_price",
    "price_option",
]
__version__ = "1.0.0"
