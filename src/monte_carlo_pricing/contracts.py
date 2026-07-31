"""Market inputs and European payoff definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


@dataclass(frozen=True)
class Market:
    spot: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.volatility < 0:
            raise ValueError("volatility cannot be negative")


@dataclass(frozen=True)
class EuropeanCall:
    strike: float
    maturity: float

    def __post_init__(self) -> None:
        _validate_contract(self.strike, self.maturity)

    def payoff(self, terminal_price: np.ndarray) -> np.ndarray:
        return np.maximum(terminal_price - self.strike, 0.0)


@dataclass(frozen=True)
class EuropeanPut:
    strike: float
    maturity: float

    def __post_init__(self) -> None:
        _validate_contract(self.strike, self.maturity)

    def payoff(self, terminal_price: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike - terminal_price, 0.0)


@dataclass(frozen=True)
class CashOrNothingCall:
    strike: float
    maturity: float
    payout: float = 1.0

    def __post_init__(self) -> None:
        _validate_contract(self.strike, self.maturity)
        if self.payout <= 0:
            raise ValueError("payout must be positive")

    def payoff(self, terminal_price: np.ndarray) -> np.ndarray:
        return self.payout * (terminal_price > self.strike)


Contract: TypeAlias = EuropeanCall | EuropeanPut | CashOrNothingCall


def _validate_contract(strike: float, maturity: float) -> None:
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity <= 0:
        raise ValueError("maturity must be positive")


def contract_name(contract: Contract) -> str:
    if isinstance(contract, EuropeanCall):
        return "European call"
    if isinstance(contract, EuropeanPut):
        return "European put"
    return "Cash-or-nothing call"
