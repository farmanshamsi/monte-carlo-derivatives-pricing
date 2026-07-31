"""Black-Scholes benchmarks and analytical call Greeks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .contracts import CashOrNothingCall, Contract, EuropeanCall, EuropeanPut, Market


@dataclass(frozen=True)
class CallGreeks:
    delta: float
    vega: float
    rho: float


def _d1_d2(market: Market, strike: float, maturity: float) -> tuple[float, float]:
    volatility_time = market.volatility * math.sqrt(maturity)
    d1 = (
        math.log(market.spot / strike)
        + (
            market.rate
            - market.dividend_yield
            + 0.5 * market.volatility**2
        )
        * maturity
    ) / volatility_time
    return d1, d1 - volatility_time


def analytical_price(contract: Contract, market: Market) -> float:
    """Return the Black-Scholes price for a supported European contract."""

    if market.volatility == 0:
        terminal = market.spot * math.exp(
            (market.rate - market.dividend_yield) * contract.maturity
        )
        return math.exp(-market.rate * contract.maturity) * float(
            contract.payoff(np.asarray([terminal], dtype=float))[0]
        )

    d1, d2 = _d1_d2(market, contract.strike, contract.maturity)
    spot_leg = market.spot * math.exp(-market.dividend_yield * contract.maturity)
    strike_discount = contract.strike * math.exp(-market.rate * contract.maturity)

    if isinstance(contract, EuropeanCall):
        return spot_leg * norm.cdf(d1) - strike_discount * norm.cdf(d2)
    if isinstance(contract, EuropeanPut):
        return strike_discount * norm.cdf(-d2) - spot_leg * norm.cdf(-d1)
    if isinstance(contract, CashOrNothingCall):
        return contract.payout * math.exp(-market.rate * contract.maturity) * norm.cdf(d2)
    raise TypeError(f"Unsupported contract type: {type(contract).__name__}")


def analytical_call_greeks(contract: EuropeanCall, market: Market) -> CallGreeks:
    if market.volatility == 0:
        raise ValueError("analytical Greeks require positive volatility")

    d1, d2 = _d1_d2(market, contract.strike, contract.maturity)
    discount_dividend = math.exp(-market.dividend_yield * contract.maturity)
    discount_rate = math.exp(-market.rate * contract.maturity)
    return CallGreeks(
        delta=discount_dividend * norm.cdf(d1),
        vega=(
            market.spot
            * discount_dividend
            * norm.pdf(d1)
            * math.sqrt(contract.maturity)
        ),
        rho=contract.strike * contract.maturity * discount_rate * norm.cdf(d2),
    )
