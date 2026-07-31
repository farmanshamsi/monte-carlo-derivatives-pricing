import math

import pytest

from monte_carlo_pricing.analytical import analytical_price
from monte_carlo_pricing.contracts import (
    CashOrNothingCall,
    EuropeanCall,
    EuropeanPut,
    Market,
)


def test_black_scholes_reference_values() -> None:
    market = Market(spot=100, rate=0.05, volatility=0.20)

    assert analytical_price(EuropeanCall(100, 1), market) == pytest.approx(
        10.450583572185565
    )
    assert analytical_price(EuropeanPut(100, 1), market) == pytest.approx(
        5.573526022256971
    )
    assert analytical_price(CashOrNothingCall(100, 1), market) == pytest.approx(
        0.5323248154537634
    )


def test_put_call_parity() -> None:
    market = Market(spot=100, rate=0.05, volatility=0.20)
    call = analytical_price(EuropeanCall(105, 1.5), market)
    put = analytical_price(EuropeanPut(105, 1.5), market)

    parity = market.spot - 105 * math.exp(-market.rate * 1.5)
    assert call - put == pytest.approx(parity)
