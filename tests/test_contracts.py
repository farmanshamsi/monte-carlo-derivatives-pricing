import pytest

from monte_carlo_pricing.contracts import CashOrNothingCall, EuropeanCall, Market


def test_invalid_market_and_contract_inputs_fail_fast() -> None:
    with pytest.raises(ValueError):
        Market(spot=0, rate=0.05, volatility=0.2)
    with pytest.raises(ValueError):
        Market(spot=100, rate=0.05, volatility=-0.1)
    with pytest.raises(ValueError):
        EuropeanCall(strike=-1, maturity=1)
    with pytest.raises(ValueError):
        CashOrNothingCall(strike=100, maturity=1, payout=0)
