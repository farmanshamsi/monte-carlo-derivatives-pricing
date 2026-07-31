from monte_carlo_pricing.contracts import EuropeanCall, Market
from monte_carlo_pricing.greeks import pathwise_call_greeks


def test_pathwise_greeks_match_analytical_values() -> None:
    market = Market(spot=100, rate=0.05, volatility=0.20)
    table = pathwise_call_greeks(
        EuropeanCall(strike=100, maturity=1),
        market,
        paths=200_000,
        seed=42,
    )

    assert (table["z_score"] < 4).all()
