import pytest

from monte_carlo_pricing.contracts import CashOrNothingCall, EuropeanCall, Market
from monte_carlo_pricing.simulation import price_option


@pytest.fixture
def market() -> Market:
    return Market(spot=100, rate=0.05, volatility=0.20)


def test_exact_monte_carlo_contains_benchmark_within_four_standard_errors(
    market: Market,
) -> None:
    for contract in (EuropeanCall(100, 1), CashOrNothingCall(100, 1)):
        estimate = price_option(contract, market, paths=100_000, seed=42)
        assert estimate.z_score < 4


def test_fixed_seed_is_reproducible(market: Market) -> None:
    contract = EuropeanCall(100, 1)
    first = price_option(contract, market, paths=20_000, seed=7)
    second = price_option(contract, market, paths=20_000, seed=7)

    assert first.price == second.price
    assert first.standard_error == second.standard_error


def test_variance_reduction_improves_efficiency(market: Market) -> None:
    call = EuropeanCall(100, 1)
    binary = CashOrNothingCall(100, 1)

    plain_call = price_option(call, market, paths=100_000, seed=42)
    efficient_call = price_option(
        call,
        market,
        paths=100_000,
        variance_reduction="antithetic_control",
        seed=42,
    )
    plain_binary = price_option(binary, market, paths=100_000, seed=42)
    antithetic_binary = price_option(
        binary,
        market,
        paths=100_000,
        variance_reduction="antithetic",
        seed=42,
    )

    assert efficient_call.standard_error < 0.5 * plain_call.standard_error
    assert antithetic_binary.standard_error < plain_binary.standard_error
