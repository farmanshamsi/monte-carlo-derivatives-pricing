import math

import numpy as np

from monte_carlo_pricing.contracts import Market
from monte_carlo_pricing.simulation import (
    exact_terminal_from_brownian,
    simulate_terminal_from_increments,
)


def test_milstein_has_lower_coupled_terminal_error_than_euler() -> None:
    market = Market(spot=100, rate=0.05, volatility=0.20)
    maturity = 1.0
    steps = 16
    paths = 20_000
    rng = np.random.default_rng(123)
    increments = math.sqrt(maturity / steps) * rng.standard_normal((steps, paths))
    exact = exact_terminal_from_brownian(market, maturity, increments.sum(axis=0))

    euler = simulate_terminal_from_increments(market, maturity, increments, "euler")
    milstein = simulate_terminal_from_increments(
        market,
        maturity,
        increments,
        "milstein",
    )

    euler_rmse = np.sqrt(np.mean((euler - exact) ** 2))
    milstein_rmse = np.sqrt(np.mean((milstein - exact) ** 2))
    assert milstein_rmse < 0.35 * euler_rmse
