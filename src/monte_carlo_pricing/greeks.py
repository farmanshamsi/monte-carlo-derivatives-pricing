"""Pathwise Monte Carlo Greeks for a European call."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .analytical import analytical_call_greeks
from .contracts import EuropeanCall, Market
from .simulation import exact_terminal_from_brownian


def pathwise_call_greeks(
    contract: EuropeanCall,
    market: Market,
    *,
    paths: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate Delta, Vega, and Rho from pathwise payoff derivatives."""

    rng = np.random.default_rng(seed)
    normal = rng.standard_normal(paths)
    brownian_terminal = math.sqrt(contract.maturity) * normal
    terminal = exact_terminal_from_brownian(
        market,
        contract.maturity,
        brownian_terminal,
    )
    in_the_money = terminal > contract.strike
    discount = math.exp(-market.rate * contract.maturity)

    samples = {
        "delta": discount * in_the_money * terminal / market.spot,
        "vega": (
            discount
            * in_the_money
            * terminal
            * (brownian_terminal - market.volatility * contract.maturity)
        ),
        "rho": (
            contract.maturity
            * contract.strike
            * discount
            * in_the_money
        ),
    }
    benchmark = analytical_call_greeks(contract, market)
    analytical_values = {
        "delta": benchmark.delta,
        "vega": benchmark.vega,
        "rho": benchmark.rho,
    }

    rows = []
    for name, sample in samples.items():
        estimate = float(np.mean(sample))
        standard_error = float(np.std(sample, ddof=1) / math.sqrt(paths))
        analytical_value = analytical_values[name]
        rows.append(
            {
                "greek": name,
                "mc_estimate": estimate,
                "standard_error": standard_error,
                "analytical": analytical_value,
                "absolute_error": abs(estimate - analytical_value),
                "z_score": (
                    abs(estimate - analytical_value) / standard_error
                    if standard_error > 0
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)
