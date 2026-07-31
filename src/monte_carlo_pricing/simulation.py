"""Monte Carlo estimators under risk-neutral geometric Brownian motion."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .analytical import analytical_price
from .contracts import Contract, Market

SimulationMethod = Literal["exact", "euler", "milstein"]
VarianceReduction = Literal["none", "antithetic", "control", "antithetic_control"]


@dataclass(frozen=True)
class Estimate:
    price: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    analytical: float
    absolute_error: float
    z_score: float
    raw_paths: int
    independent_samples: int
    method: str
    variance_reduction: str
    control_beta: float | None = None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


def price_option(
    contract: Contract,
    market: Market,
    *,
    paths: int = 100_000,
    steps: int = 100,
    method: SimulationMethod = "exact",
    variance_reduction: VarianceReduction = "none",
    seed: int = 42,
) -> Estimate:
    """Price a European payoff and report sampling uncertainty.

    Antithetic estimates treat each plus/minus payoff average as one independent
    sample. The control variate uses discounted terminal stock value, whose
    risk-neutral expectation is known exactly.
    """

    _validate_request(paths, steps, method, variance_reduction)
    rng = np.random.default_rng(seed)
    antithetic = variance_reduction in {"antithetic", "antithetic_control"}

    if antithetic:
        plus, minus = _simulate_antithetic_terminals(
            market,
            contract.maturity,
            paths // 2,
            steps,
            method,
            rng,
        )
        discounted_payoff = math.exp(-market.rate * contract.maturity) * 0.5 * (
            contract.payoff(plus) + contract.payoff(minus)
        )
        discounted_stock = math.exp(-market.rate * contract.maturity) * 0.5 * (
            plus + minus
        )
    else:
        terminal = _simulate_terminals(
            market,
            contract.maturity,
            paths,
            steps,
            method,
            rng,
        )
        discounted_payoff = (
            math.exp(-market.rate * contract.maturity) * contract.payoff(terminal)
        )
        discounted_stock = math.exp(-market.rate * contract.maturity) * terminal

    control_beta: float | None = None
    if variance_reduction in {"control", "antithetic_control"}:
        control_expectation = market.spot * math.exp(
            -market.dividend_yield * contract.maturity
        )
        control_variance = float(np.var(discounted_stock, ddof=1))
        if control_variance > 0:
            covariance = float(np.cov(discounted_payoff, discounted_stock, ddof=1)[0, 1])
            control_beta = covariance / control_variance
            discounted_payoff = discounted_payoff - control_beta * (
                discounted_stock - control_expectation
            )

    independent_samples = len(discounted_payoff)
    price = float(np.mean(discounted_payoff))
    standard_error = float(np.std(discounted_payoff, ddof=1) / math.sqrt(independent_samples))
    confidence_radius = 1.959963984540054 * standard_error
    benchmark = analytical_price(contract, market)
    error = abs(price - benchmark)

    return Estimate(
        price=price,
        standard_error=standard_error,
        confidence_low=price - confidence_radius,
        confidence_high=price + confidence_radius,
        analytical=benchmark,
        absolute_error=error,
        z_score=error / standard_error if standard_error > 0 else 0.0,
        raw_paths=paths,
        independent_samples=independent_samples,
        method=method,
        variance_reduction=variance_reduction,
        control_beta=control_beta,
    )


def simulate_terminal_from_increments(
    market: Market,
    maturity: float,
    brownian_increments: np.ndarray,
    method: Literal["euler", "milstein"],
) -> np.ndarray:
    """Simulate terminal values from supplied Brownian increments.

    This public primitive makes coupled convergence experiments testable and
    ensures Euler and Milstein use independent Brownian increments at each step.
    """

    if brownian_increments.ndim != 2:
        raise ValueError("brownian_increments must have shape (steps, paths)")
    steps, paths = brownian_increments.shape
    if steps < 1 or paths < 1:
        raise ValueError("brownian_increments cannot be empty")

    dt = maturity / steps
    terminal = np.full(paths, market.spot, dtype=float)
    drift = market.rate - market.dividend_yield
    for increment in brownian_increments:
        if method == "euler":
            terminal *= 1.0 + drift * dt + market.volatility * increment
        elif method == "milstein":
            terminal *= (
                1.0
                + drift * dt
                + market.volatility * increment
                + 0.5 * market.volatility**2 * (increment**2 - dt)
            )
        else:
            raise ValueError("method must be 'euler' or 'milstein'")
    return terminal


def exact_terminal_from_brownian(
    market: Market,
    maturity: float,
    terminal_brownian_motion: np.ndarray,
) -> np.ndarray:
    drift = market.rate - market.dividend_yield
    return market.spot * np.exp(
        (drift - 0.5 * market.volatility**2) * maturity
        + market.volatility * terminal_brownian_motion
    )


def _simulate_terminals(
    market: Market,
    maturity: float,
    paths: int,
    steps: int,
    method: SimulationMethod,
    rng: np.random.Generator,
) -> np.ndarray:
    drift = market.rate - market.dividend_yield
    if method == "exact":
        normal = rng.standard_normal(paths)
        return market.spot * np.exp(
            (drift - 0.5 * market.volatility**2) * maturity
            + market.volatility * math.sqrt(maturity) * normal
        )

    dt = maturity / steps
    sqrt_dt = math.sqrt(dt)
    terminal = np.full(paths, market.spot, dtype=float)
    for _ in range(steps):
        normal = rng.standard_normal(paths)
        if method == "euler":
            terminal *= (
                1.0 + drift * dt + market.volatility * sqrt_dt * normal
            )
        else:
            terminal *= (
                1.0
                + drift * dt
                + market.volatility * sqrt_dt * normal
                + 0.5 * market.volatility**2 * dt * (normal**2 - 1.0)
            )
    return terminal


def _simulate_antithetic_terminals(
    market: Market,
    maturity: float,
    pairs: int,
    steps: int,
    method: SimulationMethod,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    drift = market.rate - market.dividend_yield
    if method == "exact":
        normal = rng.standard_normal(pairs)
        deterministic = (drift - 0.5 * market.volatility**2) * maturity
        diffusion = market.volatility * math.sqrt(maturity) * normal
        return (
            market.spot * np.exp(deterministic + diffusion),
            market.spot * np.exp(deterministic - diffusion),
        )

    dt = maturity / steps
    sqrt_dt = math.sqrt(dt)
    plus = np.full(pairs, market.spot, dtype=float)
    minus = np.full(pairs, market.spot, dtype=float)
    for _ in range(steps):
        normal = rng.standard_normal(pairs)
        correction = 0.5 * market.volatility**2 * dt * (normal**2 - 1.0)
        plus_factor = 1.0 + drift * dt + market.volatility * sqrt_dt * normal
        minus_factor = 1.0 + drift * dt - market.volatility * sqrt_dt * normal
        if method == "milstein":
            plus_factor += correction
            minus_factor += correction
        plus *= plus_factor
        minus *= minus_factor
    return plus, minus


def _validate_request(
    paths: int,
    steps: int,
    method: str,
    variance_reduction: str,
) -> None:
    if paths < 2:
        raise ValueError("paths must be at least 2")
    if steps < 1:
        raise ValueError("steps must be positive")
    if method not in {"exact", "euler", "milstein"}:
        raise ValueError("method must be exact, euler, or milstein")
    if variance_reduction not in {
        "none",
        "antithetic",
        "control",
        "antithetic_control",
    }:
        raise ValueError("unsupported variance-reduction method")
    if variance_reduction in {"antithetic", "antithetic_control"} and paths % 2:
        raise ValueError("antithetic pricing requires an even path count")
