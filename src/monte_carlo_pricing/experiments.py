"""Reproducible numerical experiments and publication-ready artifacts."""

from __future__ import annotations

import json
import math
import platform
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analytical import analytical_price
from .contracts import (
    CashOrNothingCall,
    Contract,
    EuropeanCall,
    EuropeanPut,
    Market,
    contract_name,
)
from .greeks import pathwise_call_greeks
from .simulation import (
    exact_terminal_from_brownian,
    price_option,
    simulate_terminal_from_increments,
)


def run_experiments(
    output_dir: str | Path,
    *,
    paths: int = 100_000,
    steps: int = 100,
    seed: int = 42,
) -> dict[str, object]:
    """Run the full benchmark, efficiency, convergence, and Greeks study."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    market = Market(spot=100.0, rate=0.05, volatility=0.20)
    call = EuropeanCall(strike=100.0, maturity=1.0)
    put = EuropeanPut(strike=100.0, maturity=1.0)
    binary = CashOrNothingCall(strike=100.0, maturity=1.0, payout=1.0)

    baseline = _baseline_experiment(market, [call, put, binary], paths, steps, seed)
    variance = _variance_experiment(market, [call, binary], paths, seed)
    convergence, convergence_orders = _coupled_convergence(
        market,
        call,
        paths=min(paths, 40_000),
        seed=seed + 10,
    )
    path_convergence, path_slope = _path_convergence(
        market,
        call,
        binary,
        maximum_paths=paths,
        seed=seed + 20,
    )
    sensitivity = _sensitivity_experiment(
        market,
        call,
        binary,
        paths=min(paths, 50_000),
        seed=seed + 30,
    )
    greeks = pathwise_call_greeks(call, market, paths=paths, seed=seed + 40)

    baseline.to_csv(output_path / "baseline_pricing.csv", index=False)
    variance.to_csv(output_path / "variance_reduction.csv", index=False)
    convergence.to_csv(output_path / "discretization_convergence.csv", index=False)
    path_convergence.to_csv(output_path / "path_convergence.csv", index=False)
    sensitivity.to_csv(output_path / "sensitivity_analysis.csv", index=False)
    greeks.to_csv(output_path / "call_greeks.csv", index=False)

    _plot_baseline(baseline, output_path / "baseline_benchmarks.png")
    _plot_variance(variance, output_path / "variance_reduction.png")
    _plot_discretization(
        convergence,
        convergence_orders,
        output_path / "discretization_convergence.png",
    )
    _plot_path_convergence(path_convergence, path_slope, output_path / "path_convergence.png")
    _plot_sensitivity(sensitivity, output_path / "sensitivity_analysis.png")

    variance_summary = {}
    for contract_label in variance["contract"].unique():
        rows = variance[variance["contract"] == contract_label].set_index(
            "variance_reduction"
        )
        variance_summary[contract_label] = {
            strategy: float(rows.loc[strategy, "se_reduction_percent"])
            for strategy in rows.index
            if strategy != "none"
        }

    summary: dict[str, object] = {
        "configuration": {
            "paths": paths,
            "steps": steps,
            "seed": seed,
            "spot": market.spot,
            "strike": call.strike,
            "maturity": call.maturity,
            "rate": market.rate,
            "volatility": market.volatility,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
        },
        "analytical_benchmarks": {
            "European call": analytical_price(call, market),
            "European put": analytical_price(put, market),
            "Cash-or-nothing call": analytical_price(binary, market),
        },
        "baseline": baseline.to_dict(orient="records"),
        "variance_reduction_percent": variance_summary,
        "strong_convergence_order": convergence_orders,
        "path_count_standard_error_slope": path_slope,
        "call_greeks": greeks.to_dict(orient="records"),
    }
    with (output_path / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _baseline_experiment(
    market: Market,
    contracts: list[Contract],
    paths: int,
    steps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for contract in contracts:
        name = contract_name(contract)
        for method in ("exact", "euler", "milstein"):
            estimate = price_option(
                contract,
                market,
                paths=paths,
                steps=steps,
                method=method,
                seed=seed,
            )
            rows.append({"contract": name, **estimate.as_dict()})
    return pd.DataFrame(rows)


def _variance_experiment(
    market: Market,
    contracts: list[Contract],
    paths: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for contract in contracts:
        name = contract_name(contract)
        estimates = {}
        for strategy in ("none", "antithetic", "control", "antithetic_control"):
            estimates[strategy] = price_option(
                contract,
                market,
                paths=paths,
                method="exact",
                variance_reduction=strategy,
                seed=seed,
            )
        plain_error = estimates["none"].standard_error
        for strategy, estimate in estimates.items():
            rows.append(
                {
                    "contract": name,
                    **estimate.as_dict(),
                    "se_reduction_percent": (
                        100.0 * (1.0 - estimate.standard_error / plain_error)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _coupled_convergence(
    market: Market,
    contract: EuropeanCall,
    *,
    paths: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    maximum_steps = 128
    step_counts = (4, 8, 16, 32, 64, 128)
    rng = np.random.default_rng(seed)
    fine_increments = (
        math.sqrt(contract.maturity / maximum_steps)
        * rng.standard_normal((maximum_steps, paths))
    )
    terminal_brownian = fine_increments.sum(axis=0)
    exact_terminal = exact_terminal_from_brownian(
        market,
        contract.maturity,
        terminal_brownian,
    )
    discount = math.exp(-market.rate * contract.maturity)
    benchmark = analytical_price(contract, market)

    rows = []
    for count in step_counts:
        block = maximum_steps // count
        increments = fine_increments.reshape(count, block, paths).sum(axis=1)
        euler = simulate_terminal_from_increments(
            market,
            contract.maturity,
            increments,
            "euler",
        )
        milstein = simulate_terminal_from_increments(
            market,
            contract.maturity,
            increments,
            "milstein",
        )
        rows.append(
            {
                "steps": count,
                "dt": contract.maturity / count,
                "euler_terminal_rmse": float(np.sqrt(np.mean((euler - exact_terminal) ** 2))),
                "milstein_terminal_rmse": float(
                    np.sqrt(np.mean((milstein - exact_terminal) ** 2))
                ),
                "euler_call_price": float(discount * np.mean(contract.payoff(euler))),
                "milstein_call_price": float(discount * np.mean(contract.payoff(milstein))),
                "analytical_call_price": benchmark,
            }
        )
    table = pd.DataFrame(rows)
    table["euler_call_abs_error"] = (
        table["euler_call_price"] - table["analytical_call_price"]
    ).abs()
    table["milstein_call_abs_error"] = (
        table["milstein_call_price"] - table["analytical_call_price"]
    ).abs()

    log_dt = np.log(table["dt"])
    orders = {
        "Euler-Maruyama": float(
            np.polyfit(log_dt, np.log(table["euler_terminal_rmse"]), 1)[0]
        ),
        "Milstein": float(
            np.polyfit(log_dt, np.log(table["milstein_terminal_rmse"]), 1)[0]
        ),
    }
    return table, orders


def _path_convergence(
    market: Market,
    call: EuropeanCall,
    binary: CashOrNothingCall,
    *,
    maximum_paths: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    candidates = np.asarray([1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000])
    counts = candidates[candidates <= maximum_paths]
    if len(counts) < 3:
        counts = np.asarray(
            sorted({max(100, maximum_paths // 4), max(200, maximum_paths // 2), maximum_paths})
        )

    rng = np.random.default_rng(seed)
    normal = rng.standard_normal(int(counts.max()))
    terminal = exact_terminal_from_brownian(
        market,
        call.maturity,
        math.sqrt(call.maturity) * normal,
    )
    discount = math.exp(-market.rate * call.maturity)

    rows = []
    for count in counts:
        subset = terminal[:count]
        call_samples = discount * call.payoff(subset)
        binary_samples = discount * binary.payoff(subset)
        rows.append(
            {
                "paths": int(count),
                "call_price": float(np.mean(call_samples)),
                "call_standard_error": float(
                    np.std(call_samples, ddof=1) / math.sqrt(count)
                ),
                "binary_price": float(np.mean(binary_samples)),
                "binary_standard_error": float(
                    np.std(binary_samples, ddof=1) / math.sqrt(count)
                ),
            }
        )
    table = pd.DataFrame(rows)
    slope = float(
        np.polyfit(np.log(table["paths"]), np.log(table["call_standard_error"]), 1)[0]
    )
    return table, slope


def _sensitivity_experiment(
    market: Market,
    call: EuropeanCall,
    binary: CashOrNothingCall,
    *,
    paths: int,
    seed: int,
) -> pd.DataFrame:
    grids = {
        "spot": np.linspace(80.0, 120.0, 9),
        "strike": np.linspace(80.0, 120.0, 9),
        "volatility": np.linspace(0.10, 0.40, 9),
        "rate": np.linspace(0.00, 0.10, 9),
        "maturity": np.linspace(0.25, 2.00, 8),
    }
    rows = []
    for parameter, values in grids.items():
        for value in values:
            scenario_market = market
            scenario_call = call
            scenario_binary = binary
            if parameter in {"spot", "volatility", "rate"}:
                scenario_market = replace(market, **{parameter: float(value)})
            elif parameter == "strike":
                scenario_call = replace(call, strike=float(value))
                scenario_binary = replace(binary, strike=float(value))
            else:
                scenario_call = replace(call, maturity=float(value))
                scenario_binary = replace(binary, maturity=float(value))

            for contract in (scenario_call, scenario_binary):
                estimate = price_option(
                    contract,
                    scenario_market,
                    paths=paths,
                    method="exact",
                    variance_reduction="antithetic_control",
                    seed=seed,
                )
                rows.append(
                    {
                        "parameter": parameter,
                        "value": float(value),
                        "contract": contract_name(contract),
                        "mc_price": estimate.price,
                        "analytical": estimate.analytical,
                        "standard_error": estimate.standard_error,
                        "absolute_error": estimate.absolute_error,
                    }
                )
    return pd.DataFrame(rows)


def _plot_baseline(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, contract_label in zip(axes, table["contract"].unique(), strict=True):
        subset = table[table["contract"] == contract_label]
        axis.bar(subset["method"], subset["price"], color=["#315f8c", "#d88732", "#4b9460"])
        axis.axhline(subset["analytical"].iloc[0], color="black", linestyle="--", label="Analytical")
        axis.errorbar(
            subset["method"],
            subset["price"],
            yerr=1.96 * subset["standard_error"],
            fmt="none",
            color="black",
            capsize=4,
        )
        axis.set_title(contract_label)
        axis.set_ylabel("Price")
        axis.legend()
    fig.suptitle("Monte Carlo estimates with 95% confidence intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_variance(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = {
        "none": "Plain",
        "antithetic": "Antithetic",
        "control": "Control",
        "antithetic_control": "Anti + control",
    }
    for axis, contract_label in zip(axes, table["contract"].unique(), strict=True):
        subset = table[table["contract"] == contract_label]
        axis.bar(
            [labels[value] for value in subset["variance_reduction"]],
            subset["standard_error"],
            color="#315f8c",
        )
        axis.set_title(contract_label)
        axis.set_ylabel("Estimator standard error")
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Variance-reduction efficiency at fixed path budget")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_discretization(
    table: pd.DataFrame,
    orders: dict[str, float],
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 5))
    axis.loglog(
        table["dt"],
        table["euler_terminal_rmse"],
        "o-",
        label=f"Euler (slope {orders['Euler-Maruyama']:.2f})",
    )
    axis.loglog(
        table["dt"],
        table["milstein_terminal_rmse"],
        "o-",
        label=f"Milstein (slope {orders['Milstein']:.2f})",
    )
    axis.invert_xaxis()
    axis.set(
        title="Coupled strong convergence to exact GBM",
        xlabel="Time-step size Δt",
        ylabel="Terminal-price RMSE",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_path_convergence(table: pd.DataFrame, slope: float, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 5))
    axis.loglog(
        table["paths"],
        table["call_standard_error"],
        "o-",
        label=f"European call (slope {slope:.2f})",
    )
    axis.loglog(
        table["paths"],
        table["binary_standard_error"],
        "o-",
        label="Cash-or-nothing call",
    )
    axis.set(
        title="Monte Carlo sampling convergence",
        xlabel="Number of paths",
        ylabel="Estimator standard error",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_sensitivity(table: pd.DataFrame, path: Path) -> None:
    parameters = ["spot", "strike", "volatility", "rate", "maturity"]
    titles = {
        "spot": "Spot",
        "strike": "Strike",
        "volatility": "Volatility",
        "rate": "Interest rate",
        "maturity": "Maturity",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, parameter in zip(axes.flat, parameters, strict=False):
        subset = table[
            (table["parameter"] == parameter)
            & (table["contract"] == "European call")
        ]
        axis.plot(subset["value"], subset["analytical"], "--", color="black", label="Analytical")
        axis.plot(subset["value"], subset["mc_price"], "o", color="#315f8c", label="Monte Carlo")
        axis.set(title=titles[parameter], xlabel=parameter, ylabel="Call price")
    axes.flat[-1].axis("off")
    axes.flat[0].legend()
    fig.suptitle("One-factor sensitivity with common random numbers")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
