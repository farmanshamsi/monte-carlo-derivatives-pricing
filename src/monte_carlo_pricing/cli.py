"""Command-line entry point for the full numerical study."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiments import run_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo derivative-pricing experiments."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    summary = run_experiments(
        arguments.output_dir,
        paths=arguments.paths,
        steps=arguments.steps,
        seed=arguments.seed,
    )
    call_rows = [
        row
        for row in summary["baseline"]
        if row["contract"] == "European call" and row["method"] == "exact"
    ]
    call = call_rows[0]
    print("Monte Carlo research run complete")
    print(f"European call: {call['price']:.6f} ± {1.96 * call['standard_error']:.6f}")
    print(f"Analytical benchmark: {call['analytical']:.6f}")
    print(f"Artifacts: {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
