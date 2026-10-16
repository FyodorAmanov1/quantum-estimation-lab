"""CPU reference experiments for European call estimation; no hardware speedup claim."""
from __future__ import annotations

import argparse
import json
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np


def black_scholes_call(spot=100.0, strike=100.0, rate=0.03, volatility=0.2, maturity=1.0):
    if spot <= 0 or strike <= 0 or volatility < 0 or maturity < 0:
        raise ValueError("Positive spot/strike and nonnegative volatility/maturity required")
    if maturity == 0 or volatility == 0:
        return max(spot - strike * exp(-rate * maturity), 0.0)
    d1 = (log(spot / strike) + (rate + volatility**2 / 2) * maturity) / (volatility * sqrt(maturity))
    d2 = d1 - volatility * sqrt(maturity)
    normal = lambda x: (1 + erf(x / sqrt(2))) / 2
    return spot * normal(d1) - strike * exp(-rate * maturity) * normal(d2)


def discounted_payoffs(normals, spot=100.0, strike=100.0, rate=0.03, volatility=0.2, maturity=1.0):
    if spot <= 0 or strike <= 0 or volatility < 0 or maturity < 0:
        raise ValueError("Invalid option parameters")
    terminal = spot * np.exp((rate - 0.5 * volatility**2) * maturity + volatility * sqrt(maturity) * np.asarray(normals))
    return exp(-rate * maturity) * np.maximum(terminal - strike, 0)


def monte_carlo(paths=8192, seed=1729, antithetic=False):
    if paths < 2 or (antithetic and paths % 2):
        raise ValueError("At least two paths required; antithetic path counts must be even")
    rng = np.random.default_rng(seed)
    if antithetic:
        z = rng.standard_normal(paths // 2)
        # Independent observations are paired means, not the correlated individual payoffs.
        observations = (discounted_payoffs(z) + discounted_payoffs(-z)) / 2
    else:
        observations = discounted_payoffs(rng.standard_normal(paths))
    if len(observations) < 2:
        raise ValueError("At least two independent observations required")
    estimate = float(np.mean(observations))
    se = float(np.std(observations, ddof=1) / sqrt(len(observations)))
    return {"estimate": estimate, "standard_error": se,
            "ci95": [estimate - 1.96 * se, estimate + 1.96 * se],
            "payoff_evaluations": paths, "independent_observations": len(observations)}


def experiment():
    reference = black_scholes_call()
    plain, anti = monte_carlo(), monte_carlo(antithetic=True)
    return {"project": "quantum-estimation-lab", "seed": 1729,
            "metrics": {"analytic_price": reference, "monte_carlo": plain, "antithetic": anti},
            "provenance": {"data": "Synthetic GBM terminal draws; risk-neutral European call", "numpy": np.__version__},
            "limitations": ["Normal approximation confidence intervals, not finite-sample guarantees.",
                            "Classical baseline only; quantum methods and full oracle costs are later milestones."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(experiment(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result)
