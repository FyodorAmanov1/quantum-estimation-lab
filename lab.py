"""CPU reference experiments for European call estimation; no hardware speedup claim."""
from __future__ import annotations

import argparse
import json
from math import erf, exp, log, sqrt
from numbers import Integral
from pathlib import Path
import platform

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


def convergence_experiment(path_budgets=(256, 1024, 4096, 16384), n_seeds=128, seed=1729):
    """Compare estimators at equal payoff cost using fresh streams in every cell.

    Each reported seed can be replayed with monte_carlo. Budgets are not nested,
    and the two methods do not share normal draws. Independence is the practical
    pseudorandom-stream guarantee of NumPy SeedSequence, not mathematical proof.
    """
    budgets = tuple(path_budgets)
    if not budgets or any(isinstance(n, bool) or not isinstance(n, Integral)
                          or n < 4 or n % 2 for n in budgets):
        raise ValueError("Path budgets must be even integers of at least four")
    if len(set(budgets)) != len(budgets):
        raise ValueError("Path budgets must be distinct")
    if isinstance(n_seeds, bool) or not isinstance(n_seeds, Integral) or n_seeds < 2:
        raise ValueError("At least two independent seeds required")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Root seed must be a nonnegative integer")
    reference = black_scholes_call()
    children = iter(np.random.SeedSequence(int(seed)).spawn(len(budgets) * 2 * int(n_seeds)))
    rows = []
    for paths in budgets:
        methods = {}
        for method, antithetic in (("plain", False), ("paired_antithetic", True)):
            runs = []
            for replicate in range(int(n_seeds)):
                child = next(children)
                run_seed = int(child.generate_state(1, dtype=np.uint64)[0])
                run = monte_carlo(paths=int(paths), seed=run_seed, antithetic=antithetic)
                run.update(seed=run_seed, replicate=replicate,
                           error=run["estimate"] - reference,
                           covers_analytic=bool(run["ci95"][0] <= reference <= run["ci95"][1]))
                runs.append(run)
            errors = np.array([run["error"] for run in runs])
            covered = sum(run["covers_analytic"] for run in runs)
            coverage = covered / int(n_seeds)
            methods[method] = {
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "bias": float(np.mean(errors)),
                "empirical_coverage95": coverage,
                "covered_count": covered,
                "coverage_mc_standard_error": sqrt(coverage * (1 - coverage) / int(n_seeds)),
                "mean_standard_error": float(np.mean([r["standard_error"] for r in runs])),
                "mean_ci95_width": float(np.mean([r["ci95"][1] - r["ci95"][0] for r in runs])),
                "runs": runs,
            }
        rows.append({"paths": int(paths), "methods": methods})
    return {
        "analytic_price": reference, "root_seed": int(seed), "n_seeds": int(n_seeds),
        "path_budgets": [int(n) for n in budgets],
        "seed_protocol": "SeedSequence(root).spawn(budgets * 2 * n_seeds); budget/method/replicate order; uint64 child seeds",
        "stream_design": "Fresh independent pseudorandom streams for each budget, method, and replicate; no shared draws",
        "nominal_coverage": 0.95,
        "interval_method": "estimate +/- 1.96 * sample standard error; paired means for antithetic",
        "cost_model": "Equal discounted-payoff evaluations: N plain normals versus N/2 antithetic normals and N/2 independent pairs; excludes runtime and RNG costs",
        "option_parameters": {"spot": 100.0, "strike": 100.0, "rate": 0.03,
                              "volatility": 0.2, "maturity": 1.0},
        "rows": rows,
    }


def experiment(price_csv=None, as_of=None):
    from src.studio import calibrated_comparison
    import scipy
    reference = black_scholes_call()
    plain, anti = monte_carlo(), monte_carlo(antithetic=True)
    return {"project": "quantum-estimation-lab", "seed": 1729,
            "metrics": {"analytic_price": reference, "monte_carlo": plain, "antithetic": anti,
                        "convergence": convergence_experiment(),
                        "portfolio_calibration": calibrated_comparison(price_csv, as_of)},
            "provenance": {"data": "Call benchmark: synthetic risk-neutral GBM; portfolio comparison records its own dated CSV provenance",
                           "scipy": scipy.__version__,
                           "numpy": np.__version__, "python": platform.python_version(),
                           "bit_generator": type(np.random.default_rng(0).bit_generator).__name__},
            "limitations": ["Normal approximation confidence intervals, not finite-sample guarantees.",
                            "Coverage and RMSE have sampling uncertainty across a finite set of seeds; no calibration or monotonic convergence guarantee.",
                            "One synthetic option parameter setting; equal payoff counts do not establish a runtime speedup.",
                            "The call benchmark is classical; the studio's separate discrete quantum example does not estimate portfolio tail risk.",
                            "CSV calibration uses constant historical covariance with risk-neutral drift; it does not establish real-world tail accuracy."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--price-csv", type=Path)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    if args.price_csv is not None and args.as_of is None:
        parser.error("--price-csv requires --as-of YYYY-MM-DD")
    result = json.dumps(experiment(args.price_csv, args.as_of), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8", newline="\n")
    else:
        print(result)
