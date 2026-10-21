"""Multi-asset derivative risk, variance reduction and an explicit amplitude-estimation oracle."""

from __future__ import annotations
import csv
import argparse
import hashlib
from datetime import date
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri, xlogy
from scipy.stats import qmc

try:
    from .common import finalize, metric
except ImportError:
    from common import finalize, metric


def option_value(spot, strike, volatility, time, rate=0.03, kind="call"):
    spot = np.asarray(spot)
    time = max(float(time), 1e-12)
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * time) / (
        volatility * np.sqrt(time)
    )
    d2 = d1 - volatility * np.sqrt(time)
    call = spot * ndtr(d1) - strike * np.exp(-rate * time) * ndtr(d2)
    return call if kind == "call" else call - spot + strike * np.exp(-rate * time)


def calibrate_prices(path, as_of=None):
    """Optional user CSV: date followed by positive closing-price columns. No network or secrets."""
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if (
        len(rows) < 61
        or len(rows[0]) < 2
        or any(len(row) != len(rows[0]) for row in rows)
    ):
        raise ValueError(
            "CSV requires a date header and at least 60 complete price rows"
        )
    names = rows[0][1:]
    if rows[0][0] != "date":
        raise ValueError("First CSV column must be date")
    if any(not name.strip() or name != name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Asset names must be nonempty and unique")
    dates = [row[0] for row in rows[1:]]
    if any(date.fromisoformat(d).isoformat() != d for d in dates):
        raise ValueError("Use ISO dates: YYYY-MM-DD")
    prices = np.array([[float(x) for x in row[1:]] for row in rows[1:]])
    if (
        len(prices) < 60
        or prices.shape[1] != len(names)
        or np.any(prices <= 0)
        or not np.isfinite(prices).all()
    ):
        raise ValueError("CSV requires at least 60 complete dates of positive prices")
    if dates != sorted(set(dates)):
        raise ValueError("Dates must be unique and in ascending ISO order")
    if as_of is None:
        as_of = dates[-1]
    if not isinstance(as_of, str) or date.fromisoformat(as_of).isoformat() != as_of:
        raise ValueError("Use an ISO as-of date: YYYY-MM-DD")
    selected = [i for i, d in enumerate(dates) if d <= as_of]
    if len(selected) < 60:
        raise ValueError("At least 60 price rows on or before as-of required")
    prices = prices[selected]
    dates = [dates[i] for i in selected]
    returns = np.diff(np.log(prices), axis=0)
    covariance = np.cov(returns, rowvar=False) * 252
    covariance = np.atleast_2d(covariance)
    covariance = 0.9 * covariance + 0.1 * np.diag(np.diag(covariance))
    volatility = np.sqrt(np.diag(covariance))
    if np.any(volatility <= 1e-12):
        raise ValueError("Each asset needs nonzero return variance")
    correlation = covariance / np.outer(volatility, volatility)
    return {
        "assets": names,
        "spot": prices[-1].tolist(),
        "volatility": volatility.tolist(),
        "correlation": correlation.tolist(),
        "rows": len(prices),
        "as_of": as_of,
        "window_start": dates[0],
        "window_end": dates[-1],
        "return_observations": len(prices) - 1,
        "annualization": 252,
        "diagonal_shrinkage": 0.1,
        "input_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def frozen_parameters():
    """Versioned illustrative assumptions; the date is a scenario label, not a market quote."""
    return {
        "assets": ["A", "B", "C", "D"], "as_of": "2025-12-31",
        "spot": [100.0, 80.0, 120.0, 60.0],
        "volatility": [0.22, 0.28, 0.18, 0.35],
        "correlation": (0.45 ** np.abs(np.arange(4)[:, None] - np.arange(4)[None, :])).tolist(),
    }


def _book_parameters(parameters):
    """Map by asset name, never silently by CSV column order."""
    names = parameters["assets"]
    if len(names) != 4 or set(names) != {"A", "B", "C", "D"}:
        raise ValueError("This option book requires exactly assets A, B, C, D")
    order = [names.index(name) for name in ["A", "B", "C", "D"]]
    spot = np.asarray(parameters["spot"], dtype=float)
    vol = np.asarray(parameters["volatility"], dtype=float)
    corr = np.asarray(parameters["correlation"], dtype=float)
    if (spot.shape != (4,) or vol.shape != (4,) or corr.shape != (4, 4)
        or not all(np.isfinite(x).all() for x in (spot, vol, corr))
        or np.any(spot <= 0) or np.any(vol <= 0)
        or not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1)):
        raise ValueError("Invalid spots, volatilities or correlation matrix")
    corr = corr[np.ix_(order, order)]
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Correlation must be positive definite") from exc
    return spot[order], vol[order], corr


def portfolio_samples(
    paths=4096, seed=1729, method="sobol", horizon=21 / 252, stress=1,
    parameters=None,
):
    if type(paths) is not int or paths < 2 or not 0 < horizon < 0.5 or not np.isfinite(stress) or stress <= 0:
        raise ValueError(
            "Use at least two paths, a horizon within the option maturity, and positive stress"
        )
    if method not in {"plain", "antithetic", "sobol"}:
        raise ValueError("Unknown sampling method")
    n = 4
    spot, vol, correlation = _book_parameters(frozen_parameters() if parameters is None else parameters)
    vol = vol * stress
    units = np.array([50.0, -30.0, 40.0, 20.0])
    strikes = np.array([100.0, 85.0, 115.0, 65.0])
    kinds = ["call", "put", "call", "put"]
    factor = np.linalg.cholesky(correlation)
    if method == "sobol":
        u = qmc.Sobol(n, scramble=True, seed=seed).random_base2(int(np.log2(paths)))
        if len(u) != paths:
            raise ValueError("Sobol path budget must be a power of two")
        z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
    elif method == "antithetic":
        if paths % 2:
            raise ValueError("Antithetic path budget must be even")
        base = np.random.default_rng(seed).standard_normal((paths // 2, n))
        z = np.concatenate([base, -base])
    else:
        z = np.random.default_rng(seed).standard_normal((paths, n))
    terminal = spot * np.exp(
        (0.03 - 0.5 * vol**2) * horizon + np.sqrt(horizon) * (z @ factor.T) * vol
    )
    initial = sum(
        units[j] * option_value(spot[j], strikes[j], vol[j], 0.5, kind=kinds[j])
        for j in range(n)
    )
    future = sum(
        units[j]
        * option_value(terminal[:, j], strikes[j], vol[j], 0.5 - horizon, kind=kinds[j])
        for j in range(n)
    )
    losses = float(initial) - np.exp(-0.03 * horizon) * future
    return losses, np.maximum(future, 0), float(initial)


def risk_summary(losses, exposure, confidence=0.99):
    threshold = float(np.quantile(losses, confidence))
    tail = losses[losses >= threshold]
    return {
        "var": threshold,
        "es": float(tail.mean()),
        "pfe": float(np.quantile(exposure, confidence)),
        "mean_loss": float(np.mean(losses)),
    }


def amplitude_oracle(probabilities, payoff):
    """Householder preparation of scenario register + payoff ancilla; exact CPU matrices."""
    probabilities = np.asarray(probabilities, float)
    payoff = np.asarray(payoff, float)
    if len(probabilities) & (len(probabilities) - 1) or len(probabilities) != len(
        payoff
    ):
        raise ValueError("Power-of-two scenario register required")
    if (
        not np.isclose(probabilities.sum(), 1)
        or np.any(probabilities < 0)
        or np.any((payoff < 0) | (payoff > 1))
    ):
        raise ValueError("Invalid encoding")
    state = np.zeros(2 * len(payoff))
    state[0::2] = np.sqrt(probabilities * (1 - payoff))
    state[1::2] = np.sqrt(probabilities * payoff)
    e0 = np.eye(len(state))[0]
    v = e0 - state
    A = (
        np.eye(len(state))
        if np.linalg.norm(v) < 1e-14
        else np.eye(len(state)) - 2 * np.outer(v, v) / (v @ v)
    )
    S0 = np.eye(len(state))
    S0[0, 0] = -1
    Sgood = np.diag(np.tile([1.0, -1.0], len(payoff)))
    Q = -A @ S0 @ A.T @ Sgood
    assert np.allclose(A.T @ A, np.eye(len(state)), atol=1e-12)
    assert np.allclose(Q.T @ Q, np.eye(len(state)), atol=1e-12)
    return state, Q


def quantum_estimate(state, Q, powers=(0, 1, 2, 4, 8), shots=256, seed=1729):
    rng = np.random.default_rng(seed)
    counts = []
    query_count = 0
    for power in powers:
        evolved = np.linalg.matrix_power(Q, power) @ state
        p = float(np.sum(evolved[1::2] ** 2))
        success = int(rng.binomial(shots, np.clip(p, 0, 1)))
        counts.append((power, success, shots))
        query_count += shots * (2 * power + 1)
    theta = np.linspace(1e-6, np.pi / 2 - 1e-6, 32769)
    log_likelihood = np.zeros_like(theta)
    for power, success, total in counts:
        p = np.clip(np.sin((2 * power + 1) * theta) ** 2, 1e-12, 1 - 1e-12)
        log_likelihood += xlogy(success, p) + xlogy(total - success, 1 - p)
    best = int(np.argmax(log_likelihood))
    accepted = log_likelihood >= log_likelihood[best] - 1.920729
    # A profile-likelihood set may be disconnected; its bounding interval is reported honestly.
    bounds = np.sin(theta[accepted]) ** 2
    return {
        "estimate": float(np.sin(theta[best]) ** 2),
        "interval": [float(bounds.min()), float(bounds.max())],
        "queries": query_count,
        "counts": [
            {"grover_power": p, "successes": s, "shots": n} for p, s, n in counts
        ],
    }


def calibrated_comparison(price_csv=None, as_of=None, paths=32768, seed=1729):
    """Same book and normal shocks; distinguish spot changes from vol/correlation changes."""
    bundled = price_csv is None
    if bundled:
        price_csv = Path(__file__).resolve().parents[1] / "examples" / "synthetic-prices.csv"
        as_of = as_of or "2025-12-31"
    elif as_of is None:
        raise ValueError("User CSV comparisons require an explicit as_of date")
    calibrated = calibrate_prices(price_csv, as_of=as_of)
    spot, vol, corr = _book_parameters(calibrated)
    calibrated.update(assets=["A", "B", "C", "D"], spot=spot.tolist(),
                      volatility=vol.tolist(), correlation=corr.tolist())
    frozen = frozen_parameters()
    matched = dict(frozen, spot=spot.tolist(), as_of=as_of)
    variants = [("frozen", frozen), ("spot_matched", matched), ("calibrated", calibrated)]
    rows = []
    for name, parameters in variants:
        losses, exposure, pv = portfolio_samples(paths, seed=seed, parameters=parameters)
        rows.append({"name": name, "parameters": parameters, "initial_value": pv,
                     **risk_summary(losses, exposure)})
    # Paired changes across independent Sobol scrambles describe simulation sensitivity,
    # not uncertainty in calibration or the validity of either model.
    changes = {key: [] for key in ("var", "es", "pfe")}
    for replicate in range(12):
        a = risk_summary(*portfolio_samples(paths, seed=100 + replicate, parameters=matched)[:2])
        b = risk_summary(*portfolio_samples(paths, seed=100 + replicate, parameters=calibrated)[:2])
        for key in changes:
            changes[key].append(b[key] - a[key])
    return {
        "data_source": "Generated synthetic price CSV; no market observations" if bundled else "User-supplied price CSV; provenance and adjustments are the user's responsibility",
        "as_of": as_of, "paths": paths, "seed": seed, "method": "scrambled Sobol",
        "horizon_trading_days": 21, "confidence": 0.99,
        "book": {"assets": ["A", "B", "C", "D"], "units": [50, -30, 40, 20],
                 "strikes": [100, 85, 115, 65], "kinds": ["call", "put", "call", "put"],
                 "maturity_years": 0.5, "rate": 0.03},
        "rows": rows,
        "calibrated_minus_spot_matched": {key: rows[2][key] - rows[1][key] for key in changes},
        "paired_scramble_changes": {
            key: {"runs": values, "mean": float(np.mean(values)),
                  "standard_deviation": float(np.std(values, ddof=1)),
                  "standard_error_of_mean": float(np.std(values, ddof=1) / np.sqrt(12))}
            for key, values in changes.items()},
        "scramble_seeds": list(range(100, 112)),
        "limitations": [
            "Historical close-to-close covariance is used as a constant scenario and pricing volatility proxy; it is not an implied volatility surface.",
            "The drift remains the fixed 3% rate, not an estimated physical expected return. These are model scenarios, not validated real-world tail probabilities.",
            "252 equal trading steps per year are assumed; supply consecutive synchronized adjusted daily closes. No holiday, split or dividend adjustment is inferred.",
            "Gaussian GBM discards skewness, heavy tails and volatility clustering, even when present in the input CSV. Calibration does not validate this model.",
            "Twelve scramble changes measure numerical variability only; parameter uncertainty and out-of-sample accuracy are unmeasured.",
        ],
    }


def calibration_case(comparison):
    c = comparison
    fit = c["rows"][2]["parameters"]
    labels = ["Frozen synthetic", "Spot matched", "CSV calibrated"]
    return {
        "id": "calibration", "label": "Calibrated scenarios",
        "kicker": "PRICE CSV / PARAMETER SENSITIVITY",
        "title": "How much does calibration change this book?",
        "description": f"As of {c['as_of']}; closes {fit['window_start']} to {fit['window_end']}. {c['data_source']}. Same positions, strikes, horizon and normal shocks across three parameter sets.",
        "metrics": [metric("Calibrated 99% VaR", c["rows"][2]["var"], "USD"),
                    metric("VaR change vs spot matched", c["calibrated_minus_spot_matched"]["var"], "USD"),
                    metric("Calibration observations", fit["return_observations"], "returns"),
                    metric("Scenario paths per model", c["paths"], "paths")],
        "charts": [{"title": "99% risk metrics under frozen and fitted parameters",
                    "x": labels, "xlabel": "Parameter set / USD",
                    "series": [{"name": key.upper(), "values": [r[key] for r in c["rows"]]}
                               for key in ("var", "es", "pfe")]}],
        "table": {"columns": ["Model", "As of", "Initial value / USD", "VaR / USD", "ES / USD", "PFE / USD"],
                  "rows": [[label, r["parameters"]["as_of"], r["initial_value"], r["var"], r["es"], r["pfe"]]
                           for label, r in zip(labels, c["rows"])]},
        "method": [
            "CSV assets are mapped by names A, B, C, D. Cutoff excludes later prices; at least 60 complete rows are required. Full parameter snapshots and input SHA-256 are included in the exported study.",
            "Annual covariance = 252 times sample covariance of log returns (ddof=1); 90% sample plus 10% its diagonal. This fixed shrinkage is a design choice, not an estimated optimal shrinkage coefficient.",
            "Spot-matched baseline uses CSV final spots with frozen volatilities and correlations. Its difference from calibrated scenarios isolates the joint volatility/correlation change, including repricing the initial book.",
            "All models use the same Sobol normal draws. Frozen assumptions are labeled 2025-12-31; their date is illustrative, not a historical market snapshot.",
            *[f"Calibrated minus spot-matched {key.upper()}: mean {v['mean']:.4f} USD across 12 independent scrambles (seeds 100–111); numerical standard error {v['standard_error_of_mean']:.4f} USD."
              for key, v in c["paired_scramble_changes"].items()],
        ],
        "limitations": c["limitations"],
        "reproduce": "python src/studio.py --out docs/study.json" if c["data_source"].startswith("Generated") else "python src/studio.py --price-csv YOUR.csv --as-of YYYY-MM-DD --out results/custom-study.json",
        "comparison": c,
    }


def experiment(price_csv=None, as_of=None):
    cases = []
    for stress, title in [(1, "Base market"), (1.8, "Volatility shock")]:
        losses, exposure, pv = portfolio_samples(32768, stress=stress)
        summary = risk_summary(losses, exposure)
        edges = np.linspace(np.quantile(losses, 0.002), np.quantile(losses, 0.998), 55)
        hist, _ = np.histogram(losses, edges)
        x = ((edges[:-1] + edges[1:]) / 2).tolist()
        convergence = []
        reference = portfolio_samples(131072, seed=499, stress=stress)[0].mean()
        budgets = [256, 1024, 4096, 16384]
        for method in ["plain", "antithetic", "sobol"]:
            errors = []
            for budget in budgets:
                estimates = [
                    portfolio_samples(
                        budget, seed=100 + j, method=method, stress=stress
                    )[0].mean()
                    for j in range(12)
                ]
                errors.append(
                    float(np.sqrt(np.mean((np.array(estimates) - reference) ** 2)))
                )
            convergence.append({"name": method, "values": errors})
        assert summary["es"] >= summary["var"]
        cases.append(
            {
                "id": f"risk-{stress}",
                "label": title,
                "kicker": "MULTI-ASSET DERIVATIVE RISK",
                "title": "The portfolio, under pressure.",
                "description": "Four correlated option positions. Revalue the full book at a 21-day horizon and compare three simulation designs at matched path budgets.",
                "metrics": [
                    metric("99% value at risk", summary["var"], "USD"),
                    metric("99% expected shortfall", summary["es"], "USD"),
                    metric("99% potential exposure", summary["pfe"], "USD"),
                    metric("Scenarios", len(losses), "paths"),
                ],
                "charts": [
                    {
                        "title": "Discounted mark-to-market loss distribution",
                        "x": x,
                        "xlabel": "Loss / USD",
                        "series": [{"name": "Scenario count", "values": hist.tolist()}],
                    },
                    {
                        "title": "Mean-loss estimation error · 12 independent runs",
                        "x": budgets,
                        "xlabel": "Payoff paths",
                        "series": convergence,
                    },
                ],
                "table": {
                    "columns": [
                        "Asset",
                        "Position",
                        "Instrument",
                        "Spot",
                        "Strike",
                        "Annual volatility",
                    ],
                    "rows": [
                        ["A", 50, "Call", 100, 100, 0.22 * stress],
                        ["B", -30, "Put", 80, 85, 0.28 * stress],
                        ["C", 40, "Call", 120, 115, 0.18 * stress],
                        ["D", 20, "Put", 60, 65, 0.35 * stress],
                    ],
                },
                "method": [
                    "Correlated GBM terminal simulation; Black–Scholes revaluation of remaining six-month option maturities. Positions and currency scale are explicit.",
                    "Randomized Sobol and antithetic designs are compared with independent Monte Carlo using 12 separate seeds at four budgets. The high-budget independent reference is itself a simulation estimate.",
                    "VaR is an empirical loss quantile; expected shortfall averages observations in the empirical tail. PFE uses positive future book value, not losses.",
                ],
                "limitations": [
                    "Illustrative synthetic asset parameters, not calibrated market data or an investment recommendation.",
                    "Lognormal dynamics omit stochastic volatility, defaults, liquidity and funding; tail estimates remain model dependent.",
                    "The chart compares mean-loss error. It does not establish quantum advantage or tail-quantile confidence intervals.",
                ],
                "reproduce": "python src/studio.py --out docs/study.json",
            }
        )
    p = np.array([0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05])
    payoff = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
    state, Q = amplitude_oracle(p, payoff)
    truth = float(p @ payoff)
    powers = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    actual = [
        float(np.sum((np.linalg.matrix_power(Q, k) @ state)[1::2] ** 2)) for k in powers
    ]
    analytic = np.sin((2 * np.array(powers) + 1) * np.arcsin(np.sqrt(truth))) ** 2
    assert np.allclose(actual, analytic, atol=1e-10)
    estimate = quantum_estimate(state, Q)
    queries = estimate["queries"]
    classical = np.random.default_rng(431).binomial(1, truth, queries).mean()
    cases.append(
        {
            "id": "amplitude",
            "label": "Quantum oracle",
            "kicker": "AMPLITUDE ESTIMATION / EXPLICIT STATEVECTOR",
            "title": "Look inside the quantum estimator.",
            "description": "An eight-scenario register and a payoff ancilla, explicit unitary preparation, amplitude amplification and maximum-likelihood estimation.",
            "metrics": [
                metric("Exact encoded expectation", truth),
                metric("Estimated expectation", estimate["estimate"]),
                metric("Oracle preparation calls", queries),
                metric("Logical qubits", 4),
            ],
            "charts": [
                {
                    "title": "Amplified success probability",
                    "x": powers,
                    "xlabel": "Grover power",
                    "series": [
                        {"name": "Explicit unitary evolution", "values": actual},
                        {"name": "Analytic prediction", "values": analytic.tolist()},
                    ],
                }
            ],
            "table": {
                "columns": ["Grover power", "Successful shots", "Total shots"],
                "rows": [
                    [r["grover_power"], r["successes"], r["shots"]]
                    for r in estimate["counts"]
                ],
            },
            "method": [
                "Prepare amplitudes sqrt(p_j(1-f_j)) and sqrt(p_j f_j) on the scenario/ancilla basis. Householder state preparation is unitary and explicitly inverted inside the Grover operator.",
                f"A Bernoulli likelihood over Grover powers estimates the bounded expectation. The profile-likelihood bounding interval is {estimate['interval'][0]:.5f} to {estimate['interval'][1]:.5f}; it is an asymptotic, potentially disconnected likelihood set.",
                f"At the same {queries} preparation-query budget a recorded direct Bernoulli estimate is {classical:.5f}. One random trial is not a comparative performance study.",
            ],
            "limitations": [
                "Dense unitary simulation is exponential and is not a compiled fault-tolerant implementation.",
                "This bounded discrete payoff is separate from the portfolio's unbounded continuous loss; no claim that the oracle directly estimates its VaR is made.",
                "State preparation, inverse preparation and Grover queries are counted. Device gates, errors, compilation and classical inference costs are not hidden in a speedup claim.",
            ],
            "reproduce": "python src/studio.py --out docs/study.json",
        }
    )
    cases.append(calibration_case(calibrated_comparison(price_csv, as_of)))
    return {
        "schema_version": 1,
        "slug": "quantum-estimation-lab",
        "title": "Quantum Risk Observatory",
        "subtitle": "From a correlated derivatives book to the quantum estimation oracle.",
        "accent": "#65d6b1",
        "seed": 1729,
        "cases": cases,
        "references": [
            {
                "title": "Original portfolio PFE project",
                "url": "https://github.com/FyodorAmanov1/QStockPrediction",
            },
            {
                "title": "Amplitude amplification and estimation",
                "url": "https://arxiv.org/abs/quant-ph/0005055",
            },
            {
                "title": "Sample covariance definition (NumPy)",
                "url": "https://numpy.org/doc/1.20/reference/generated/numpy.cov.html",
            },
            {
                "title": "Ledoit and Wolf: covariance shrinkage background (fixed 10% diagonal choice used here)",
                "url": "https://ledoit.net/Honey_2004.pdf",
            },
        ],
        "provenance": "New implementation inspired by the application scope of QStockPrediction. Legacy cases use frozen synthetic parameters; calibration case records its own CSV provenance and dates. Exact CPU statevector experiments. No embedded market dataset or cloud hardware.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce the risk studio and optional CSV calibration comparison")
    parser.add_argument("--price-csv", type=Path, help="User price CSV with assets A, B, C, D; defaults to generated synthetic fixture")
    parser.add_argument("--as-of", help="ISO calibration cutoff; required with --price-csv")
    parser.add_argument("--out", type=Path, default=Path("docs/study.json"))
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.price_csv is not None and args.as_of is None:
        parser.error("--price-csv requires --as-of YYYY-MM-DD")
    remaining = ["--out", str(args.out)]
    if args.check is not None:
        remaining.extend(["--check", str(args.check)])
    finalize(experiment(args.price_csv, args.as_of), argv=remaining)
