# Quantum Risk Observatory

**How do simulation choices affect derivatives risk estimates, and what would a quantum expectation estimator actually need to encode?**

Quantum Risk Observatory connects a small derivatives portfolio to inspectable numerical experiments. It compares classical sampling methods, separates tail loss from future exposure, and implements a bounded quantum oracle whose preparation cost is visible.

[Explore the dashboard](https://fyodoramanov1.github.io/quantum-estimation-lab/) · [Read the study](reports/applied-study.md) · [Inspect the implementation](src/studio.py)

## The experiment

A four-asset portfolio holds long and short European calls and puts. Correlated geometric Brownian motion generates future spots over a 21-trading-day horizon; Black–Scholes revaluation produces the portfolio distribution.

- **Risk:** empirical 99% value at risk (VaR), expected shortfall (ES) and potential future exposure (PFE).
- **Sampling:** independent Monte Carlo, paired antithetic paths and independently scrambled Sobol sequences.
- **Convergence:** mean-loss estimation error across 12 seeds and four path budgets, measured against a separate high-budget Sobol reference.
- **Stress:** asset volatilities multiplied by 1.8, with the correlation structure held fixed.

VaR and ES describe portfolio losses. PFE is a quantile of positive future mark-to-market value. They answer different risk questions and are reported separately.

## Recorded results

The bundled study uses 32,768 paths for each risk distribution. Values are in USD for the synthetic portfolio defined in the source.

| Scenario | 99% VaR | 99% ES | 99% PFE |
|---|---:|---:|---:|
| Baseline | 683.88 | 764.17 | 1,541.60 |
| Volatility stress | 1,118.24 | 1,236.20 | 2,617.16 |

The dashboard pairs these risk summaries with the convergence experiment. Its RMSE curves concern **mean loss**, rather than the sampling error of VaR or ES.

## Inside the quantum estimator

A separate eight-scenario example encodes a bounded expectation using three scenario qubits and one ancilla. The implementation constructs the state-preparation matrix, reflections and Grover operator explicitly, then fits an amplitude from binomial measurement counts using maximum likelihood.

The recorded example estimates **0.250180** against an exact expectation of **0.249500**, using **8,960 preparation-or-inverse calls**. The accounting includes the repeated preparations within Grover powers.

This example exposes the oracle assumptions behind amplitude estimation. It does not directly estimate the portfolio's VaR, compile a market-data oracle or establish a hardware speedup.

## Scope and interpretation

Prices, positions and correlations are synthetic. The model omits transaction costs, counterparty default and a calibrated volatility surface. Tail estimates retain Monte Carlo uncertainty. A price-CSV calibration helper exists in the source, but is not yet connected to the scenario-generation command.

The most useful extension is a controlled comparison of tail-estimation error and oracle construction cost on the same financial quantity.

## Run locally

Use Python 3.12. From the repository root:

```bash
python -m venv .venv
```

Activate it with `.venv/Scripts/Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux. Then run:

```bash
python -m pip install -r requirements.txt
python -m http.server 8000 --directory docs
```

Open
<http://localhost:8000> to explore the bundled study without recomputing it.

To regenerate the study or compare it with the saved artifact:

```bash
python src/studio.py --out docs/study.json
python src/studio.py --check docs/study.json
```

The browser displays recorded experiments; it does not run the Python simulation.
Use its charts, case selector and data exports to inspect the results.

## Explore the implementation

- [Experiment implementation](src/studio.py)
- [Recorded study](docs/study.json)
- [Applied study report](reports/applied-study.md)
- [References](REFERENCES.md)
- [MIT license](LICENSE)
