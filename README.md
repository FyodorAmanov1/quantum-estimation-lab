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

Bundled prices, positions and correlations are synthetic. The model omits transaction costs, counterparty default and a calibrated volatility surface. Tail estimates retain simulation uncertainty. The **Calibrated scenarios** dashboard case connects a validated price CSV to the same correlated option-book engine, with explicit cutoff dates, parameter snapshots and paired numerical comparisons.

The bundled synthetic calibration, as of **2025-12-31**, gives 99% VaR of **243.98 USD**, compared with **333.05 USD** using the same final spots and frozen volatility/correlation assumptions. This is parameter sensitivity, not evidence that the lower estimate is more accurate. See [QE02 methods and findings](reports/QE02.md).

The **Non-Gaussian model risk** dashboard case holds fitted spots, covariance
inputs and Black–Scholes repricing fixed while comparing GBM with a specified
two-state common variance mixture. At shared seed 1729 and 32,768 paths,
99% VaR changes from **242.99 to 271.51 USD**, and ES from **278.71 to
355.31 USD**. Across twelve paired seeds the mean changes are **+26.55 USD**
for VaR and **+81.28 USD** for ES, with numerical standard errors **1.57** and
**2.04 USD**. The mixture assumptions are not fitted or validated on market
returns; this measures scenario-model sensitivity. This case uses plain Monte
Carlo, so its GBM row differs from the calibration case's Sobol row.
See [QE03 methods and findings](reports/QE03.md). Both CSV commands below also
generate this comparison for the supplied history.

The remaining connection is a controlled comparison of tail-estimation error
and oracle construction cost on the same financial quantity.

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

## Calibrate your own scenarios

Supply UTF-8 CSV with header `date,A,B,C,D` (asset columns may be reordered),
strictly increasing ISO dates, and at least 60 complete positive price rows on
or before the explicit cutoff. Use synchronized, adjusted daily closes; no
calendar or corporate-action corrections are inferred. Assets A–D map to the
four existing positions and fixed strikes shown in the studio. Other asset
sets are rejected. No market dataset is bundled or downloaded.

```bash
python src/studio.py --price-csv YOUR.csv --as-of 2025-12-31 --out results/custom-study.json
python src/studio.py --price-csv YOUR.csv --as-of 2025-12-31 --check results/custom-study.json
python lab.py --price-csv YOUR.csv --as-of 2025-12-31 --output results/custom-results.json
```

To view your own comparison in the dashboard, use `--out docs/study.json` in
the first command and reload the browser. The exported artifact contains fitted
parameters, dates and a SHA-256 fingerprint, but no raw input price history or
local input path. Reproduction requires the same CSV and cutoff. Later rows
are excluded from fitting; the final included date is reported separately from
the requested as-of date.

The comparison includes frozen synthetic assumptions, those assumptions at CSV
final spots, and fitted volatility/correlation at those same spots. It uses
common Sobol shocks and records 12 independently scrambled paired differences.
Historical volatility also supplies the Black–Scholes repricing volatility;
this is a proxy, not an implied-volatility calibration. Drift remains 3%.
Fitting covariance does not preserve non-Gaussian tails or validate physical
risk probabilities. The additional mixture comparison specifies a common
variance multiplier of 4 with probability 10%, otherwise 2/3: mean multiplier
1 and standardized innovation kurtosis 6. Conditional GBM drift preserves
expected terminal spots. A shared latent state changes joint tail dependence;
there is no estimated regime transition or volatility-clustering model.

Regenerate the clearly labeled synthetic fixture and both measured artifacts:

```bash
python examples/generate_prices.py
python -m unittest discover -s tests -v
python lab.py --output results/latest.json
python src/studio.py --out docs/study.json
python src/studio.py --check docs/study.json
```

## Explore the implementation

- [Experiment implementation](src/studio.py)
- [Recorded study](docs/study.json)
- [Applied study report](reports/applied-study.md)
- [References](REFERENCES.md)
- [MIT license](LICENSE)
