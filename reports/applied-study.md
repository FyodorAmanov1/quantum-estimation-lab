# Applied case study: Quantum Risk Observatory

## Research question

Build a defensible derivatives-risk research application connecting classical scenario methods to explicit quantum estimation costs.

## Reproduce

`python src/studio.py --out docs/study.json`

## Base market

Four correlated option positions. Revalue the full book at a 21-day horizon and compare three simulation designs at matched path budgets.

### Method

Correlated GBM terminal simulation; Black–Scholes revaluation of remaining six-month option maturities. Positions and currency scale are explicit.

Randomized Sobol and antithetic designs are compared with independent Monte Carlo using 12 separate seeds at four budgets. The high-budget independent reference is itself a simulation estimate.

VaR is an empirical loss quantile; expected shortfall averages observations in the empirical tail. PFE uses positive future book value, not losses.

### Measured comparison

| Asset | Position | Instrument | Spot | Strike | Annual volatility |
| --- | --- | --- | --- | --- | --- |
| A | 50 | Call | 100 | 100 | 0.22 |
| B | -30 | Put | 80 | 85 | 0.28 |
| C | 40 | Call | 120 | 115 | 0.18 |
| D | 20 | Put | 60 | 65 | 0.35 |

### Limits

- Illustrative synthetic asset parameters, not calibrated market data or an investment recommendation.
- Lognormal dynamics omit stochastic volatility, defaults, liquidity and funding; tail estimates remain model dependent.
- The chart compares mean-loss error. It does not establish quantum advantage or tail-quantile confidence intervals.

## Volatility shock

Four correlated option positions. Revalue the full book at a 21-day horizon and compare three simulation designs at matched path budgets.

### Method

Correlated GBM terminal simulation; Black–Scholes revaluation of remaining six-month option maturities. Positions and currency scale are explicit.

Randomized Sobol and antithetic designs are compared with independent Monte Carlo using 12 separate seeds at four budgets. The high-budget independent reference is itself a simulation estimate.

VaR is an empirical loss quantile; expected shortfall averages observations in the empirical tail. PFE uses positive future book value, not losses.

### Measured comparison

| Asset | Position | Instrument | Spot | Strike | Annual volatility |
| --- | --- | --- | --- | --- | --- |
| A | 50 | Call | 100 | 100 | 0.396 |
| B | -30 | Put | 80 | 85 | 0.504 |
| C | 40 | Call | 120 | 115 | 0.324 |
| D | 20 | Put | 60 | 65 | 0.63 |

### Limits

- Illustrative synthetic asset parameters, not calibrated market data or an investment recommendation.
- Lognormal dynamics omit stochastic volatility, defaults, liquidity and funding; tail estimates remain model dependent.
- The chart compares mean-loss error. It does not establish quantum advantage or tail-quantile confidence intervals.

## Quantum oracle

An eight-scenario register and a payoff ancilla, explicit unitary preparation, amplitude amplification and maximum-likelihood estimation.

### Method

Prepare amplitudes sqrt(p_j(1-f_j)) and sqrt(p_j f_j) on the scenario/ancilla basis. Householder state preparation is unitary and explicitly inverted inside the Grover operator.

A Bernoulli likelihood over Grover powers estimates the bounded expectation. The profile-likelihood bounding interval is 0.24740 to 0.25293; it is an asymptotic, potentially disconnected likelihood set.

At the same 8960 preparation-query budget a recorded direct Bernoulli estimate is 0.24967. One random trial is not a comparative performance study.

### Measured comparison

| Grover power | Successful shots | Total shots |
| --- | --- | --- |
| 0 | 63 | 256 |
| 1 | 256 | 256 |
| 2 | 57 | 256 |
| 4 | 256 | 256 |
| 8 | 65 | 256 |

### Limits

- Dense unitary simulation is exponential and is not a compiled fault-tolerant implementation.
- This bounded discrete payoff is separate from the portfolio's unbounded continuous loss; no claim that the oracle directly estimates its VaR is made.
- State preparation, inverse preparation and Grover queries are counted. Device gates, errors, compilation and classical inference costs are not hidden in a speedup claim.

## Interpretation

The [calibrated portfolio comparison](QE02.md) adds a fourth studio case.
The bundled input is a generated synthetic price CSV, with 253 weekday closes
from 2025-01-13 through the explicit 2025-12-31 cutoff. At 32,768 common Sobol
paths, frozen assumptions give VaR/ES/PFE of 683.88/764.17/1,541.60 USD;
using CSV spots with frozen volatility/correlation gives
333.05/380.42/410.19 USD; fitting volatility/correlation gives
243.98/277.37/378.77 USD. The last two cases isolate the joint fitted-parameter
change at fixed spots and fixed contracts, including initial-book repricing.
Their paired 12-scramble VaR change averages −89.30 USD with numerical standard
error 0.59 USD. This describes model sensitivity, not an accuracy improvement.
Full dates, parameter snapshots, input fingerprint and replicate changes are
exported in the studio and lab artifacts.

User CSVs can be supplied with `--price-csv YOUR.csv --as-of YYYY-MM-DD` to
either command. Historical covariance is a constant pricing/scenario proxy,
while the drift stays at 3%. The Gaussian scenario model discards non-Gaussian
features even though the synthetic fixture uses Student-t innovations.

The convergence experiment measures mean-loss estimation, while the tail-risk metrics use the full recorded scenario set. The explicit quantum oracle is a separate bounded discrete problem. Connecting its encoding to portfolio exposure requires a documented approximation-error budget.

## Non-Gaussian scenario sensitivity

The [non-Gaussian model comparison](QE03.md) adds a fifth studio case. At
fixed fitted parameters, common normal draws and the same Black–Scholes
repricing rule, a specified two-state common variance changes 99% VaR from
242.99 to 271.51 USD and ES from 278.71 to 355.31 USD at seed 1729.
Across twelve paired seeds, mixture-minus-GBM differences average +26.55 USD
for VaR and +81.28 USD for ES, with numerical SEs 1.57 and 2.04 USD.
Mean loss decreases by 0.45 USD across paired means despite increased tail
risk. The new case uses plain Monte Carlo; its baseline need not equal the
calibration case's Sobol estimate.

The common variance is 4 with probability 0.1 and 2/3 otherwise. Expected
variance multiplier is 1 and standardized innovation kurtosis is 6. Conditional
GBM drift preserves expected terminal spots, while random drift adds a small
term to total log-return covariance. These are specified sensitivity states,
not fitted dynamics or a power-law distribution. The fixed pricing rule is
not a consistent mixture option-pricing model. Shared variance changes joint
tail dependence; fitted transitions, clustering and held-out validation remain
outside this comparison. Full run-level evidence and costs are exported.

## Concepts to explain in an interview

Explain the objective and units, where each baseline obtains its information, what was fixed before evaluation, how costs and uncertainty were measured, and one failure the model cannot capture. Reproduce the data before presenting a number; distinguish an implementation demonstration from a validated research conclusion.

## Next work

- Validate the specified non-Gaussian comparison on held-out observations and quantify sensitivity to regime assumptions; covariance fitting alone does not identify tail shape.
- Map a bounded positive-exposure payoff into the oracle and separate discretization, truncation and estimation errors.
- Measure amplitude-estimator bias and interval coverage across independent trials, including boundary probabilities.
