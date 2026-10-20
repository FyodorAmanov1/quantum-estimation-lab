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

## Concepts to explain in an interview

Explain the objective and units, where each baseline obtains its information, what was fixed before evaluation, how costs and uncertainty were measured, and one failure the model cannot capture. Reproduce the data before presenting a number; distinguish an implementation demonstration from a validated research conclusion.

## Next work

- Calibrate multi-asset scenarios from a user-supplied price CSV and compare model risk under non-Gaussian returns.
- Map a bounded positive-exposure payoff into the oracle and separate discretization, truncation and estimation errors.
- Measure amplitude-estimator bias and interval coverage across independent trials, including boundary probabilities.
