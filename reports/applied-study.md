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

The convergence experiment measures mean-loss estimation, while the tail-risk metrics use the full recorded scenario set. The original eight-scenario quantum example remains reproducible. QE04 adds a separate encoding of the actual option book's bounded empirical positive exposure with a documented approximation-error budget.

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

## Bounded option-book exposure

The [QE04 experiment](QE04.md) adds **Bounded exposure oracle** to the studio.
It uses the same dated fitted parameters and actual four-contract book under
GBM and the specified variance mixture. Positive undiscounted net future value
is capped at 400 USD, then rounded down to 32 nodes, including zero and the cap.
Empirical bin probabilities and normalized node payoffs enter the existing
explicit amplitude oracle. Full nodes, counts, probabilities, measurement
counts and signed errors are exported for replay. User CSV commands feed this
case as well as calibration and model risk.

For GBM, the 32768-path raw mean is 145.6810 USD. Truncation removes 0.1725 USD
and grid rounding 5.8415 USD; the exact encoded mean is 139.6669 USD. Amplitude
estimation gives 138.5031 USD, a further error of -1.1638 USD. Under the mixture,
raw mean is 146.7929 USD, cap loss 1.3641 USD and grid loss 5.8761 USD; the
estimate is 138.3389 USD. Larger caps can reduce cap loss while worsening grid
resolution at fixed register size. Independent numerical references and twelve
scenario scrambles are reported separately from conditional measurement error.

The exact classical histogram expectation is already known during preparation.
At 8960 preparation/inverse calls per model the recorded amplitude estimates
are less accurate against that expectation than either classical sampling
baseline with 8960 draws. Both conditional likelihood intervals miss their
exact grid targets in these trials; no coverage study is claimed. This mean
exposure encoding estimates neither a tail quantile nor counterparty credit
loss, and supplies no hardware or quantum advantage evidence.

## Next work

- Validate the specified non-Gaussian comparison on held-out observations and quantify sensitivity to regime assumptions; covariance fitting alone does not identify tail shape.
- Develop a coherent scenario/repricing circuit with gate and precision costs; the bounded empirical exposure oracle currently requires classical preprocessing.
- Refine likelihood-set numerical edges and test cutoff calibration beyond the measured fixed-schedule amplitudes; endpoint inclusion alone does not ensure finite-sample coverage.

## Oracle compilation ledger

[QE06](QE06.md) adds replayable gate sequences and a studio case for both
bounded option-book exposure laws. A binary joint-probability tree prepares
the same state as the dense oracle, using Gray-code Ry/CNOT multiplexors.
The unitary completion differs from Householder preparation; the state and
Grover reflection agree numerically. State errors are below 3e-16 and Grover
matrix errors below 2e-15 for both bundled laws.

Each 32-node preparation has 63 Ry and 62 CNOT gates. Powers 0,1,2,4,8 with
256 shots each require 5,120 preparations, 3,840 inverse preparations and
3,840 Grover iterations per model. The full logical inventory is 564,480 Ry,
555,520 CNOT, 30,720 CCX, 3,840 CZ, 46,080 X and 3,840 Z. Six data and four
clean reusable workspace qubits are included. CCX is a separate three-qubit
primitive; arbitrary rotations are ideal, so these are not elementary-gate
or fault-tolerant totals. Controlled-A and controlled-Q counts are zero.
Forward A contributes 322,560 Ry and 317,440 CNOT; inverse A contributes
241,920 Ry and 238,080 CNOT. The studio charts these operation subtotals.
Full emitted shot circuits, including workspace, agree with dense Grover
probabilities to within 4.3e-15 for both laws over all scheduled powers;
nonzero-workspace probability is zero in the ideal replay.

Power-zero sampling with 8,960 shots uses the same preparation Ry/CNOT total
and avoids reflection gates. Equal calls do not imply equal runtime or
accuracy. The direct classical histogram sum already supplies the encoded
expectation. At fixed 400 USD cap, 16/32/64 grid nodes have measured GBM floor
errors -11.9546/-5.8415/-2.8672 USD and require 31/63/127 Ry per preparation.
Finer grids reduce this approximation error while increasing circuit size;
classical scenarios and option revaluation retain their original costs.
Power-zero shots use six peak qubits because they never need the four clean
reflection work bits; the amplitude schedule uses ten.

## Conditional amplitude interval coverage

[QE05](QE05.md) adds **Amplitude interval coverage** to the studio. It freezes
the two bounded empirical exposure laws and tests them alongside nine boundary
and interior amplitudes, using 128 independent trials per target. Powers
0,1,2,4,8 with 256 shots each cost 8960 preparation/inverse calls per trial;
the classical exact binomial comparator uses 8960 power-zero preparations.
All counts, seeds, intervals, coverage uncertainty, bias and RMSE are exported.

Legacy interval coverage is 0/128 at both exact boundaries because the theta
mesh excludes them. Adding exact endpoint likelihoods repairs support
exclusion, giving 128/128, but supplies no finite-sample calibration guarantee.
The GBM and mixture likelihood intervals cover 119/128 and 123/128 trials,
with Monte Carlo 95% bounds [87.07%,96.74%] and [91.12%,98.72%]. Their
conditional amplitude RMSEs are 0.00163487 and 0.00152844, compared with
0.00537047 and 0.00504297 for the sampled classical baseline. This retains
QE04's unfavorable individual trials while giving repeated-trial context.

Disconnected sets and finite-mesh hull errors are recorded explicitly.
Coverage is conditional on known empirical grid means and does not include
scenario, cap, grid or model error. Exact histogram means are already known
classically; ideal binomial simulation and equal logical calls establish no
hardware performance or quantum advantage.
