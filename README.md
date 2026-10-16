# Quantum Estimation Lab

Reproducible classical and quantum estimation benchmarks for financial payoffs.

## Current implementation

An analytic Black-Scholes call reference, independent Monte Carlo samples, and correctly paired antithetic observations with approximate confidence intervals.

QE01 adds a repeated-seed study: 128 independent pseudorandom replicates per
method at 256, 1,024, 4,096, and 16,384 payoff evaluations. The default command
records RMSE and empirical 95% interval coverage against the analytic price,
plus every seed, estimate, standard error, and interval under
`metrics.convergence` in the JSON artifact. Original metrics and function/CLI
arguments remain available. See [reports/QE01.md](reports/QE01.md) for measured
findings, including coverage shortfalls.

## Reproduce

Python 3.12 and CPU execution. From this repository:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python lab.py --output results/latest.json

The measured artifact is [results/latest.json](results/latest.json). Data provenance and scientific limitations are recorded alongside the metrics. Primary references and implementation boundaries are in [REFERENCES.md](REFERENCES.md).

For a smaller exploratory study in Python, call
`lab.convergence_experiment(path_budgets=(256, 1024), n_seeds=12, seed=1729)`.
Each budget must be distinct, even, and at least four. The published default
uses four budgets and 128 replicates; changing the design changes the seed
allocation. Runs use fresh streams across budgets and methods, with no shared
draws. Compare methods at equal payoff counts: plain uses N normal draws;
antithetic uses N/2 normal draws and N/2 independent pair means. These are
synthetic CPU experiments; payoff counts do not measure runtime or quantum costs.

## Research process


Each completed milestone adds a report under reports/ explaining the question, method, executable evidence, limitations, and concepts to understand for an interview. Negative results are retained.
