# Primary references and boundaries

- Black and Scholes (1973), The Pricing of Options and Corporate Liabilities: https://doi.org/10.1086/260062
- Glasserman (2004), Monte Carlo Methods in Financial Engineering: https://doi.org/10.1007/978-0-387-21617-1
- Brassard et al., Quantum Amplitude Amplification and Estimation: https://arxiv.org/abs/quant-ph/0005055

The initial baseline implements classical European-call estimation. Citing amplitude estimation identifies the planned research direction; it does not mean a quantum estimator is already implemented.

## QE01 reference verification

The following primary source pages were verified by web search on 2026-10-16:

- Black and Scholes (1973), [publisher record](https://www.journals.uchicago.edu/doi/abs/10.1086/260062).
  Supports the analytic European-call pricing reference under the stated model assumptions.
- Boyle, Broadie, and Glasserman (2001), *Monte Carlo methods for security pricing*,
  in *Option Pricing, Interest Rates and Risk Management*, pp. 185-239.
  [Author institution record](https://business.columbia.edu/faculty/research/monte-carlo-methods-security-pricing-0)
  and [author-hosted chapter](https://business.columbia.edu/sites/default/files-efs/pubfiles/4336/monte_carlo_methods_security_pricing.pdf).
  Pages 188-190 describe normal-approximation errors, computational cost, and
  antithetic pair averaging. The PDF was indexed in web search; direct PDF opening
  failed, so verification used indexed passages and the institution record.
- NumPy, [parallel random number generation](https://numpy.org/doc/stable/reference/random/parallel.html)
  and [SeedSequence API](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html).
  Documents reproducible spawned pseudorandom streams and their very high probability
  of independence. QE01 executes these streams sequentially on the CPU.

RMSE and coverage in QE01 are measured summaries, not claims of a new estimator.
Normal-approximation intervals need not attain nominal coverage in a finite sample.
