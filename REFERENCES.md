# Primary references and implementation boundaries

- [Original portfolio PFE project](https://github.com/FyodorAmanov1/QStockPrediction)
- [Amplitude amplification and estimation](https://arxiv.org/abs/quant-ph/0005055)
- [NumPy sample covariance definition](https://numpy.org/doc/1.20/reference/generated/numpy.cov.html): the adapter uses observations in rows and the N−1 normalization. Annualization by 252 is an explicit equal-trading-step assumption.
- [Ledoit and Wolf, Honey, I Shrunk the Sample Covariance Matrix (2004)](https://ledoit.net/Honey_2004.pdf): primary background on covariance shrinkage. This application uses a fixed 10% diagonal target; it does **not** implement their estimated optimal intensity or claim their empirical results.
- [Durham, SV mixture models with application to S&P 500 index returns (2007)](https://digitalcommons.calpoly.edu/fin_fac/15/), Journal of Financial Economics 85, 822–856, [DOI](https://doi.org/10.1016/j.jfineco.2006.06.005): primary background on discrete normal mixtures and conditional return shape. QE03 uses a specified two-state horizon variance mixture, not Durham's fitted dynamic model or its empirical findings. The variance normalization, innovation kurtosis and conditional expected-spot identities are derived explicitly in [QE03](reports/QE03.md).

The implementation, data provenance and scope limits are documented in [the technical case study](reports/applied-study.md). Citation identifies an algorithmic foundation, not equivalence with the full paper or a claim of hardware performance.
