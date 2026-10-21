# Primary references and implementation boundaries

- [Original portfolio PFE project](https://github.com/FyodorAmanov1/QStockPrediction)
- [Amplitude amplification and estimation](https://arxiv.org/abs/quant-ph/0005055)
- [NumPy sample covariance definition](https://numpy.org/doc/1.20/reference/generated/numpy.cov.html): the adapter uses observations in rows and the N−1 normalization. Annualization by 252 is an explicit equal-trading-step assumption.
- [Ledoit and Wolf, Honey, I Shrunk the Sample Covariance Matrix (2004)](https://ledoit.net/Honey_2004.pdf): primary background on covariance shrinkage. This application uses a fixed 10% diagonal target; it does **not** implement their estimated optimal intensity or claim their empirical results.

The implementation, data provenance and scope limits are documented in [the technical case study](reports/applied-study.md). Citation identifies an algorithmic foundation, not equivalence with the full paper or a claim of hardware performance.
