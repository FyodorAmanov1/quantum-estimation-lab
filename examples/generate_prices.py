"""Generate the bundled synthetic calibration fixture, never market observations.

Run: python examples/generate_prices.py
Student-t innovations (df=5) deliberately challenge the Gaussian fitted model.
Dates are weekdays, without an exchange holiday calendar.
"""
from datetime import date, timedelta
from pathlib import Path
import csv
import numpy as np


def generate(path):
    dates = []
    day = date(2025, 12, 31)
    while len(dates) < 253:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day -= timedelta(days=1)
    dates.reverse()
    rng = np.random.default_rng(202502)
    correlation = 0.7 ** np.abs(np.arange(4)[:, None] - np.arange(4)[None, :])
    z = rng.standard_t(5, size=(252, 4)) * np.sqrt(3 / 5)
    returns = (z @ np.linalg.cholesky(correlation).T) * np.array([0.30, 0.20, 0.26, 0.40]) / np.sqrt(252)
    prices = np.array([100, 80, 120, 60]) * np.exp(np.vstack([np.zeros(4), np.cumsum(returns, axis=0)]))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["date", "A", "B", "C", "D"])
        for day, values in zip(dates, prices):
            writer.writerow([day, *[f"{value:.10f}" for value in values]])


if __name__ == "__main__":
    generate(Path(__file__).with_name("synthetic-prices.csv"))
