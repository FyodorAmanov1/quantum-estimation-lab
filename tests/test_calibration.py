import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from src import studio
from examples.generate_prices import generate


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "prices.csv"
        generate(self.path)

    def read_rows(self):
        with self.path.open(newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle))

    def write_rows(self, rows):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)

    def test_covariance_and_dates_against_independent_calculation(self):
        rows = self.read_rows()
        prices = np.array([r[1:] for r in rows[1:]], float)
        returns = np.log(prices[1:] / prices[:-1])
        centered = returns - returns.mean(axis=0)
        raw = centered.T @ centered / (len(returns) - 1) * 252
        expected = raw * 0.9 + np.diag(np.diag(raw)) * 0.1
        fitted = studio.calibrate_prices(self.path, "2025-12-31")
        vol = np.array(fitted["volatility"])
        np.testing.assert_allclose(np.outer(vol, vol) * fitted["correlation"], expected, rtol=1e-11)
        np.testing.assert_array_equal(fitted["spot"], prices[-1])
        self.assertEqual(fitted["as_of"], "2025-12-31")
        self.assertEqual(fitted["window_end"], "2025-12-31")
        self.assertEqual(fitted["return_observations"], 252)
        self.assertGreater(np.linalg.eigvalsh(fitted["correlation"]).min(), 0)

    def test_cutoff_prevents_future_price_leakage(self):
        rows = self.read_rows()
        cutoff = rows[121][0]
        first = studio.calibrate_prices(self.path, cutoff)
        rows[-1][1] = "999999"
        self.write_rows(rows)
        second = studio.calibrate_prices(self.path, cutoff)
        for key in ("spot", "volatility", "correlation", "rows", "window_end"):
            self.assertEqual(first[key], second[key])
        self.assertNotEqual(first["input_sha256"], second["input_sha256"])
        self.assertEqual(first["window_end"], cutoff)
        with self.assertRaises(ValueError):
            studio.calibrate_prices(self.path, rows[30][0])

    def test_adapter_validation(self):
        original = self.read_rows()
        mutations = [lambda r: r[0].__setitem__(0, "timestamp"),
                     lambda r: r[0].__setitem__(2, "A"),
                     lambda r: r[1].__setitem__(1, "nan"),
                     lambda r: r[1].__setitem__(1, "0"),
                     lambda r: r[2].__setitem__(0, r[1][0]),
                     lambda r: r[1].__setitem__(0, "2025-99-99"),
                     lambda r: r[1].pop()]
        for mutation in mutations:
            rows = [list(r) for r in original]
            mutation(rows)
            self.write_rows(rows)
            with self.assertRaises(ValueError):
                studio.calibrate_prices(self.path)
        self.write_rows([original[0]] + [r[:1] + ["100"] * 4 for r in original[1:]])
        with self.assertRaises(ValueError):
            studio.calibrate_prices(self.path)

    def test_asset_mapping_and_engine_connection(self):
        fitted = studio.calibrate_prices(self.path)
        expected = studio.portfolio_samples(256, seed=91, parameters=fitted)
        rows = self.read_rows()
        self.write_rows([[r[i] for i in [0, 4, 2, 1, 3]] for r in rows])
        permuted = studio.calibrate_prices(self.path)
        actual = studio.portfolio_samples(256, seed=91, parameters=permuted)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-10)
        # Reconstruct correlated terminal spots and the first path book by hand.
        z = np.random.default_rng(31).standard_normal((4, 4))
        spot, vol, corr = studio._book_parameters(fitted)
        horizon = 21 / 252
        terminal = spot * np.exp((0.03 - vol**2 / 2) * horizon + np.sqrt(horizon) * (z @ np.linalg.cholesky(corr).T) * vol)
        units, strikes, kinds = [50, -30, 40, 20], [100, 85, 115, 65], ["call", "put", "call", "put"]
        initial = sum(units[j] * studio.option_value(spot[j], strikes[j], vol[j], .5, kind=kinds[j]) for j in range(4))
        future = sum(units[j] * studio.option_value(terminal[:, j], strikes[j], vol[j], .5-horizon, kind=kinds[j]) for j in range(4))
        losses, exposure, pv = studio.portfolio_samples(4, seed=31, method="plain", parameters=fitted)
        np.testing.assert_allclose(losses, initial - np.exp(-.03*horizon)*future)
        np.testing.assert_allclose(exposure, np.maximum(future, 0))
        self.assertEqual(pv, initial)
        self.assertFalse(np.allclose(expected[0], studio.portfolio_samples(256, seed=91)[0]))

    def test_legacy_defaults_and_bad_parameters(self):
        for method in ("plain", "antithetic", "sobol"):
            for a, b in zip(studio.portfolio_samples(256, method=method),
                            studio.portfolio_samples(256, method=method, parameters=studio.frozen_parameters())):
                np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(studio.risk_summary(*studio.portfolio_samples(32768)[:2])["var"], 683.8816824040545)
        for key, bad in [("assets", ["X", "B", "C", "D"]), ("volatility", [0, 1, 1, 1]),
                         ("correlation", np.ones((4, 4)).tolist()), ("spot", [1, 2, 3, float("inf")])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                studio.portfolio_samples(256, parameters=dict(studio.frozen_parameters(), **{key: bad}))

    def test_comparison_replay_and_summaries(self):
        with self.assertRaises(ValueError):
            studio.calibrated_comparison(self.path, paths=256)
        c = studio.calibrated_comparison(self.path, "2025-12-31", paths=256)
        self.assertEqual(c, studio.calibrated_comparison(self.path, "2025-12-31", paths=256))
        self.assertEqual(c["rows"][1]["parameters"]["spot"], c["rows"][2]["parameters"]["spot"])
        for row in c["rows"]:
            result = studio.portfolio_samples(256, parameters=row["parameters"])
            self.assertEqual(row["initial_value"], result[2])
            for key, value in studio.risk_summary(*result[:2]).items():
                self.assertEqual(row[key], value)
        for key, stats in c["paired_scramble_changes"].items():
            self.assertAlmostEqual(stats["mean"], sum(stats["runs"])/12)
            self.assertAlmostEqual(stats["standard_error_of_mean"], np.std(stats["runs"], ddof=1)/np.sqrt(12))
            self.assertEqual(c["calibrated_minus_spot_matched"][key], c["rows"][2][key]-c["rows"][1][key])
        json.dumps(studio.calibration_case(c), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
