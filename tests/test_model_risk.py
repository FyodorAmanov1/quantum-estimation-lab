import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src import studio
from examples.generate_prices import generate


class ModelRiskTests(unittest.TestCase):
    def test_mixture_formula_and_shared_normals(self):
        paths, seed, horizon = 512, 73, 21 / 252
        parameters = studio.frozen_parameters()
        spot, vol, corr = studio._book_parameters(parameters)
        z = np.random.default_rng(seed).standard_normal((paths, 4))
        u = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(3103,))).random(paths)
        variance = np.where(u < .1, 4, 2 / 3)[:, None]
        terminal = spot * np.exp((.03 - variance * vol**2 / 2) * horizon
                                 + vol * np.sqrt(horizon * variance) * (z @ np.linalg.cholesky(corr).T))
        future = sum(units * studio.option_value(terminal[:, j], strike, vol[j], .5-horizon, kind=kind)
                     for j, (units, strike, kind) in enumerate(zip(
                         [50, -30, 40, 20], [100, 85, 115, 65], ["call", "put", "call", "put"])))
        losses, exposure, initial = studio.portfolio_samples(paths, seed=seed, method="plain", scenario_model="variance_mixture")
        np.testing.assert_allclose(losses, initial - np.exp(-.03*horizon)*future, atol=1e-10)
        np.testing.assert_allclose(exposure, np.maximum(future, 0), atol=1e-10)
        self.assertEqual(initial, studio.portfolio_samples(paths, seed=seed, method="plain")[2])

    def test_variance_normalization_and_conditional_martingale(self):
        probabilities, states = np.array([.9, .1]), np.array([2/3, 4.])
        self.assertAlmostEqual(probabilities @ states, 1)
        self.assertAlmostEqual(3 * (probabilities @ states**2), 6)
        # Integrating a conditional Gaussian analytically cancels its drift correction.
        sigma, horizon, rate = .4, 21/252, .03
        expected_growth = np.exp((rate-states*sigma**2/2)*horizon + states*sigma**2*horizon/2)
        np.testing.assert_allclose(expected_growth, np.exp(rate*horizon), atol=1e-15)
        # Sample innovations independently: covariance should match, fourth moment should differ.
        n = 131072
        v = studio._scenario_variance(n, 991, "variance_mixture")
        z = np.random.default_rng(991).standard_normal((n, 4))
        x = np.sqrt(v[:, None]) * z
        np.testing.assert_allclose(np.cov(x, rowvar=False), np.eye(4), atol=.03)
        self.assertGreater(np.mean(x[:, 0]**4) / np.mean(x[:, 0]**2)**2, 5.5)

    def test_paired_runs_replay_and_summaries(self):
        c = studio.risk_model_comparison(paths=512)
        self.assertEqual(c, studio.risk_model_comparison(paths=512))
        self.assertEqual(c["cost"]["book_paths"], 26 * 512)
        self.assertEqual(c["cost"]["future_option_revaluations"], 26 * 512 * 4)
        for run in c["paired_runs"]:
            for name in ("gbm", "variance_mixture"):
                losses, exposure, initial = studio.portfolio_samples(
                    512, seed=run["seed"], method="plain", parameters=c["parameters"], scenario_model=name)
                self.assertEqual(initial, run[name]["initial_value"])
                self.assertEqual(run[name]["tail_observations"], 6)
                for key, value in studio.risk_summary(losses, exposure).items():
                    self.assertEqual(value, run[name][key])
                self.assertGreaterEqual(run[name]["es"], run[name]["var"])
            self.assertEqual(run["gbm"]["initial_value"], run["variance_mixture"]["initial_value"])
            for key, value in run["difference"].items():
                self.assertEqual(value, run["variance_mixture"][key] - run["gbm"][key])
        for key, summary in c["paired_differences"].items():
            values = [r["difference"][key] for r in c["paired_runs"]]
            self.assertAlmostEqual(summary["mean"], sum(values) / 12)
            self.assertAlmostEqual(summary["standard_deviation"], np.std(values, ddof=1))
            self.assertAlmostEqual(summary["standard_error_of_mean"], np.std(values, ddof=1)/np.sqrt(12))
            self.assertEqual(c["mixture_minus_gbm"][key], c["rows"][1][key] - c["rows"][0][key])
        case = studio.risk_model_case(c)
        self.assertEqual(case["id"], "model-risk")
        self.assertEqual(len(case["charts"][1]["x"]), 12)
        json.dumps(case, allow_nan=False)

    def test_legacy_compatibility_and_model_validation(self):
        for method in ("plain", "antithetic", "sobol"):
            for a, b in zip(studio.portfolio_samples(256, method=method),
                            studio.portfolio_samples(256, method=method, scenario_model="gbm")):
                np.testing.assert_array_equal(a, b)
            result = studio.portfolio_samples(256, method=method, scenario_model="variance_mixture")
            self.assertTrue(np.isfinite(result[0]).all())
        v = studio._scenario_variance(256, 3, "variance_mixture", "antithetic")
        np.testing.assert_array_equal(v[:128], v[128:])
        with self.assertRaises(ValueError):
            studio.portfolio_samples(256, scenario_model="unknown")
        with self.assertRaises(ValueError):
            studio.risk_model_comparison("unused.csv", paths=256)

    def test_csv_cutoff_and_asset_mapping_for_mixture(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            generate(path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            cutoff = rows[121][0]
            first = studio.risk_model_comparison(path, cutoff, paths=128)
            rows[-1][1] = "999999"
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows([[r[i] for i in [0, 4, 2, 1, 3]] for r in rows])
            second = studio.risk_model_comparison(path, cutoff, paths=128)
            self.assertNotEqual(first["parameters"]["input_sha256"], second["parameters"]["input_sha256"])
            self.assertEqual(first["parameters"]["window_end"], cutoff)
            for a, b in zip(first["rows"], second["rows"]):
                for key in ("initial_value", "var", "es", "pfe", "mean_loss"):
                    self.assertAlmostEqual(a[key], b[key], places=10)


if __name__ == "__main__":
    unittest.main()
