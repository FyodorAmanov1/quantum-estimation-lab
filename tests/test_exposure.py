import json
from pathlib import Path
import unittest

import numpy as np

from src.studio import (amplitude_oracle, bounded_exposure_case,
                        bounded_exposure_comparison, exposure_grid,
                        portfolio_samples)


class ExposureTests(unittest.TestCase):
    def test_cap_atoms_rounding_and_exact_error_identity(self):
        x = np.array([0., 5., 10., 15., 20., 30., 40., 100.])
        g = exposure_grid(x, cap=30, levels=4)
        np.testing.assert_array_equal(g["counts"], [2, 2, 1, 3])
        np.testing.assert_allclose(g["nodes_usd"], [0, 10, 20, 30])
        self.assertAlmostEqual(g["capped_mean_usd"], np.minimum(x, 30).mean())
        self.assertAlmostEqual(g["grid_mean_usd"], np.mean([0, 0, 10, 10, 20, 30, 30, 30]))
        self.assertAlmostEqual(g["truncation_error_usd"], -10)
        self.assertAlmostEqual(g["tail_excess_mean_usd"], 10)
        self.assertEqual(g["fraction_above_cap"], 0.25)
        self.assertAlmostEqual(g["grid_mean_usd"] - x.mean(),
                               g["truncation_error_usd"] + g["grid_error_usd"])
        self.assertTrue(-g["spacing_usd"] <= g["grid_error_usd"] <= 0)

    def test_actual_unitary_encodes_grid_and_amplifies(self):
        for x in ([0.] * 4, [30., 50.], [0., 5., 15., 21., 100.]):
            g = exposure_grid(x, cap=30, levels=4)
            state, Q = amplitude_oracle(g["probabilities"], g["normalized_payoff"])
            a = g["grid_mean_usd"] / 30
            self.assertAlmostEqual(np.sum(state[1::2] ** 2), a)
            np.testing.assert_allclose(Q.T @ Q, np.eye(8), atol=1e-12)
            for k in (0, 1, 2, 4, 8):
                actual = np.sum((np.linalg.matrix_power(Q, k) @ state)[1::2] ** 2)
                expected = np.sin((2 * k + 1) * np.arcsin(np.sqrt(a))) ** 2
                self.assertAlmostEqual(actual, expected, places=11)

    def test_grid_bound_for_arbitrary_inputs(self):
        x = np.r_[0., np.random.default_rng(3).lognormal(4, 2, 1000)]
        for cap in (10., 200., 800.):
            for levels in (2, 16, 32, 64):
                g = exposure_grid(x, cap, levels)
                self.assertGreaterEqual(g["grid_error_usd"], -g["spacing_usd"] - 1e-10)
                self.assertLessEqual(g["grid_error_usd"], 1e-10)
                self.assertLessEqual(g["truncation_error_usd"], 0)
                self.assertAlmostEqual(sum(g["probabilities"]), 1)

    def test_invalid_grid(self):
        for x in ([], [-1], [np.nan], [np.inf], [[1, 2]]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                exposure_grid(x)
        for cap in (0, -1, np.nan, np.inf):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                exposure_grid([1], cap=cap)
        for levels in (0, 1, 3, True, 4.0):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                exposure_grid([1], levels=levels)

    def test_book_replay_error_components_costs_and_interface(self):
        c = bounded_exposure_comparison(paths=256, seed=91)
        self.assertEqual(c, bounded_exposure_comparison(paths=256, seed=91))
        for r in c["rows"]:
            x = portfolio_samples(256, seed=91, parameters=c["parameters"],
                                  scenario_model=r["model"])[1]
            g = r["grid"]
            self.assertEqual(g, exposure_grid(x))
            total = (g["truncation_error_usd"] + g["grid_error_usd"]
                     + r["amplitude_estimation_error_usd"])
            self.assertAlmostEqual(total, r["total_error_vs_empirical_usd"])
            self.assertAlmostEqual(total + r["scenario_reference_difference_usd"],
                                   r["total_difference_vs_reference_usd"])
            calls = sum(v["shots"] * (2 * v["grover_power"] + 1)
                        for v in r["amplitude_estimation"]["counts"])
            self.assertEqual(calls, 8960)
            self.assertEqual(calls, r["classical"]["draws_per_method"])
            self.assertEqual(len(r["scenario_runs"]), 12)
            self.assertEqual(len(r["sensitivity"]), 9)
            self.assertAlmostEqual(r["uncapped_mean_sd_across_scrambles_usd"],
                                   np.std([v["uncapped_mean_usd"] for v in r["scenario_runs"]], ddof=1))
        self.assertEqual(c["costs"]["future_option_revaluations"],
                         4 * c["costs"]["scenario_book_paths"])
        case = bounded_exposure_case(c)
        for chart in case["charts"]:
            for series in chart["series"]:
                self.assertEqual(len(chart["x"]), len(series["values"]))
        json.dumps(case, allow_nan=False)

    def test_user_csv_cutoff_reaches_oracle(self):
        with self.assertRaises(ValueError):
            bounded_exposure_comparison("unused.csv", paths=64)
        fixture = Path(__file__).resolve().parents[1] / "examples" / "synthetic-prices.csv"
        c = bounded_exposure_comparison(fixture, "2025-10-01", paths=64)
        self.assertEqual(c["parameters"]["window_end"], "2025-10-01")
        self.assertEqual(c["as_of"], "2025-10-01")
        self.assertEqual(c["data_source"], "User-supplied price CSV")
        self.assertNotIn(str(fixture), json.dumps(c))


if __name__ == "__main__":
    unittest.main()
