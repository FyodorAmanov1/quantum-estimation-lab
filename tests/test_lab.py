import unittest
from unittest.mock import patch
import json
from math import sqrt
import numpy as np
import lab


class EstimationTests(unittest.TestCase):
    def test_known_analytic_case(self):
        self.assertAlmostEqual(lab.black_scholes_call(rate=0.05), 10.450583572185565, places=10)

    def test_expiry_and_deterministic_payoff(self):
        self.assertEqual(lab.black_scholes_call(spot=90, maturity=0), 0)
        expected = 100 - 100 * np.exp(-0.03)
        self.assertAlmostEqual(lab.black_scholes_call(volatility=0), expected)
        np.testing.assert_allclose(lab.discounted_payoffs([0, 1], volatility=0), expected)

    def test_sampling_error(self):
        result = lab.monte_carlo(paths=65536)
        self.assertLess(abs(result["estimate"] - lab.black_scholes_call()), 5 * result["standard_error"])
        anti = lab.monte_carlo(antithetic=True)
        self.assertEqual(anti["independent_observations"], 4096)

    def test_invalid_paths(self):
        with self.assertRaises(ValueError):
            lab.monte_carlo(antithetic=True, paths=3)

    def test_antithetic_interval_uses_pair_means(self):
        # Reconstruct the independent observations, including the pair count.
        z = np.random.default_rng(91).standard_normal(32)
        pairs = (lab.discounted_payoffs(z) + lab.discounted_payoffs(-z)) / 2
        result = lab.monte_carlo(paths=64, seed=91, antithetic=True)
        expected_se = np.std(pairs, ddof=1) / sqrt(32)
        self.assertAlmostEqual(result["estimate"], np.mean(pairs))
        self.assertAlmostEqual(result["standard_error"], expected_se)
        np.testing.assert_allclose(result["ci95"],
                                   [np.mean(pairs) - 1.96 * expected_se,
                                    np.mean(pairs) + 1.96 * expected_se])
        self.assertEqual(result["payoff_evaluations"], 64)
        self.assertEqual(result["independent_observations"], 32)

    def test_summary_against_hand_calculated_errors(self):
        # Errors -2 and +4 give bias 1, RMSE sqrt(10), coverage 1/2.
        fixtures = [
            {"estimate": 8.0, "standard_error": 1.0, "ci95": [8.0, 10.0],
             "payoff_evaluations": 4, "independent_observations": 4},
            {"estimate": 14.0, "standard_error": 0.5, "ci95": [13.0, 15.0],
             "payoff_evaluations": 4, "independent_observations": 4},
        ]
        with patch.object(lab, "black_scholes_call", return_value=10.0), \
             patch.object(lab, "monte_carlo", side_effect=[dict(r) for r in fixtures * 2]):
            result = lab.convergence_experiment(path_budgets=[4], n_seeds=2)
        for summary in result["rows"][0]["methods"].values():
            self.assertAlmostEqual(summary["rmse"], sqrt(10))
            self.assertEqual(summary["bias"], 1)
            self.assertEqual(summary["empirical_coverage95"], 0.5)
            self.assertEqual(summary["covered_count"], 1)
            self.assertAlmostEqual(summary["coverage_mc_standard_error"], sqrt(0.125))
            self.assertEqual(summary["mean_standard_error"], 0.75)
            self.assertEqual(summary["mean_ci95_width"], 2)
            self.assertTrue(summary["runs"][0]["covers_analytic"])
            self.assertFalse(summary["runs"][1]["covers_analytic"])

    def test_seed_streams_replay_and_equal_cost(self):
        kwargs = dict(path_budgets=[32, 128], n_seeds=12, seed=123)
        study = lab.convergence_experiment(**kwargs)
        self.assertEqual(study, lab.convergence_experiment(**kwargs))
        seeds = []
        for row in study["rows"]:
            for method, summary in row["methods"].items():
                antithetic = method == "paired_antithetic"
                self.assertEqual(len(summary["runs"]), 12)
                for run in summary["runs"]:
                    seeds.append(run["seed"])
                    replay = lab.monte_carlo(paths=row["paths"], seed=run["seed"],
                                            antithetic=antithetic)
                    for key, value in replay.items():
                        self.assertEqual(run[key], value)
                    self.assertEqual(run["payoff_evaluations"], row["paths"])
                    self.assertEqual(run["independent_observations"],
                                     row["paths"] // (2 if antithetic else 1))
        self.assertEqual(len(seeds), len(set(seeds)))
        changed = lab.convergence_experiment(**dict(kwargs, seed=124))
        self.assertNotEqual(study["rows"], changed["rows"])

    def test_invalid_convergence_designs(self):
        for budgets in ([], [2], [3, 8], [4, 4], [4.0], [True], [-4]):
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                lab.convergence_experiment(path_budgets=budgets)
        for n_seeds in (0, 1, 1.5, True):
            with self.subTest(n_seeds=n_seeds), self.assertRaises(ValueError):
                lab.convergence_experiment(n_seeds=n_seeds)
        for seed in (-1, 1.5, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                lab.convergence_experiment(seed=seed)

    def test_default_experiment_acceptance_and_compatibility(self):
        result = lab.experiment()
        # The original public entry point and metrics remain available.
        self.assertEqual(result["metrics"]["monte_carlo"], lab.monte_carlo())
        self.assertEqual(result["metrics"]["antithetic"], lab.monte_carlo(antithetic=True))
        study = result["metrics"]["convergence"]
        self.assertGreaterEqual(study["n_seeds"], 12)
        self.assertGreaterEqual(len(set(study["path_budgets"])), 4)
        self.assertEqual(study["analytic_price"], result["metrics"]["analytic_price"])
        all_seeds = []
        for row in study["rows"]:
            self.assertEqual(set(row["methods"]), {"plain", "paired_antithetic"})
            for summary in row["methods"].values():
                self.assertEqual(len(summary["runs"]), study["n_seeds"])
                all_seeds.extend(run["seed"] for run in summary["runs"])
                self.assertTrue(np.isfinite(summary["rmse"]))
                self.assertGreaterEqual(summary["rmse"], 0)
                self.assertTrue(0 <= summary["empirical_coverage95"] <= 1)
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
