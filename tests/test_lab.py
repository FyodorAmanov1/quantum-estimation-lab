import unittest
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


if __name__ == "__main__":
    unittest.main()
