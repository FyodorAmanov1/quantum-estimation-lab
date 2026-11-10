import json
import unittest

import numpy as np
from scipy.special import xlogy
from scipy.stats import binomtest

from src.coverage import (POWERS, SHOTS, LR_DROP, amplitude_coverage_case,
                          amplitude_coverage_comparison, exact_binomial_interval,
                          likelihood_diagnostics)
from src.studio import amplitude_oracle, quantum_estimate


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = amplitude_coverage_comparison()

    def test_default_acceptance_fixed_cost_and_independent_streams(self):
        c = self.study
        self.assertGreaterEqual(c['n_trials'], 100)
        self.assertEqual(c['queries_per_trial'], sum(SHOTS * (2*k+1) for k in POWERS))
        self.assertEqual(c['queries_per_trial'], 8960)
        self.assertEqual(c['shots_per_trial'], 1280)
        seeds = []
        for row in c['rows']:
            self.assertEqual(len(row['runs']), c['n_trials'])
            seeds += [r[k] for r in row['runs'] for k in ('measurement_seed', 'classical_seed')]
            np.testing.assert_allclose(row['explicit_probabilities'], row['measurement_probabilities'], atol=1e-10)
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(c['total_amplified_preparation_calls'], 11 * 128 * 8960)
        self.assertEqual(c['total_amplified_preparation_calls'], c['total_classical_preparation_calls'])
        self.assertEqual({r['target'] for r in c['rows'][-2:]}, {'gbm', 'variance_mixture'})
        for row in c['rows'][-2:]:
            self.assertAlmostEqual(row['encoded_mean_usd'], 400 * row['amplitude'])
        json.dumps(c, allow_nan=False)

    def test_boundary_failure_is_retained_and_endpoint_variant_includes_support(self):
        for row in (self.study['rows'][0], self.study['rows'][8]):
            self.assertIn(row['amplitude'], (0., 1.))
            self.assertEqual(row['methods']['legacy']['covered_count'], 0)
            self.assertEqual(row['methods']['legacy']['likelihood_set_covered_count'], 0)
            for method in ('endpoint_inclusive', 'classical_exact'):
                self.assertEqual(row['methods'][method]['covered_count'], 128)
            # A coverage uncertainty interval must not collapse even with 0 or 128 hits.
            self.assertGreater(row['methods']['legacy']['coverage_mc_interval95'][1], 0)
            self.assertLess(row['methods']['endpoint_inclusive']['coverage_mc_interval95'][0], 1)

    def test_replay_counts_intervals_and_summaries(self):
        for row in self.study['rows']:
            for run in (row['runs'][0], row['runs'][-1]):
                counts = np.random.default_rng(run['measurement_seed']).binomial(SHOTS, row['measurement_probabilities'])
                np.testing.assert_array_equal(counts, run['successes_by_power'])
                replay = likelihood_diagnostics(counts, row['amplitude'])
                self.assertEqual(replay['legacy'], run['legacy'])
                self.assertEqual(replay['endpoint_inclusive'], run['endpoint_inclusive'])
                count = np.random.default_rng(run['classical_seed']).binomial(8960, row['amplitude'])
                self.assertEqual(count, run['classical_successes'])
                expected = binomtest(int(count), 8960).proportion_ci(method='exact')
                np.testing.assert_allclose(run['classical_exact']['interval'], expected, atol=1e-12)
            for method, summary in row['methods'].items():
                runs = [r[method] for r in row['runs']]
                errors = np.array([r['estimate'] - row['amplitude'] for r in runs])
                hits = sum(r['interval'][0] <= row['amplitude'] <= r['interval'][1] for r in runs)
                self.assertEqual(summary['covered_count'], hits)
                self.assertEqual(summary['empirical_coverage95'], hits / 128)
                self.assertAlmostEqual(summary['bias'], errors.mean())
                self.assertAlmostEqual(summary['rmse'], np.sqrt(np.mean(errors**2)))

    def test_likelihood_agrees_with_legacy_public_estimator_and_direct_formula(self):
        truth = .3491673613293586
        state, Q = amplitude_oracle([.25] * 4, [truth] * 4)
        old = quantum_estimate(state, Q, seed=71)
        counts = np.array([r['successes'] for r in old['counts']])
        result = likelihood_diagnostics(counts, truth)['legacy']
        self.assertEqual(result['estimate'], old['estimate'])
        np.testing.assert_allclose(result['interval'], old['interval'], atol=1e-14)
        # Independent likelihood reconstruction, including disconnected-set count.
        theta = np.linspace(1e-6, np.pi/2 - 1e-6, 32769)
        ll = np.zeros_like(theta)
        for k, s in zip(POWERS, counts):
            p = np.clip(np.sin((2*k+1)*theta)**2, 1e-12, 1-1e-12)
            ll += xlogy(s, p) + xlogy(SHOTS-s, 1-p)
        accepted = ll >= ll.max() - LR_DROP
        self.assertEqual(result['components'], np.count_nonzero(accepted & ~np.r_[False, accepted[:-1]]))

    def test_interface_and_invalid_designs(self):
        case = amplitude_coverage_case(self.study)
        self.assertEqual(case['id'], 'amplitude-coverage')
        self.assertEqual(len(case['table']['rows']), 33)
        for chart in case['charts']:
            self.assertTrue(all(len(s['values']) == len(chart['x']) for s in chart['series']))
        for n in (0, 1, True, 2.5):
            with self.assertRaises(ValueError):
                amplitude_coverage_comparison(n_trials=n)
        for seed in (-1, True, 2.5):
            with self.assertRaises(ValueError):
                amplitude_coverage_comparison(seed=seed)
        for counts in ([0]*4, [-1]*5, [257]*5, [1.5]*5):
            with self.assertRaises(ValueError):
                likelihood_diagnostics(counts, .5)
        self.assertEqual(exact_binomial_interval(0, 128)[0], 0.)
        self.assertEqual(exact_binomial_interval(128, 128)[1], 1.)


if __name__ == '__main__':
    unittest.main()
