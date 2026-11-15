import json
import unittest

import numpy as np

from src.oracle_resources import (preparation_circuit, inverse_circuit, replay,
                                  zero_reflection_circuit, schedule_ledger,
                                  oracle_compilation_comparison, oracle_compilation_case)
from src.studio import amplitude_oracle, bounded_exposure_comparison


class OracleResourceTests(unittest.TestCase):
    def test_compiled_state_inverse_and_grover_for_sparse_and_boundary_laws(self):
        rng = np.random.default_rng(92)
        for levels in (2, 4, 8, 32):
            for p, f in ((rng.dirichlet(np.ones(levels)), rng.random(levels)),
                         (np.eye(levels)[-1], np.ones(levels)),
                         (np.eye(levels)[0], np.zeros(levels))):
                with self.subTest(levels=levels, payoff=f):
                    circuit = preparation_circuit(p, f)
                    A = replay(circuit['gates'], np.eye(2 * levels))
                    state, Q = amplitude_oracle(p, f)
                    np.testing.assert_allclose(A[:, 0], state, atol=2e-14)
                    np.testing.assert_allclose(replay(inverse_circuit(circuit['gates']), A),
                                               np.eye(2 * levels), atol=2e-14)
                    S0 = np.eye(2 * levels); S0[0, 0] = -1
                    q = -A @ S0 @ A.T @ np.diag(np.tile([1., -1.], levels))
                    np.testing.assert_allclose(q, Q, atol=3e-14)
                    for k in (0, 1, 2, 4, 8):
                        actual = np.sum((np.linalg.matrix_power(q, k) @ A[:, 0])[1::2]**2)
                        exact = np.sin((2*k+1) * np.arcsin(np.sqrt(np.dot(p, f))))**2
                        self.assertAlmostEqual(actual, exact, places=11)

    def test_zero_reflection_cleans_workspace_on_every_basis_and_superposition(self):
        for n in range(2, 7):
            circuit = zero_reflection_circuit(n)
            work = circuit['work_qubits']
            # Simultaneously test all data basis columns, with clean trailing work bits.
            inputs = np.eye(2**(n + work))[:, np.arange(2**n) * 2**work]
            expected = inputs.copy(); expected[:, 0] *= -1
            np.testing.assert_array_equal(replay(circuit['gates'], inputs), expected)

    def test_counts_for_each_power_and_equal_call_baseline(self):
        circuit = preparation_circuit([1/32]*32, np.linspace(0, 1, 32))
        counts = [{'grover_power': k, 'shots': 256} for k in (0, 1, 2, 4, 8)]
        ledger = schedule_ledger(circuit, counts)
        t = ledger['totals']
        self.assertEqual(ledger['per_preparation'], {'Ry': 63, 'CNOT': 62})
        self.assertEqual(ledger['peak_qubits'], 10)
        self.assertEqual(t['preparation_calls'], 5120)
        self.assertEqual(t['inverse_preparation_calls'], 3840)
        self.assertEqual(t['preparation_or_inverse_calls'], 8960)
        self.assertEqual(t['controlled_A_calls'], 0)
        self.assertEqual(t['gates'], {'Ry': 564480, 'CNOT': 555520, 'X': 46080,
                                     'Z': 3840, 'CZ': 3840, 'CCX': 30720})
        for row in ledger['rows']:
            self.assertEqual(row['preparation_calls'] - row['inverse_preparation_calls'], row['shots'])
            self.assertEqual(row['gates']['CCX'], 8 * row['grover_iterations'])
        baseline = schedule_ledger(circuit, [{'grover_power': 0, 'shots': 8960}])['totals']
        self.assertEqual(baseline['gates']['CNOT'], t['gates']['CNOT'])
        self.assertEqual(baseline['gates']['CCX'], 0)
        self.assertEqual(baseline['inverse_preparation_calls'], 0)

    def test_applied_evidence_and_studio(self):
        bounded = bounded_exposure_comparison(paths=64, seed=18)
        c = oracle_compilation_comparison(bounded)
        for row, source in zip(c['rows'], bounded['rows']):
            self.assertEqual(row['exact_grid_mean_usd'], source['grid']['grid_mean_usd'])
            self.assertEqual(row['ledger']['totals']['preparation_or_inverse_calls'],
                             source['amplitude_estimation']['queries'])
            self.assertLess(row['validation']['state_max_abs_error'], 1e-13)
            self.assertLess(row['validation']['grover_matrix_max_abs_error'], 1e-13)
            self.assertLess(max(p['absolute_difference'] for p in row['validation']['success_probabilities']), 1e-12)
        self.assertEqual([r['per_preparation']['Ry'] for r in c['grid_sensitivity_gbm']], [31, 63, 127])
        case = oracle_compilation_case(c)
        for chart in case['charts']:
            self.assertTrue(all(len(s['values']) == len(chart['x']) for s in chart['series']))
        json.dumps(case, allow_nan=False)

    def test_invalid_inputs(self):
        for p, f in (([], []), ([1], [0]), ([.2, .2], [0, 1]), ([1, 0], [0]),
                     ([1, 0], [0, np.nan]), ([1, 0], [0, 2]),
                     ([np.inf, 0], [0, 1]), ([[1, 0]], [[0, 1]])):
            with self.subTest(p=p, f=f), self.assertRaises(ValueError):
                preparation_circuit(p, f)
        c = preparation_circuit([.5, .5], [0, 1])
        for k, shots in ((-1, 4), (1.5, 4), (True, 4), (1, 0), (1, 2.5)):
            with self.assertRaises(ValueError):
                schedule_ledger(c, [{'grover_power': k, 'shots': shots}])


if __name__ == '__main__':
    unittest.main()
