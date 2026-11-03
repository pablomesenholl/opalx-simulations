"""Independent limits and consistency checks; no OPALX execution."""
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.constants import c, elementary_charge, proton_mass
from scipy.integrate import solve_ivp
from kv_model import (beam_parameters, plane_map, periodic_twiss, envelope_rhs,
                      propagate, matched, constant_solution, envelope_multipliers)

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())


class KVTests(unittest.TestCase):
    def test_proton_units(self):
        beta, gamma, rigidity, ia = beam_parameters(2.5)
        self.assertAlmostEqual((gamma-1)*proton_mass*c*c/elementary_charge/1e6, 2.5)
        self.assertAlmostEqual(rigidity, 0.228624, places=5)
        self.assertTrue(31e6 < ia < 32e6)
        self.assertAlmostEqual(beta, 0.07285, places=4)
        with self.assertRaises(ValueError):
            beam_parameters(0)

    def test_maps_and_twiss(self):
        for k in [-2.84, 0., .25]:
            m = plane_map(k, .3)
            self.assertAlmostEqual(np.linalg.det(m), 1.)
            np.testing.assert_allclose(m @ m, plane_map(k, .6), atol=1e-14)
        bx, ax = periodic_twiss([dict(length_m=1., kx_m2=4.)], 'x')
        self.assertAlmostEqual(bx, .5)
        self.assertAlmostEqual(ax, 0.)
        with self.assertRaises(ValueError):
            periodic_twiss([dict(length_m=1., kx_m2=-1.)], 'x')

    def test_zero_current_lattice(self):
        els = CONFIG['elements']
        initial, tunes, error = matched(els, 12e-6, 12e-6, 0.)
        np.testing.assert_allclose(6*tunes, [.7701075691328534, 1.6895363970202109], atol=2e-9)
        for axis, j in [('x', 0), ('y', 2)]:
            beta, alpha = periodic_twiss(els, axis)
            self.assertAlmostEqual(initial[j]**2/12e-6, beta, places=7)
            self.assertAlmostEqual(initial[j+1], -alpha*initial[j]/beta, places=10)
        self.assertLess(error, 2e-8)

    def test_exact_round_focusing(self):
        k, eps, K = 1., 12e-6, 8e-6
        a, omega = constant_solution(k, eps, K)
        els = [dict(length_m=1., kx_m2=k, ky_m2=k)]
        rhs = envelope_rhs(0., [a, 0., a, 0.], k, k, eps, eps, K)
        np.testing.assert_allclose(rhs[:4], 0., atol=1e-14)
        initial, tunes, _ = matched(els, eps, eps, K)
        np.testing.assert_allclose(initial, [a, 0., a, 0.], atol=2e-11)
        np.testing.assert_allclose(tunes, omega/(2*np.pi), atol=1e-10)
        end, ss, yy = propagate(initial, els, eps, eps, K, samples=10)
        self.assertEqual(len(ss), 11)
        np.testing.assert_allclose(yy[:, [0, 2]], a, atol=2e-11)
        np.testing.assert_allclose(np.abs(envelope_multipliers(initial, els, eps, eps, K)), 1., atol=2e-9)

    def test_current_scaling_and_convergence(self):
        els = CONFIG['elements']
        z, q0, _ = matched(els, 12e-6, 12e-6, 0.)
        a, q1, _ = matched(els, 12e-6, 12e-6, 1e-8)
        b, q2, _ = matched(els, 12e-6, 12e-6, 2e-8)
        np.testing.assert_allclose((q0-q2)/(q0-q1), 2., rtol=.003)
        self.assertTrue(np.all(q1 < q0))
        fine, qfine, _ = matched(els, 12e-6, 12e-6, 2e-8, rtol=2e-12)
        np.testing.assert_allclose(qfine, q2, atol=1e-9)
        np.testing.assert_allclose(fine, b, atol=1e-10)
        # Scaling both emittance and perveance leaves depressed optics unchanged.
        scaled, qs, _ = matched(els, 24e-6, 24e-6, 4e-8)
        np.testing.assert_allclose(qs, q2, atol=1e-9)
        np.testing.assert_allclose(scaled, np.sqrt(2)*b, atol=1e-10)

    def test_envelope_variational_map(self):
        els, ex, ey, K = CONFIG['elements'], 12e-6, 12e-6, 2.8530465327582383e-6
        initial, _, _ = matched(els, ex, ey, K)
        matrix = np.empty((4, 4))
        for j in range(4):
            step = np.eye(4)[j]*1e-7
            plus = propagate(initial+step, els, ex, ey, K)[0][:4]
            minus = propagate(initial-step, els, ex, ey, K)[0][:4]
            matrix[:, j] = (plus-minus)/2e-7
        numerical = np.linalg.eigvals(matrix)
        analytic = envelope_multipliers(initial, els, ex, ey, K)
        for value in analytic:
            self.assertLess(np.min(np.abs(numerical-value)), 2e-6)
        self.assertGreater(max(abs(numerical)), 1.06)

    def test_particle_matrix_phase(self):
        # Independent particle equations verify that envelope phase integrals
        # describe incoherent motion, rather than envelope-mode frequencies.
        els, ex, ey, K = CONFIG['elements'], 12e-6, 12e-6, 1.7048939963498039e-6
        initial, tunes, _ = matched(els, ex, ey, K)
        state = np.r_[initial, np.eye(2).ravel(), np.eye(2).ravel()]
        def rhs(s, state, kx, ky):
            a, _, b, _ = state[:4]
            mx = state[4:8].reshape(2, 2)
            my = state[8:12].reshape(2, 2)
            fx = np.array([[0., 1.], [-kx+2*K/(a*(a+b)), 0.]])
            fy = np.array([[0., 1.], [-ky+2*K/(b*(a+b)), 0.]])
            return np.r_[envelope_rhs(s, state[:4], kx, ky, ex, ey, K)[:4],
                         (fx@mx).ravel(), (fy@my).ravel()]
        for el in els:
            sol = solve_ivp(rhs, (0., el['length_m']), state, args=(el['kx_m2'], el['ky_m2']),
                            method='DOP853', rtol=2e-11, atol=2e-13)
            self.assertTrue(sol.success)
            state = sol.y[:, -1]
        for j, tune in zip([4, 8], tunes):
            matrix = state[j:j+4].reshape(2, 2)
            self.assertAlmostEqual(np.linalg.det(matrix), 1., places=9)
            self.assertAlmostEqual(np.trace(matrix)/2, np.cos(2*np.pi*tune), places=9)


if __name__ == '__main__':
    unittest.main()
