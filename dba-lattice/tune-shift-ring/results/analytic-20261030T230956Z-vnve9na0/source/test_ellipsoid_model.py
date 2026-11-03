import json
from pathlib import Path
import unittest
import numpy as np
from scipy.constants import c, elementary_charge, epsilon_0, proton_mass
from scipy.integrate import quad
from scipy.linalg import expm, block_diag
from ellipsoid_model import (depolarization, field_tensor, axis_coefficients, transverse_pass,
                             match_transverse, transport_generator, initial_covariance,
                             rf_kick, ring_slip, evolve_covariance)
from kv_model import beam_parameters

CFG = json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())


class EllipsoidTests(unittest.TestCase):
    def test_shape_integrals(self):
        np.testing.assert_allclose(depolarization([1, 1, 1]), 1/3, atol=1e-15)
        a = np.array([.7, 1.3, 3.])
        expected = [np.prod(a)/2*quad(lambda t: 1/((a[i]**2+t)*np.sqrt(np.prod(a*a+t))), 0, np.inf,
                                     epsabs=1e-12)[0] for i in range(3)]
        np.testing.assert_allclose(depolarization(a), expected, rtol=1e-10)
        self.assertAlmostEqual(sum(depolarization(a)), 1.)
        np.testing.assert_allclose(depolarization(a*1e-6), depolarization(a), atol=1e-15)
        with self.assertRaises(ValueError):
            depolarization([0, 1, 2])

    def test_sphere_gauss_and_rotation(self):
        radius, charge, beta, gamma = .01, 1e-11, .1, 1.
        expected = elementary_charge*charge/(4*np.pi*epsilon_0*proton_mass*c*c*beta**2*radius**3)
        np.testing.assert_allclose(axis_coefficients(radius, radius, radius, charge, beta, gamma), expected)
        cov = np.diag(np.array([.01, .02, .03])**2/5)
        rot = expm(np.array([[0, -.7, 0], [.7, 0, 0], [0, 0, 0]]))
        np.testing.assert_allclose(field_tensor(rot@cov@rot.T, charge, beta, gamma),
                                   rot@field_tensor(cov, charge, beta, gamma)@rot.T, atol=1e-15)

    def test_long_bunch_cylinder_limit(self):
        beta, gamma, _, ia = beam_parameters(2.5)
        a, b, length, current = .01, .006, 100., .01
        charge = 4*length*current/(3*beta*c)  # match central line density
        K = 2*current/(ia*beta**3*gamma**3)
        sx, sy, sz = axis_coefficients(a, b, length, charge, beta, gamma)
        np.testing.assert_allclose([sx, sy], [2*K/(a*(a+b)), 2*K/(b*(a+b))], rtol=2e-7)
        self.assertLess(sz/max(sx, sy), 1e-7)

    def test_transverse_zero_charge(self):
        beta, gamma, _, _ = beam_parameters(2.5)
        initial, q, err = match_transverse(CFG['elements'], 3e-6, 3e-6, .05, 0., beta, gamma)
        np.testing.assert_allclose(6*q, [.7701075691328534, 1.6895363970202109], atol=1e-9)
        end, _, _ = transverse_pass(initial, CFG['elements'], 15e-6, 15e-6, .05, 0., beta, gamma)
        np.testing.assert_allclose(end[:4], initial, atol=1e-10)

    def test_transport_and_rf_symplectic(self):
        J = block_diag(*([np.array([[0, 1], [-1, 0]])]*3))
        for el in CFG['elements']:
            A = transport_generator(el, 1.003)
            np.testing.assert_allclose(A.T@J+J@A, 0., atol=1e-15)
        kick = rf_kick(.3)
        np.testing.assert_allclose(kick.T@J@kick, J, atol=1e-15)
        self.assertEqual((kick@np.array([0, 0, 0, 0, .02, 0]))[5], .006)
        slip = ring_slip(CFG['elements'], 6, 1.003)
        np.testing.assert_allclose(slip['dispersion'], [9.337460094228828, 0.], atol=1e-12)

    def test_coupled_covariance_exact_linear_limit(self):
        beta, gamma, _, _ = beam_parameters(2.5)
        initial = initial_covariance([.01, .002, .006, -.001], 3e-6, 3e-6, .05, 1e-4)
        rows, reason = evolve_covariance(initial, CFG['elements'], 1, 1, 0., beta, gamma, maximum_axis=10.)
        matrix = np.eye(6)
        for el in CFG['elements']:
            matrix = expm(transport_generator(el, gamma)*el['length_m'])@matrix
        np.testing.assert_allclose(rows[-1][1], matrix@initial@matrix.T, rtol=2e-6, atol=1e-12)
        self.assertIn('completed', reason)
        self.assertAlmostEqual(np.linalg.det(rows[-1][1])/np.linalg.det(initial), 1., places=6)


if __name__ == '__main__':
    unittest.main()
