"""Pure continuum checks, without OPALX execution."""
import unittest

import numpy as np
from scipy.constants import c, epsilon_0
from scipy.spatial.transform import Rotation

from ellipsoid_boost_sensitivity import cross_matrix, continuum_gradients, analyze_covariances
from large_ellipsoid_benchmark import analytic_fields


class BoostSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.covariance = np.diag(np.array([.013, .006, .05])**2 / 5)
        self.charge = 160e-12
        self.beta = .072854113
        self.direction = np.array([0., 0., 1.])

    def test_composition_matches_direct_lorentz_force(self):
        electric, magnetic, force = continuum_gradients(
            self.covariance, self.charge, self.beta, self.direction,
            np.array([.001, .002, self.beta]))
        position = np.array([.001, -.0005, .004])
        velocity = c * np.array([.001, .002, self.beta])
        np.testing.assert_allclose(force @ position,
            electric @ position + np.cross(velocity, magnetic @ position), rtol=2e-15)
        np.testing.assert_allclose(cross_matrix(velocity) @ position,
                                   np.cross(velocity, position), rtol=2e-15)

    def test_aligned_matches_existing_ellipsoid_oracle(self):
        p0 = self.beta / np.sqrt(1 - self.beta**2)
        positions = np.diag([.001, .002, .003])
        expected = analytic_fields(positions, self.covariance, self.charge, p0)
        gradients = continuum_gradients(self.covariance, self.charge, self.beta,
                                         self.direction, self.beta * self.direction)
        for matrix, values in zip(gradients, expected):
            np.testing.assert_allclose(positions @ matrix.T, values, rtol=2e-14, atol=1e-20)

    def test_stationary_sphere_gauss_law(self):
        radius = .02
        electric, magnetic, force = continuum_gradients(np.eye(3) * radius**2 / 5,
            self.charge, 0., self.direction, np.zeros(3))
        expected = np.eye(3) * self.charge / (4 * np.pi * epsilon_0 * radius**3)
        np.testing.assert_allclose(electric, expected, rtol=3e-15)
        np.testing.assert_array_equal(magnetic, np.zeros((3, 3)))
        np.testing.assert_array_equal(force, electric)

    def test_rotational_covariance(self):
        rotation = Rotation.from_rotvec([.3, -.4, .2]).as_matrix()
        base = continuum_gradients(self.covariance, self.charge, self.beta,
                                     self.direction, self.beta * self.direction)
        rotated = continuum_gradients(rotation @ self.covariance @ rotation.T,
            self.charge, self.beta, rotation @ self.direction,
            rotation @ (self.beta * self.direction))
        for expected, actual in zip(base, rotated):
            np.testing.assert_allclose(actual, rotation @ expected @ rotation.T,
                                       rtol=2e-13, atol=1e-20)

    def test_charge_and_size_scaling(self):
        args = (self.beta, self.direction, self.beta * self.direction)
        base = continuum_gradients(self.covariance, self.charge, *args)
        charge = continuum_gradients(self.covariance, 2 * self.charge, *args)
        size = continuum_gradients(4 * self.covariance, self.charge, *args)
        zero = continuum_gradients(self.covariance, 0., *args)
        for original, twice, expanded, empty in zip(base, charge, size, zero):
            np.testing.assert_allclose(twice, 2 * original, rtol=1e-15)
            np.testing.assert_allclose(expanded, original / 8, rtol=1e-15)
            np.testing.assert_array_equal(empty, np.zeros((3, 3)))

    def test_operator_bound_covers_sampled_force_directions(self):
        p0 = self.beta / np.sqrt(1 - self.beta**2)
        result = analyze_covariances([self.covariance], p0, [0., 62.5e-12, 125e-12])
        self.assertTrue((result.loc[result.dt_s == 0, 'relative_force_bound'] == 0).all())
        bounds = result.groupby('dt_s').relative_force_bound.max()
        self.assertAlmostEqual(bounds.loc[125e-12] / bounds.loc[62.5e-12], 2., delta=.003)
        angle = result.loc[result.dt_s == 125e-12, 'lag_angle_rad'].iloc[0]
        _, _, exact = continuum_gradients(self.covariance, self.charge, self.beta,
                                           self.direction, self.beta * self.direction)
        _, _, lagged = continuum_gradients(self.covariance, self.charge, self.beta,
            [np.sin(angle), 0., np.cos(angle)], self.beta * self.direction)
        positions = np.random.default_rng(401).normal(size=(1000, 3))
        ratios = np.linalg.norm(positions @ (lagged - exact).T, axis=1) / np.linalg.norm(positions @ exact.T, axis=1)
        self.assertLessEqual(ratios.max(), bounds.loc[125e-12] * (1 + 1e-10))

    def test_invalid_physical_inputs(self):
        with self.assertRaises(ValueError):
            continuum_gradients(-self.covariance, self.charge, self.beta,
                                 self.direction, self.beta * self.direction)
        with self.assertRaises(ValueError):
            continuum_gradients(self.covariance, self.charge, 1.,
                                 self.direction, self.beta * self.direction)
        with self.assertRaises(ValueError):
            continuum_gradients(self.covariance, self.charge, self.beta,
                                 [0., 0., 2.], self.beta * self.direction)
        with self.assertRaises(ValueError):
            analyze_covariances([self.covariance], .073, [-1e-10])


if __name__ == '__main__':
    unittest.main()
