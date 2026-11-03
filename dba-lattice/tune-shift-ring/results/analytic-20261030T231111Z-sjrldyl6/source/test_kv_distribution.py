import unittest
import numpy as np
from kv_distribution import quiet_sphere, transverse_coordinates, section_fromfile, envelopes_at, shell_invariant, moments
from kv_model import constant_solution


class DistributionTests(unittest.TestCase):
    def test_quiet_uniform_shell(self):
        u = quiet_sphere(65536, 123)
        np.testing.assert_allclose(np.sum(u*u, axis=1), 1., atol=1e-15)
        np.testing.assert_allclose(u.mean(axis=0), 0., atol=1e-16)
        np.testing.assert_allclose(u.T@u/len(u), np.eye(4)/4, atol=1e-14)
        # S3 has <u_i^4>=1/8; distinguish a sphere from fixed-amplitude tori.
        self.assertLess(abs(np.mean(u[:, 0]**4)-.125), .005)
        np.testing.assert_array_equal(u, quiet_sphere(65536, 123))
        with self.assertRaises(ValueError):
            quiet_sphere(63, 1)

    def test_matched_covariance_and_momentum(self):
        u = quiet_sphere(1024, 7)
        env, ex, ey = np.array([.01, .002, .004, -.001]), 12e-6, 8e-6
        xy = transverse_coordinates(u, env, ex, ey)
        report = moments(xy)
        self.assertAlmostEqual(report['rms_emittance_x_m_rad'], ex/4, places=16)
        self.assertAlmostEqual(report['rms_emittance_y_m_rad'], ey/4, places=16)
        covariance = np.array(report['covariance'])
        self.assertAlmostEqual(covariance[0, 1], env[0]*env[1]/4, places=16)
        np.testing.assert_allclose(shell_invariant(xy, env, ex, ey), 1., atol=2e-15)
        data = section_fromfile(xy, .073)
        np.testing.assert_allclose(np.linalg.norm(data[:, [1, 3, 5]], axis=1), .073, atol=3e-17)
        np.testing.assert_allclose(data[:, 1]/data[:, 5], xy[:, 1], atol=1e-17)
        np.testing.assert_allclose(data[:, 3]/data[:, 5], xy[:, 3], atol=1e-17)

    def test_periodic_envelope_sampling(self):
        eps, K = 12e-6, 2e-6
        a, _ = constant_solution(1., eps, K)
        els = [dict(length_m=.5, kx_m2=1., ky_m2=1.)]*2
        env = np.array([a, 0., a, 0.])
        values = envelopes_at(np.array([0., .5, 1., 3.2]), env, els, eps, eps, K)
        np.testing.assert_allclose(values, np.tile(env, (4, 1)), atol=2e-14)


if __name__ == '__main__':
    unittest.main()
