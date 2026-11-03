"""Physics checks against closed-form maps and the independent KV solver."""
import json
from pathlib import Path
import sys
import re
import unittest

import numpy as np
from scipy.constants import elementary_charge, proton_mass, c

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'model'))
sys.path.insert(0, str(ROOT/'tune-shift-ring/analysis'))
from machine import J, advance_pose, element_map, generate, hessian
from kv_model import matched

SPEC = json.loads((ROOT/'model/cell.json').read_text())


class MachineTests(unittest.TestCase):
    def test_sector_dispersion_and_longitudinal_map(self):
        gamma, rho, angle = 1.3, 2., np.pi/6
        m = element_map(SPEC['elements'][0], gamma)
        cosine, sine = np.cos(angle), np.sin(angle)
        expected = np.eye(6)
        expected[:2, :2] = [[cosine, rho*sine], [-sine/rho, cosine]]
        expected[2, 3] = rho*angle
        expected[0, 5], expected[1, 5] = rho*(1-cosine), sine
        expected[4, 0], expected[4, 1] = -sine, -rho*(1-cosine)
        expected[4, 5] = rho*(sine-angle)+rho*angle/gamma**2
        np.testing.assert_allclose(m, expected, atol=2e-15, rtol=2e-15)

    def test_full_ring_symplecticity_and_geometry(self):
        result = generate(SPEC)
        self.assertEqual(len(result['elements']), 30)
        self.assertLess(result['position_closure_m'], 2e-14)
        self.assertAlmostEqual(result['total_bend_rad'], 2*np.pi, places=13)
        self.assertAlmostEqual(result['circumference_m'], 25.766370614359175, places=12)
        self.assertLess(result['symplectic_error'], 2e-13)
        composed = np.eye(6)
        for el in result['elements']:
            matrix = np.array(el['map_6d'])
            np.testing.assert_allclose(matrix.T@J@matrix, J, atol=2e-14)
            composed = matrix@composed
        np.testing.assert_allclose(composed, result['ring_map_6d'], atol=3e-13)
        ring = np.array(result['ring_map_6d'])
        d = np.array(result['periodic_dispersion_m_and_slope'])
        np.testing.assert_allclose(ring[:2, :2]@d+ring[:2, 5], d, atol=2e-14)

    def test_tunes_against_independent_envelope_phases(self):
        kv = json.loads((ROOT/'tune-shift-ring/config.json').read_text())
        for a, b in zip(SPEC['elements'], kv['elements']):
            for key in ['length_m', 'kx_m2', 'ky_m2']:
                self.assertEqual(a[key], b[key])
        _, phase, _ = matched(kv['elements'], 4*kv['rms_emittance_x_m_rad'],
                              4*kv['rms_emittance_y_m_rad'], 0.)
        np.testing.assert_allclose(generate(SPEC)['full_tunes'], kv['cells']*phase,
                                   atol=2e-10, rtol=0)

    def test_drift_and_canonical_generator(self):
        el = SPEC['elements'][1]
        m = element_map(el, 2.)
        self.assertEqual(m[0, 1], 1.)
        self.assertEqual(m[2, 3], 1.)
        self.assertEqual(m[4, 5], .25)
        u = np.array([.001, .002, .003, .004, .005, .006])
        bend = SPEC['elements'][0]
        du = J@hessian(bend, 2.)@u
        np.testing.assert_allclose(du, [u[1], -.25*u[0]+.5*u[5], u[3], 0.,
                                        -.5*u[0]+u[5]/4, 0.], atol=1e-16)
        np.testing.assert_allclose(advance_pose([0, 0, 0], 1, 0), [0, 1, 0])
        with self.assertRaises(ValueError):
            hessian(el, 1.)

class DesignTests(unittest.TestCase):
    def test_reconstruct_then_local_tune_match(self):
        from optimize import tune_design
        result = tune_design(SPEC, [.8, 1.7])
        machine = generate(result)
        np.testing.assert_allclose(machine['full_tunes'], [.8, 1.7], atol=1e-9)
        self.assertLess(machine['position_closure_m'], 1e-13)
        self.assertEqual(result['elements'][0], SPEC['elements'][0])
        self.assertNotEqual(result['elements'][1]['length_m'], SPEC['elements'][1]['length_m'])
        with self.assertRaises(ValueError):
            tune_design(SPEC, [-1, 1.7])


class ArtifactTests(unittest.TestCase):
    def test_exported_field_force_normalization(self):
        # Independent Lorentz-force conversion ties exported lab fields to
        # the KV particle equations for every reference operating point.
        index = json.loads((ROOT/'REFERENCE_RUNS.json').read_text())
        for relative in index.values():
            folder = ROOT/relative
            summary = json.loads((folder/'summary.json').read_text())
            reference = json.loads((folder/'fft2d5_reference.json').read_text())
            beta, gamma = summary['beta_rel'], summary['gamma']
            p0 = beta*gamma*proton_mass*c
            for row in reference['targets']:
                for axis in ['x', 'y']:
                    field = row[f'lab_E{axis}_over_{axis}_V_per_m2']
                    gradient = elementary_charge*field/(p0*beta*c*gamma**2)
                    np.testing.assert_allclose(gradient, row[f'sc_defocusing_{axis}_per_m2'],
                                               rtol=5e-15, atol=1e-18)

    def test_placements_against_independent_opalx_input(self):
        text = (ROOT/'tune-shift-ring/bare-cof.in').read_text()
        for el in generate(SPEC)['elements']:
            line = next(line for line in text.splitlines() if line.startswith(el['name']+':'))
            pose = [float(re.search(r'(?:^|,\s*)'+key+r'=([^,;]+)', line).group(1))
                    for key in ['X', 'Z', 'THETA']]
            np.testing.assert_allclose(pose, el['entry_X_Z_theta'], atol=2e-14, rtol=0)


if __name__ == '__main__':
    unittest.main()
