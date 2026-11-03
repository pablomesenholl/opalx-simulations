"""Numerical and Hamiltonian checks for periodic coupled moment matching."""
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.linalg import expm
from match_ellipsoid_ring import (J, bare_cell_map, mode_covariance, cell_pass,
                                 eigenemittances, match_covariance, rf_voltage, rf_layout, envelope_multipliers, normal_mode_basis)
from kv_model import beam_parameters


class MatchedEllipsoidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
        cls.beta,cls.gamma,*_=beam_parameters(cls.cfg['kinetic_energy_MeV'])
        cls.elements=rf_layout(cls.cfg['elements']); cls.emits=np.array([3e-6,3e-6,2e-6])
        cls.matrix=bare_cell_map(cls.elements,cls.gamma,.001)
        cls.cov,*_=mode_covariance(cls.matrix,cls.emits)

    def test_bare_map_symplectic_and_invariant_positive_covariance(self):
        np.testing.assert_allclose(self.matrix.T@J@self.matrix,J,rtol=0,atol=3e-14)
        self.assertGreater(np.linalg.eigvalsh(self.cov).min(),0)
        scale=np.sqrt(np.outer(np.diag(self.cov),np.diag(self.cov)))
        self.assertLess(np.max(abs((self.matrix@self.cov@self.matrix.T-self.cov)/scale)),3e-13)
        np.testing.assert_allclose(eigenemittances(self.cov),np.sort(self.emits),rtol=2e-13)

    def test_normal_modes_are_symplectic_and_rotate_with_signed_phase(self):
        basis,phases=normal_mode_basis(self.matrix)
        np.testing.assert_allclose(basis.T@J@basis,J,atol=3e-13,rtol=0)
        expected=np.zeros((6,6))
        for i,phase in enumerate(phases):
            expected[2*i:2*i+2,2*i:2*i+2]=[[np.cos(phase),np.sin(phase)],[-np.sin(phase),np.cos(phase)]]
        np.testing.assert_allclose(np.linalg.solve(basis,self.matrix@basis),expected,atol=3e-13,rtol=0)
        np.testing.assert_allclose(np.linalg.solve(basis,self.cov)@np.linalg.inv(basis.T),
                                   np.diag(np.repeat(self.emits,2)),atol=2e-18,rtol=0)
        self.assertLess(phases[2],0)

    def test_unstable_rf_mode_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unstable'):
            mode_covariance(bare_cell_map(self.elements,self.gamma,-.001),self.emits)

    def test_zero_charge_ode_matches_exact_coupled_map(self):
        # A mismatch exercises nonstationary covariance transport, not merely
        # cancellation for the periodic covariance used to construct the test.
        stretch=np.diag([1.2,1/1.2,.8,1/.8,1.1,1/1.1])
        initial=stretch@self.cov@stretch
        final,matrix,_=cell_pass(initial,self.elements,0,self.beta,self.gamma,.001,rtol=2e-11)
        scale=np.sqrt(np.outer(np.diag(initial),np.diag(initial)))
        self.assertLess(np.max(abs((final-self.matrix@initial@self.matrix.T)/scale)),2e-10)
        np.testing.assert_allclose(matrix,self.matrix,atol=8e-11,rtol=1e-9)

    def test_sampled_particle_maps_reproduce_covariance_trajectory(self):
        final,matrix,rows,maps=cell_pass(self.cov,self.elements,1e-12,self.beta,self.gamma,.001,
                                        rtol=2e-11,samples=3,return_maps=True)
        self.assertEqual(len(rows),len(maps))
        scale=np.sqrt(np.outer(np.diag(self.cov),np.diag(self.cov)))
        for (_,covariance),particle_map in zip(rows,maps):
            self.assertLess(np.max(abs((covariance-particle_map@self.cov@particle_map.T)/scale)),2e-9)
        np.testing.assert_array_equal(matrix,maps[-1])

    def test_nonzero_charge_symplectic_emittances_and_covariance_map(self):
        final,matrix,_=cell_pass(self.cov,self.elements,1e-12,self.beta,self.gamma,.001,rtol=2e-11)
        scale=np.sqrt(np.outer(np.diag(self.cov),np.diag(self.cov)))
        self.assertLess(np.max(abs((final-matrix@self.cov@matrix.T)/scale)),1e-9)
        np.testing.assert_allclose(eigenemittances(final),np.sort(self.emits),rtol=2e-8)
        np.testing.assert_allclose(matrix.T@J@matrix,J,atol=2e-9,rtol=0)

    def test_periodic_charge_match_closes_and_retains_eigenemittances(self):
        result=match_covariance(self.elements,1e-12,self.beta,self.gamma,.001,self.emits)
        self.assertLess(result['residual'],2e-7)
        np.testing.assert_allclose(eigenemittances(result['covariance']),np.sort(self.emits),rtol=2e-7)
        self.assertLess(result['phase'][0],mode_covariance(self.matrix,self.emits)[1][0])
        self.assertLess(result['phase'][1],mode_covariance(self.matrix,self.emits)[1][1])

    def test_observation_section_shift_conjugates_map_and_covariance(self):
        from shift_matched_ellipsoid_section import shifted_layout
        rotated,prefix=shifted_layout(self.elements)
        self.assertEqual(rotated[0]['name'],'D1b');self.assertEqual(rotated[-1]['name'],'D1a')
        self.assertAlmostEqual(sum(e['length_m'] for e in rotated),sum(e['length_m'] for e in self.elements))
        # Prefix contains no RF; zero slope disables the historical exit gap.
        covariance,transfer,_=cell_pass(self.cov,prefix,0,self.beta,self.gamma,0.,rtol=2e-11)
        new_map=bare_cell_map(rotated,self.gamma,.001)
        np.testing.assert_allclose(new_map,transfer@self.matrix@np.linalg.inv(transfer),atol=3e-11,rtol=1e-9)
        scale=np.sqrt(np.outer(np.diag(covariance),np.diag(covariance)))
        self.assertLess(np.max(abs((new_map@covariance@new_map.T-covariance)/scale)),2e-10)

    def test_finite_rf_layout_preserves_length_and_symplecticity(self):
        self.assertAlmostEqual(sum(e['length_m'] for e in self.elements),
                               sum(e['length_m'] for e in self.cfg['elements']),places=14)
        gaps=[e for e in self.elements if 'rf_weight_per_m' in e]
        self.assertEqual(len(gaps),1)
        self.assertAlmostEqual(gaps[0]['length_m']*gaps[0]['rf_weight_per_m'],1)

    def test_zero_charge_collective_modes_are_stable(self):
        values,matrix=envelope_multipliers(self.cov,self.elements,0,self.beta,self.gamma,.001)
        self.assertEqual(matrix.shape,(21,21))
        self.assertLess(np.max(abs(abs(values)-1)),2e-7)

    def test_rf_voltage_reconstructs_requested_momentum_slope(self):
        from scipy.constants import c, elementary_charge, proton_mass
        circumference=6*sum(e['length_m'] for e in self.elements)
        voltage,frequency=rf_voltage(.001,self.beta,self.gamma,circumference,20)
        slope=elementary_charge*voltage*(2*np.pi*frequency/(self.beta*c))/(self.beta**2*self.gamma*proton_mass*c*c)
        self.assertAlmostEqual(slope,.001,places=15)

if __name__=='__main__': unittest.main()
