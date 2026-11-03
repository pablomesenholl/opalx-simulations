"""Independent density, pair-force and normal-mode observable checks."""
import json,unittest
from pathlib import Path
import numpy as np
from scipy.constants import epsilon_0,elementary_charge,proton_mass,c
from ellipsoid_observables import (uniform_density_diagnostics,moment_diagnostics,virial_force_gradient,
                                  normal_mode_coordinates,phase_frequency,spectral_frequency,individual_mode_tunes)
from large_ellipsoid_benchmark import generate_ellipsoid
from match_ellipsoid_ring import rf_layout,bare_cell_map,normal_mode_basis
from kv_model import beam_parameters


class EllipsoidObservableTests(unittest.TestCase):
    def test_uniform_radial_profile_and_affine_invariance(self):
        covariance=np.diag([.02**2/5,1e-5,.01**2/5,2e-5,.06**2/5,1e-4])
        sample=generate_ellipsoid(covariance,32768);r=sample[:,[0,2,4]];cr=covariance[np.ix_([0,2,4],[0,2,4])]
        first=uniform_density_diagnostics(r,cr)
        self.assertLess(first['radial_cdf_error'],2e-4)
        self.assertLess(abs(first['fourth_radial_moment_relative_error']),2e-4)
        transform=np.array([[2,1,.5],[0,.8,-.3],[.2,0,1.]])
        second=uniform_density_diagnostics(r@transform.T+[1,2,3],transform@cr@transform.T)
        for key in first:self.assertAlmostEqual(first[key],second[key],places=10)

    def test_gaussian_density_is_not_confused_with_uniform_covariance_match(self):
        rng=np.random.default_rng(42);r=rng.normal(size=(131072,3));report=uniform_density_diagnostics(r)
        self.assertGreater(report['radial_cdf_error'],.15)
        self.assertGreater(report['outside_ellipsoid_fraction'],.15)
        self.assertAlmostEqual(report['fourth_radial_moment_relative_error'],.4,delta=.015)

    def test_virial_gradient_recovers_uniform_sphere_and_charge_scaling(self):
        radius=.01;beta=.1;gamma=1.;charge=1e-12
        covariance=np.diag([radius**2/5,1e-5]*3)
        sample=generate_ellipsoid(covariance,32768)[:,[0,2,4]]
        gradient,error=virial_force_gradient(sample,charge,beta,gamma,pairs=262144,batches=16)
        expected=elementary_charge*charge/(4*np.pi*epsilon_0*proton_mass*c*c*beta*beta*radius**3)
        self.assertTrue(np.all(abs(np.diag(gradient)/expected-1)<6*np.diag(error)/expected+.003))
        doubled,_=virial_force_gradient(sample,2*charge,beta,gamma,pairs=262144,batches=16)
        np.testing.assert_allclose(doubled,2*gradient,rtol=2e-15)

    def test_phase_and_independent_spectrum_resolve_signed_and_integer_tunes(self):
        turns=np.arange(193)/6;frequency=np.array([.6095578205,1.48565709,-.25971868])
        signal=np.exp(2j*np.pi*turns[:,None]*frequency+[.2j,.7j,1.1j])
        estimate,residual=phase_frequency(signal,turns)
        np.testing.assert_allclose(estimate,frequency,atol=2e-14,rtol=0)
        self.assertLess(max(residual),2e-12)
        for i,value in enumerate(frequency):
            fitted,coherence=spectral_frequency(signal[:,i],turns,int(np.sign(value)))
            self.assertAlmostEqual(fitted[0],value,delta=3e-6)
            self.assertGreater(coherence[0],.999999)

    def test_dispersion_separated_modes_and_explicit_centroid(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
        beta,gamma,*_=beam_parameters(cfg['kinetic_energy_MeV'])
        basis,signed=normal_mode_basis(bare_cell_map(rf_layout(cfg['elements']),gamma,.001))
        turns=np.arange(193)/6;mu=signed*6/(2*np.pi);phase=np.zeros((len(turns),4,6))
        coordinates=np.zeros_like(phase)
        for mode in range(3):
            angle=2*np.pi*mu[mode]*turns+.2
            coordinates[:,mode+1,2*mode]=.01*np.cos(angle)
            coordinates[:,mode+1,2*mode+1]=-.01*np.sin(angle)
        drift=np.column_stack([.001*np.sin(3*turns+i) for i in range(6)])
        phase=coordinates@basis.T+drift[:,None,:]
        result=individual_mode_tunes(phase,turns,np.linalg.inv(basis),signed,centroid=phase[:,0],batch_size=2)
        self.assertEqual(result['active'].sum(),3)
        for mode in range(3):
            self.assertAlmostEqual(result['phase_tune_signed'][mode+1,mode],mu[mode],places=11)
            self.assertLess(result['estimator_difference'][mode+1,mode],3e-6)
        np.testing.assert_allclose(normal_mode_coordinates(phase,np.linalg.inv(basis),drift),coordinates,atol=2e-16)

    def test_observation_npz_cli_contract_and_csv_output(self):
        import tempfile
        from analyze_ellipsoid_modes import analyze
        turns=np.arange(193)/6;frequencies=np.array([.61,1.49,-.26]);phase=np.zeros((len(turns),4,6))
        for mode,frequency in enumerate(frequencies):
            phase[:,mode+1,2*mode]=np.cos(2*np.pi*frequency*turns)
            phase[:,mode+1,2*mode+1]=-np.sin(2*np.pi*frequency*turns)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            np.savez(path/'observations.npz',phase=phase,turn_coordinate=turns,ids=np.arange(4))
            np.savez(path/'match.npz',normal_mode_inverse=np.eye(6),signed_cell_phases=frequencies*2*np.pi/6)
            report=analyze(path/'observations.npz',path/'match.npz',path/'output',central_id=0)
            self.assertEqual(sum(row['count'] for row in report['modes']),3)
            self.assertTrue((path/'output/mode-tunes.csv').is_file())

    def test_nonuniform_cell_sampling_is_rejected_by_spectrum(self):
        with self.assertRaisesRegex(ValueError,'uniformly'):
            spectral_frequency(np.ones(9),np.arange(9)**1.1)

    def test_matched_moments_do_not_imply_uniform_density(self):
        covariance=np.diag([2e-4,1e-5,3e-5,2e-5,1e-3,1e-4])
        sample=generate_ellipsoid(covariance,32768)
        self.assertLess(moment_diagnostics(sample,covariance)['max_scaled_covariance_error'],.002)
        rotation=np.eye(6)
        a=np.sqrt(covariance[0,0]/covariance[1,1]);rotation[:2,:2]=[[0,a],[-1/a,0]]
        rotated=sample@rotation.T
        self.assertLess(moment_diagnostics(rotated,covariance)['max_scaled_covariance_error'],.002)
        self.assertGreater(uniform_density_diagnostics(rotated[:,[0,2,4]])['radial_cdf_error'],.08)

if __name__=='__main__':unittest.main()
