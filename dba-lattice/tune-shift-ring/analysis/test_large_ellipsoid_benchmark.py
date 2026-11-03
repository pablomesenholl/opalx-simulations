"""Unit tests for the physical sampler, units and force-gradient extraction."""
import tempfile
import re
import json
import subprocess
from unittest.mock import patch, Mock
import h5py
import large_ellipsoid_benchmark as benchmark
import unittest
from pathlib import Path
import numpy as np
from scipy.constants import c, elementary_charge, proton_mass
from large_ellipsoid_benchmark import (generate_ellipsoid, absolute_particles,
    analytic_fields, fit_gradient, drift_deck, write_particles, write_json, sha256)
from ellipsoid_model import field_tensor


class TestLargeEllipsoid(unittest.TestCase):
    def setUp(self):
        self.cov = np.diag([.01**2/5, .001**2, .005**2/5, .002**2, .05**2/5, .0001**2])
        self.cov[0,1] = self.cov[1,0] = -.2*np.sqrt(self.cov[0,0]*self.cov[1,1])

    def test_uniform_spatial_support_radial_cdf_and_covariance(self):
        phase = generate_ellipsoid(self.cov, 131072)
        radial = np.einsum('ni,ij,nj->n', phase[:,[0,2,4]], np.linalg.inv(5*self.cov[np.ix_([0,2,4],[0,2,4])]), phase[:,[0,2,4]])**.5
        self.assertLess(radial.max(), 1)
        np.testing.assert_allclose(np.mean(phase, axis=0), 0, atol=1e-17)
        # A volume-uniform sphere has P(r<R)=R^3, unlike a shell or Gaussian.
        for radius in [.25, .5, .75]:
            self.assertAlmostEqual(np.mean(radial < radius), radius**3, delta=5e-5)
        scale = np.sqrt(np.diag(self.cov))
        empirical = np.cov(phase/scale, rowvar=False, ddof=0)
        np.testing.assert_allclose(empirical, self.cov/np.outer(scale,scale), atol=5e-4)

    def test_cold_and_invalid_covariance(self):
        cold = self.cov.copy()
        cold[[1,3,5],:] = 0
        cold[:,[1,3,5]] = 0
        np.testing.assert_array_equal(generate_ellipsoid(cold, 32)[:,[1,3,5]], 0)
        with self.assertRaises(ValueError):
            generate_ellipsoid(self.cov, 31)
        bad = self.cov.copy()
        bad[1,1] = -1
        with self.assertRaises(ValueError):
            generate_ellipsoid(bad, 32)

    def test_fromfile_units_and_absolute_orbit(self):
        phase = generate_ellipsoid(self.cov, 32)
        particles = absolute_particles(phase, [1,2,3], [0,0,.073])
        np.testing.assert_allclose(particles.mean(axis=0), [1,2,3,0,0,.073], atol=1e-15)
        np.testing.assert_allclose(particles[:,3], .073*phase[:,1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'particles.txt'
            write_particles(path, particles)
            np.testing.assert_array_equal(np.loadtxt(path, skiprows=2), particles)
            self.assertEqual(path.read_text().splitlines()[1], 'x y z px py pz')
            before = sha256(path)
            write_particles(path, particles)
            self.assertEqual(before, sha256(path))
            write_json(Path(directory)/'summary.json', {'n':32})
            with self.assertRaises(ValueError):
                write_json(Path(directory)/'bad.json', {'bad':float('nan')})

    def test_relativistic_force_and_field_tensor_units(self):
        cr = self.cov[np.ix_([0,2,4],[0,2,4])]
        position = np.array([[.001,.002,.003],[-.002,.001,-.004]])
        p0, charge = .8, 41e-12
        gamma = np.sqrt(1+p0*p0)
        beta = p0/gamma
        electric, magnetic, force = analytic_fields(position, cr, charge, p0)
        np.testing.assert_allclose(force[:,:2], electric[:,:2]/gamma**2, rtol=1e-14)
        np.testing.assert_allclose(force[:,2], electric[:,2], rtol=1e-14)
        np.testing.assert_allclose(magnetic, np.cross([0,0,beta*c],electric)/c**2, rtol=1e-14)
        normalized_gradient = elementary_charge/(proton_mass*c*c*beta**2*gamma)
        np.testing.assert_allclose(force*normalized_gradient, position@field_tensor(cr,charge,beta,gamma).T, rtol=1e-14)

    def test_full_tensor_fit_intercept(self):
        rng = np.random.default_rng(17)
        position = rng.normal(size=(1000,3))*[.001,.01,.1]
        gradient = np.array([[1,2,3],[-4,5,6],[7,-8,9.]])*10000
        intercept = [7,-8,9]
        fit, offset, residual = fit_gradient(position, position@gradient.T+intercept)
        np.testing.assert_allclose(fit, gradient, atol=1e-9)
        np.testing.assert_allclose(offset, intercept, atol=1e-11)
        self.assertLess(max(residual), 1e-11)

    def test_deck_charge_does_not_change_with_macroparticle_count(self):
        for n in [131072,524288]:
            deck = drift_deck(n,64,41e-12)
            self.assertIn(f'NPARTDIST={n}',deck)
            self.assertEqual(float(re.search(r'BCHARGE=([^,;]+)',deck)[1]),41e-12)
            self.assertIn('REPARTFREQ=0',deck)
            self.assertIn('MAXBINS=1',deck)
            self.assertNotIn('INITIALORBIT',deck)
            self.assertNotIn('PC=',deck)


    def test_prepare_run_and_bounded_probe_without_opalx(self):
        # Mock only external execution. The actual sampler, deck writer, immutable
        # input checks and timeout-provenance paths still execute.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'repo'/'sandbox'/'tests'/'study'
            (root/'results').mkdir(parents=True)
            binary = root/'binary'
            binary.write_text('unit-test executable placeholder')
            (root/'config.json').write_text(json.dumps(dict(elements=[], cells=6,
                rms_emittance_x_m_rad=3e-6, rms_emittance_y_m_rad=3e-6)))
            (root/'bare-cof.in').write_text('ic: COF, DT=1e-10, TIMEINTEGRATOR="BORIS";')
            (root/'bare-gaussian-cof.in').write_text('LATTICE: RING=(D);\nDIST: DISTRIBUTION;')
            cof = dict(converged=True, residual=[0,0,0,0], section_rotation_wxyz=[1,0,0,0],
                       momentum_mc=[0,0,.073], position_m=[0,0,0])
            def external_run(command, **kwargs):
                if command[-1] == 'cof.in':
                    benchmark.write_json(Path(kwargs['cwd'])/'bare-cof.json',cof)
                return subprocess.CompletedProcess(command,0)
            with patch.object(benchmark,'ROOT',root), patch.object(benchmark.subprocess,'check_output',return_value=''), \
                 patch.object(benchmark.subprocess,'run',side_effect=external_run), \
                 patch.object(benchmark,'match_transverse',return_value=(np.array([.01,0,.005,0]),np.array([.1,.2]),0.)):
                study = benchmark.prepare(binary, cases=[(16,4)])
                self.assertEqual(json.loads((study/'study.json').read_text())['cases'],['n16-m4'])
                benchmark.run_case(study,'n16-m4')
                self.assertEqual(json.loads((study/'n16-m4/provenance.json').read_text())['status'],'completed')
                with self.assertRaises(ValueError):
                    benchmark.run_case(study,'n16-m4')
                child = Mock(pid=777,returncode=-15)
                child.wait.side_effect = [subprocess.TimeoutExpired('unit-test',.01),None]
                with patch.object(benchmark.subprocess,'Popen',return_value=child):
                    record = benchmark.ring_probe(study,timeout=.01)
                self.assertEqual(record['status'],'timed_out')
                child.terminate.assert_called_once()
                child.kill.assert_not_called()

    def test_field_analysis_synthetic_si_fixture_without_opalx(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            directory = study/'n128-m8'
            directory.mkdir()
            covariance = self.cov.copy()
            covariance[[1,3,5],:] = 0
            covariance[:,[1,3,5]] = 0
            phase = generate_ellipsoid(covariance,128)
            p0,charge,dt = .073,41e-12,1e-12
            gamma = np.sqrt(1+p0*p0)
            particles = absolute_particles(phase,[0,0,.2],[0,0,p0])
            write_particles(directory/'particles.txt',particles)
            cr = covariance[np.ix_([0,2,4],[0,2,4])]
            electric,magnetic,force = analytic_fields(phase[:,[0,2,4]],cr,charge,p0)
            delta = elementary_charge/(proton_mass*c)*dt*force
            benchmark.write_json(study/'study.json',dict(cases=['n128-m8'],covariance_analytic=covariance.tolist(),
                p0_mc=p0, charge_C=charge,beta=p0/gamma,gamma=gamma,semiaxes_m=[.01,.005,.05]))
            benchmark.write_json(directory/'provenance.json',dict(status='completed',count=128,mesh=8,dt_s=dt,elapsed_seconds=0))
            with h5py.File(directory/'case.h5','w') as output:
                output.attrs['ExUnit']='MV/m'  # The known metadata error must not rescale SI data.
                group=output.create_group('Step#0')
                group.attrs['TIME']=dt
                group.attrs['RefPartP']=[0,0,p0]
                group.create_dataset('id',data=np.arange(128))
                group.create_dataset('q',data=np.full(128,charge/128))
                for i,axis in enumerate('xyz'):
                    group.create_dataset(axis,data=phase[:,[0,2,4]][:,i])
                    group.create_dataset('p'+axis,data=particles[:,3+i]+delta[:,i])
                    group.create_dataset('E'+axis,data=electric[:,i])
                    group.create_dataset('B'+axis,data=magnetic[:,i])
            frame=benchmark.analyze(study)
            self.assertLess(frame.core_tensor_relative_frobenius_error.iloc[0],1e-12)
            self.assertLess(frame.kick_relative_error.iloc[0],1e-6)
            self.assertTrue((study/'force-convergence.png').is_file())


if __name__ == '__main__':
    unittest.main()
