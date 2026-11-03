"""RF deck geometry, coordinate transforms and physical phase convention tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import subprocess
import numpy as np
import pandas as pd
from scipy.constants import c
import h5py

import rf_ring_study as study


class RFStudyTests(unittest.TestCase):
    def setUp(self):
        self.base=(study.ROOT/'bare-gaussian-cof.in').read_text()

    def test_d1_cyclic_geometry_and_rf_replacement(self):
        base,sections=study.cyclic_d1_lattice(self.base)
        self.assertIn('LATTICE: RING = (C1_D1, C1_Q',base)
        self.assertIn('C1_B1);',base)
        expected,theta=study.read_pose(next(line for line in self.base.splitlines() if line.startswith('C1_D1:')))
        np.testing.assert_allclose(sections[0]['centre_m'],expected+.5*np.array([np.sin(theta),0,np.cos(theta)]))
        deck,poses=study.rf_lattice(base,3.4e6,.05,np.arange(6)/3)
        self.assertEqual(deck.count(': VARIABLE_RF_CAVITY'),6)
        self.assertEqual(deck.count('_D1: DRIFT, L=1,'),6)
        self.assertNotIn('_D1A:',deck)
        self.assertEqual(deck.count('D2A: DRIFT, L=.45'),6)
        self.assertEqual(deck.count('D2B: DRIFT, L=.45'),6)
        # All RF centres are exactly the middle of their original D2 drift.
        for cell,pose in enumerate(poses,1):
            original,theta=study.read_pose(next(line for line in self.base.splitlines() if line.startswith(f'C{cell}_D2:')))
            np.testing.assert_allclose(pose['centre_m'],original+.5*np.array([np.sin(theta),0,np.cos(theta)]))
        self.assertIn('RFF: POLYNOMIAL_TIME_DEPENDENCE, P0=3.3999999999999999;',deck)
        with self.assertRaises(ValueError): study.rf_lattice(base,0,.05,[0]*6)
        with self.assertRaises(ValueError): study.read_pose('X=0,Y=0,Z=0,THETA=0,PHI=1,PSI=0;')

    def test_straight_launch_has_exact_orbit_centroid_and_momentum(self):
        phase=np.array([[.001,.002,.003,.004,.005,.006],[-.001,-.002,-.003,-.004,-.005,-.006]])
        theta=-np.pi/6; refp=.073*np.array([np.sin(theta),0,np.cos(theta)]); refr=np.array([-.5,0,1.4])
        particles=study.straight_particles(phase,refr,refp,theta)
        np.testing.assert_allclose(particles.mean(axis=0),np.r_[refr,refp],atol=3e-16)
        rotation=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        np.testing.assert_allclose((particles[:,:3]-refr)@rotation,phase[:,[0,2,4]],atol=2e-16)

    def test_circular_diagnostic_mapping_roundtrip(self):
        rng=np.random.default_rng(9)
        phase=rng.normal(size=(20,6))*1e-4
        cart=study.frenet_to_cartesian(phase,.073)
        restored=study.cartesian_to_frenet(cart[:,:3],cart[:,3:],.073)
        np.testing.assert_allclose(restored,phase,atol=7e-16)
        onorbit=study.frenet_to_cartesian([0,0,0,0,.01,0],.073)[0]
        self.assertLess(onorbit[3],0)  # Head follows the bend, not a parallel ray.

    def test_sinc_amplitude_and_head_focusing_sign(self):
        p0=.073; gamma=np.sqrt(1+p0*p0); beta=p0/gamma; mass=938272088.16
        times=np.arange(6)*2e-7+1e-7
        rf=study.rf_parameters(1.2e-6,times,p0,mass)
        phase=(2*np.pi*rf['frequency_hz']*times+rf['phases_rad'])%(2*np.pi)
        np.testing.assert_allclose(phase,np.pi,atol=5e-15)
        z=1e-5; grid=np.linspace(-.05,.05,10001)
        e=rf['amplitude_mvm']*1e6*np.sin(np.pi+rf['k_per_m']*(grid-z))
        kick=np.trapezoid(e,grid)/(beta*beta*gamma*mass)
        self.assertGreater(kick,0)
        self.assertAlmostEqual(kick/(rf['g_per_m']*z),1,delta=3e-11)

    def test_hermite_event_has_si_safe_root_and_aperture_filter(self):
        p0=.073; v=c*p0/np.sqrt(1+p0*p0)
        history=dict(t=np.array([0.,1e-9]),r=np.array([[0,0,-.4*v*1e-9],[0,0,.6*v*1e-9]]),p=np.array([[0,0,p0],[0,0,p0]]))
        event=study.plane_crossings(history,np.zeros(3),0)[0]
        self.assertAlmostEqual(event['time_s'],.4e-9,delta=1e-23)
        history['r'][:,0]=1
        self.assertEqual(study.plane_crossings(history,np.zeros(3),0),[])

    def test_normal_mode_probes_and_deck(self):
        phase,labels=study.normal_mode_probes(np.eye(6),1e-5)
        self.assertEqual(phase.shape,(13,6)); self.assertEqual(labels[0],'centre')
        np.testing.assert_array_equal(phase.sum(axis=0),0)
        deck=study.tracking_deck('LATTICE: RING=(D);',13,1e-10,8)
        self.assertIn('NPARTDIST=13',deck); self.assertIn('TURNS=8',deck)
        self.assertNotIn('INITIALORBIT',deck); self.assertNotIn('PC=',deck)

    def test_read_history_duplicate_times_and_invalid_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'case.h5'
            with h5py.File(path,'w') as output:
                for i,t in enumerate([1e-9,2e-9,2e-9]):
                    group=output.create_group(f'Step#{i}')
                    group.attrs['TIME']=t;group.attrs['RefPartR']=[0,0,.01*i];group.attrs['RefPartP']=[0,0,.073]
                    group.create_dataset('id',data=[1,0])
                    for key in ['x','y','z','px','py','pz','Ex','Ey','Ez']:
                        group.create_dataset(key,data=[1.,2.])
            history=study.read_history(path)
            np.testing.assert_array_equal(history['ids'],[0,1]);self.assertEqual(len(history['t']),2)
            self.assertEqual(history['local_r'][0,0,0],2)
            with h5py.File(path,'r+') as output: output['Step#1']['id'][0]=3
            with self.assertRaises(ValueError): study.read_history(path)

    def test_execution_timeout_targets_own_child_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);binary=directory/'binary';binary.write_text('mock')
            (directory/'case.in').write_text('QUIT;');(directory/'particles.txt').write_text('1\nx y z px py pz\n0 0 0 0 0 .073\n')
            child=Mock(pid=222,returncode=-15)
            child.wait.side_effect=[subprocess.TimeoutExpired('mock',.01),None]
            with patch.object(study.subprocess,'Popen',return_value=child):
                with self.assertRaises(RuntimeError): study.execute(directory,binary,timeout=.01)
            child.terminate.assert_called_once();child.kill.assert_not_called()
            record=json.loads((directory/'provenance.json').read_text())
            self.assertEqual(record['status'],'timed_out');self.assertEqual(record['pid'],222)


    def test_section_observations_use_static_straight_basis(self):
        p0=.073;v=c*p0/np.sqrt(1+p0*p0);dz=.001
        sections=[]
        for i in range(6):
            angle=2*np.pi*i/6
            sections.append(dict(centre_m=[2*np.cos(angle)-2,0,2*np.sin(angle)],theta_rad=-angle))
        phase=np.array([[0,0,0,0,0,0],[1e-5,2e-6,-3e-5,-4e-6,5e-5,6e-6]])
        t=[];r=[];p=[]
        for j in range(1,13):
            section=sections[j%6];theta=section['theta_rad'];ez=np.array([np.sin(theta),0,np.cos(theta)])
            for sign in [-1,1]:
                t.append(j*1e-7+sign*dz/v);r.append(np.array(section['centre_m'])+sign*dz*ez);p.append(p0*ez)
        lr=np.broadcast_to(phase[:,[0,2,4]],(len(t),2,3)).copy()
        lp=np.broadcast_to(p0*(phase[:,[1,3,5]]+[0,0,1]),(len(t),2,3)).copy()
        history=dict(t=np.array(t),r=np.array(r),p=np.array(p),local_r=lr,local_p=lp,ids=np.arange(2))
        result=study.section_observations(history,sections,p0,initial_phase=phase)
        self.assertEqual(result['phase'].shape,(13,2,6))
        np.testing.assert_allclose(result['phase'],np.broadcast_to(phase,(13,2,6)),atol=8e-16)
        np.testing.assert_array_equal(result['cells'],np.arange(13)%6)


    def test_reference_summary_distinguishes_particle_error_from_acceleration(self):
        p0=.073; v=c*p0/np.sqrt(1+p0*p0)
        times=np.arange(8)*1e-9
        history=dict(t=times,r=np.column_stack((np.zeros(8),np.zeros(8),v*times)),
            p=np.tile([0,0,p0],(8,1)),local_r=np.zeros((8,1,3)),
            local_p=np.tile([0,0,p0],(8,1,1)),electric=np.tile([0,0,5.],(8,1,1)))
        poses=[dict(centre_m=[0,0,v*(i+1.5)*1e-9],theta_rad=0) for i in range(6)]
        cof=dict(momentum_mc=[0,0,p0],position_m=[0,0,0],mass_eV=938272088.16)
        result=study.reference_summary(history,cof,poses)
        np.testing.assert_allclose(result['centre_times_s'],(np.arange(6)+1.5)*1e-9,atol=1e-23)
        self.assertEqual(result['max_particle_reference_position_m'],0)
        self.assertEqual(result['max_particle_reference_momentum_mc'],0)
        self.assertEqual(result['max_electric_V_per_m'],5)


    def test_fixed_step_execution_requires_explicit_expectation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);binary=directory/'binary';binary.write_text('mock')
            (directory/'case.in').write_text('QUIT;');(directory/'particles.txt').write_text('unit fixture')
            child=Mock(pid=223,returncode=0)
            with patch.object(study.subprocess,'Popen',return_value=child):
                result=study.execute(directory,binary,expected_boundary_control=False)
                self.assertFalse(result['boundary_controller_active'])
                self.assertFalse(result['expected_boundary_control'])
                with self.assertRaises(RuntimeError): study.execute(directory,binary)


    def test_rf_registration_uses_finite_intervals_for_prototype_named_clones(self):
        log="""Key: (1.150000 - 1.250000) number of overlapping elements1
          VARIABLE_RF_CAVITY
          Key: (1.250000 - 1.250000) number of overlapping elements2
          VARIABLE_RF_CAVITY
          Key: (5.444395 - 5.544395) number of overlapping elements1
          VARIABLE_RF_CAVITY
          """
        self.assertEqual(study.rf_body_intervals(log),[(1.15,1.25),(5.444395,5.544395)])


    def test_rf_registration_merges_passive_monitor_splits(self):
        log="""Key: (1.150000 - 1.195000)
        VARIABLE_RF_CAVITY
        Key: (1.195000 - 1.205000)
        C4_MON
        VARIABLE_RF_CAVITY
        Key: (1.205000 - 1.250000)
        VARIABLE_RF_CAVITY
        """
        self.assertEqual(study.rf_body_intervals(log),[(1.15,1.25)])

    def test_primary_mode_selection_excludes_active_cross_mode_leakage(self):
        table=pd.DataFrame([dict(id=i,mode=axis,active=True,phase_tune_absolute=10*i+j,spectral_tune_absolute=10*i+j+.01)
                            for i in range(4) for j,axis in enumerate('xyz')])
        primary=study.primary_mode_table(table,['centre','x0+','y0+','z0+'],np.array([.1,.2,-.3]))
        self.assertEqual(primary.id.tolist(),[1,2,3])
        self.assertEqual(primary['mode'].tolist(),['x','y','z'])
        np.testing.assert_allclose(primary.analytic_tune,np.array([.6,1.2,1.8])/(2*np.pi))


if __name__=='__main__': unittest.main()
