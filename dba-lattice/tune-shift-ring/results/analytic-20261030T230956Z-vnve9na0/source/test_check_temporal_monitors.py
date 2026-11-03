"""Unit checks for monitor placement, frame decoding and straight interpolation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from check_temporal_monitors import (DEFAULT_STUDY, dense_phase_at, insert_monitors,
    local_to_lab, nominal_length, ordered_keys, phase_at_section, read_monitor_group)
from rf_ring_study import read_pose
import check_temporal_monitors as monitors


class TemporalMonitorTest(unittest.TestCase):
    def test_insertion_preserves_circumference_and_section_planes(self):
        original=(DEFAULT_STUDY/'bare-probes-32turns-10um/case.in').read_text()
        changed=insert_monitors(original)
        self.assertAlmostEqual(nominal_length(original),nominal_length(changed),places=12)
        sections=json.loads((DEFAULT_STUDY/'sections.json').read_text())
        import re
        for cell,section in enumerate(sections,1):
            command=re.search(r'^C'+str(cell)+r'_MON:[^;]*;',changed,re.MULTILINE)[0]
            origin,theta=read_pose(command)
            np.testing.assert_allclose(origin,section['centre_m'],atol=1e-15)
            self.assertEqual(theta,section['theta_rad'])
            self.assertIn('TYPE=TEMPORAL',command)
        self.assertEqual(changed.count(': MONITOR,'),6)

    def test_rejects_nonunit_drift(self):
        source=(DEFAULT_STUDY/'bare-probes-32turns-10um/case.in').read_text()
        with self.assertRaises(ValueError):
            insert_monitors(source.replace('C1_D1: DRIFT, L=1','C1_D1: DRIFT, L=2'))

    def test_rotated_reference_round_trip(self):
        section=dict(centre_m=[2.,0.,-3.],theta_rad=-2.1)
        p0=.073
        p_ref=p0*np.array([np.sin(-2.1),0.,np.cos(-2.1)])
        local_r=np.array([[1e-3,2e-3,3e-3],[-4e-3,-2e-3,.05]])
        local_p=np.array([[1e-4,2e-4,p0],[0.,-1e-4,p0*1.01]])
        lab_r,lab_p=local_to_lab(local_r,local_p,np.array(section['centre_m']),p_ref)
        result=phase_at_section(lab_r,lab_p,section,p0)
        np.testing.assert_allclose(result[:,[0,2,4]],local_r,atol=1e-15)
        np.testing.assert_allclose(result[:,[1,3,5]],local_p/p0-[0,0,1],atol=1e-15)

    def test_rejects_nonplanar_reference(self):
        with self.assertRaises(ValueError):
            local_to_lab(np.zeros((1,3)),np.zeros((1,3)),np.zeros(3),[0,1,1])

    def test_dense_straight_interpolation(self):
        h=dict(t=np.array([0.,1.]),r=np.array([[0,0,-.1],[0,0,.1]]),p=np.array([[0,0,.073],[0,0,.073]]),
            local_r=np.array([[[.01,0,0]],[[.02,0,0]]]),local_p=np.array([[[.001,0,.073]],[[.001,0,.073]]]))
        section=dict(centre_m=[0,0,0],theta_rad=0.)
        phase=dense_phase_at(h,.25,section,.073)
        np.testing.assert_allclose(phase,[[.0125,.001/.073,0,0,-.05,0]])
        with self.assertRaises(ValueError): dense_phase_at(h,2,section,.073)
        h['local_r'][:,:,2]=.9
        with self.assertRaises(ValueError): dense_phase_at(h,.25,section,.073)

    def test_group_time_and_id_order(self):
        with tempfile.TemporaryDirectory() as name:
            with h5py.File(Path(name)/'test.h5','w') as data:
                group=data.create_group('Step#10')
                data.create_group('Step#2'); data.create_group('metadata')
                group.attrs['TIME']=[3.]
                group.attrs['RefPartR']=[1.,0,2.]
                group.attrs['RefPartP']=[0.,0,.073]
                group['id']=[2,1]
                group['time']=[3.1,3.1]
                for key,value in dict(x=[.2,.1],y=[0,0],z=[0,0],px=[0,0],py=[0,0],pz=[.073,.073]).items():
                    group[key]=value
                result=read_monitor_group(group,dict(centre_m=[1.,0,2.],theta_rad=0),.073)
                np.testing.assert_array_equal(result['ids'],[1,2])
                np.testing.assert_allclose(result['phase'][:,0],[.1,.2])
                self.assertEqual(result['time_s'],3.)
                np.testing.assert_allclose(result['row_times_s'],3.1)
                self.assertEqual(ordered_keys(data),['Step#2','Step#10'])

class MonitorOrchestrationTest(unittest.TestCase):
    def test_prepare_fixed_run_without_executing_opalx(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(monitors.tempfile,'mkdtemp',return_value=tmp), patch.object(monitors,'execute') as execute, patch.object(monitors,'compare',return_value={}):
                directory=monitors.run(DEFAULT_STUDY,1e-10,True)
            self.assertEqual(directory,Path(tmp))
            self.assertFalse(execute.call_args.kwargs['expected_boundary_control'])
            deck=(directory/'case.in').read_text()
            self.assertIn('REPARTFREQ=10',deck)
            self.assertNotIn('REPARTFREQ=0',deck)
            self.assertIn('TURNS=2',deck)
            metadata=json.loads((directory/'study.json').read_text())
            self.assertAlmostEqual(metadata['nominal_length_m'],metadata['nominal_unmonitored_length_m'],places=12)

    def test_compare_group_coordinates_to_independent_dense_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); study=root/'study'; study.mkdir(); (study/'analytic').mkdir()
            np.savez(study/'analytic/match-0pc.npz',normal_mode_inverse=np.eye(6))
            sections=[dict(centre_m=[0,0,0],theta_rad=0) for _ in range(6)]
            (study/'sections.json').write_text(json.dumps(sections))
            (study/'bare-cof.json').write_text(json.dumps(dict(momentum_mc=[0,0,.073])))
            phase=np.vstack([np.zeros(6),np.eye(6)*1e-5])
            (root/'launch.json').write_text(json.dumps(dict(labels=['centre','x0+','x1+','y0+','y1+','z0+','z1+'])))
            (root/'provenance.json').write_text(json.dumps(dict(binary_sha256='unit-fixture')))
            for cell in range(6):
                with h5py.File(root/f'monitor-{cell+1}.h5','w') as data:
                    for turn in range(2):
                        group=data.create_group(f'Step#{turn}'); time=1+cell+6*turn
                        group.attrs['TIME']=[time]; group.attrs['RefPartR']=[0.,0.,0.]; group.attrs['RefPartP']=[0.,0.,.073]
                        group['id']=np.arange(7); group['time']=np.full(7,time+.01)
                        for column,name in enumerate(['x','px','y','py','z','pz']):
                            group[name]=phase[:,column] if column%2==0 else .073*(phase[:,column]+int(column==5))
            history=dict(t=np.arange(14.),r=np.zeros((14,3)),p=np.tile([0.,0.,.073],(14,1)),ids=np.arange(7),electric=np.ones((14,7,3)))
            with patch.object(monitors,'read_history',return_value=history), patch.object(monitors,'dense_phase_at',return_value=phase), patch.object(monitors,'plane_crossings',return_value=[dict(time_s=float(t)) for t in range(1,13)]):
                summary=monitors.compare(root,study)
            self.assertEqual(summary['passages'],12)
            self.assertLess(summary['max_position_error_m'],1e-15)
            self.assertLess(summary['max_momentum_error_p0'],1e-15)
            self.assertAlmostEqual(summary['max_row_time_error_s'],.01)

if __name__=='__main__': unittest.main()
