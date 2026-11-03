"""Short-arc unit tests; no OPALX execution."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import h5py
import numpy as np
from scipy.constants import c
import short_arc_ellipsoid as arc
from large_ellipsoid_benchmark import write_json

class ShortArcTests(unittest.TestCase):
    def setUp(self):
        self.cov=np.diag([1e-4,1e-6,4e-5,2e-6,1e-3,3e-6])
        self.cov[0,1]=self.cov[1,0]=-2e-6
    def test_paraxial_drift_and_zero_charge_closure(self):
        p0=.073; gamma=np.sqrt(1+p0*p0); s=c*p0/gamma*arc.DURATION
        expected=np.eye(6);expected[0,1]=expected[2,3]=s;expected[4,5]=s/gamma**2
        closure,drift=arc.predictions(self.cov,0,p0,arc.DURATION)
        np.testing.assert_allclose(drift,expected@self.cov@expected.T,rtol=1e-14,atol=1e-20)
        np.testing.assert_allclose(closure,drift,rtol=2e-12,atol=1e-19)
        charged,_=arc.predictions(self.cov,160e-12,p0,arc.DURATION)
        # The initially converging beam can lose momentum spread while the
        # repulsive force increases its position-momentum covariance.
        self.assertGreater(charged[0,1],closure[0,1])
    def test_exact_ballistic_and_coordinate_normalization(self):
        phase=np.zeros((3,6));phase[1,1]=.1;phase[2,5]=.01
        p0=.5; duration=2e-9; got=arc.ballistic_phase(phase,p0,duration)
        np.testing.assert_array_equal(got[0],phase[0])
        self.assertAlmostEqual(got[1,0],c*duration*.05/np.sqrt(1+.05**2+.5**2),delta=1e-16)
        self.assertLess(got[1,4],0);self.assertGreater(got[2,4],0)
        p=p0*phase[:,[1,3,5]];p[:,2]+=p0
        np.testing.assert_allclose(arc.phase_from_local(phase[:,[0,2,4]],p,p0),phase,atol=1e-16)
        with self.assertRaises(ValueError):arc.phase_from_local(np.zeros((2,3)),p,p0)
        with self.assertRaises(ValueError):arc.ballistic_phase(phase,0,duration)
    def test_dimensionless_covariance_comparison(self):
        self.assertEqual(arc.covariance_comparison(self.cov,self.cov)['normalized_frobenius_error'],0)
        scaled=arc.covariance_comparison(4*self.cov,self.cov)
        np.testing.assert_allclose(scaled['rms_relative_error'],1)
        np.testing.assert_allclose(scaled['eigenemittance_relative_error'],3)
    def test_decks_equal_endtime_fixed_charge_and_final_dumps(self):
        for dt,steps in [(6.25e-11,100),(3.125e-11,200)]:
            text=arc.make_deck('OPTION, PSDUMPFREQ=8, STATDUMPFREQ=8;\nLATTICE: RING=(D);',131072,64,dt,True)
            self.assertIn(f'MAXSTEPS={steps}',text);self.assertIn('PSDUMPFREQ=1000000000',text)
            self.assertIn('TYPE=OPEN',text);self.assertIn(f'BCHARGE={arc.CHARGE:.17g}',text)
            self.assertNotIn('TURNS',text)
        with self.assertRaises(ValueError):arc.make_deck('',16,4,2e-9,True)
    def test_final_snapshot_sorting_and_failure_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'case.h5'
            with h5py.File(path,'w') as output:
                g=output.create_group('Step#7');g.attrs['TIME']=arc.DURATION;g.attrs['RefPartP']=[0,0,.073]
                for name in ['x','y','z','px','py','Ex','Ey','Ez','Bx','By','Bz']:g[name]=[0.,0.]
                g['pz']=[.073,.073];g['q']=[arc.CHARGE/2]*2;g['id']=[1,0]
            result=arc.read_final(path,2,.073);np.testing.assert_array_equal(result['phase'],0)
            self.assertEqual(result['charge_C'],arc.CHARGE)
            with self.assertRaises(ValueError):arc.read_final(path,3,.073)
            with self.assertRaises(ValueError):arc.read_final(path,2,.073,1e-9)
    def test_prepare_and_timeout_provenance_without_opalx(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'binary';binary.write_text('test binary')
            study=arc.prepare(binary,output=root/'study',cases=[('small',16,4,6.25e-11,160)])
            self.assertEqual(json.loads((study/'study.json').read_text())['omp_threads'],8)
            self.assertEqual(np.load(study/'initial-160pc-n16.npz')['phase'].shape,(16,6))
            child=Mock(pid=42,returncode=-15);child.wait.side_effect=[subprocess.TimeoutExpired('test',.01),None]
            with patch.object(arc.subprocess,'Popen',return_value=child) as launched:
                with self.assertRaises(RuntimeError):arc.run_case(study,'small',timeout=.01)
                self.assertEqual(launched.call_args.kwargs['env']['OMP_NUM_THREADS'],'8')
            child.terminate.assert_called_once()
            record=json.loads((study/'small/provenance.json').read_text());self.assertEqual(record['status'],'timed_out')
            with self.assertRaises(ValueError):arc.run_case(study,'small')
            record['status']='prepared';write_json(study/'small/provenance.json',record);binary.write_text('changed')
            with self.assertRaises(ValueError):arc.run_case(study,'small')

    def test_analysis_recovers_exact_sample_ballistic_covariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); binary=root/'binary';binary.write_text('test binary')
            study=arc.prepare(binary,output=root/'study',cases=[('bare',16,4,6.25e-11,0)])
            meta=json.loads((study/'study.json').read_text())
            phase=np.load(study/'initial-0pc-n16.npz')['phase']
            final=arc.ballistic_phase(phase,meta['p0_mc'],arc.DURATION)
            p=meta['p0_mc']*final[:,[1,3,5]];p[:,2]+=meta['p0_mc']
            with h5py.File(study/'bare/case.h5','w') as output:
                g=output.create_group('Step#0');g.attrs['TIME']=arc.DURATION
                g.attrs['RefPartP']=[0,0,meta['p0_mc']]
                for i,name in enumerate(['x','y','z']):g[name]=final[:,[0,2,4]][:,i]
                for i,name in enumerate(['px','py','pz']):g[name]=p[:,i]
                for name in ['Ex','Ey','Ez','Bx','By','Bz']:g[name]=np.zeros(16)
                g['id']=np.arange(16);g['q']=np.full(16,arc.CHARGE/16)
            record=json.loads((study/'bare/provenance.json').read_text())
            record.update(status='completed',elapsed_seconds=0.)
            write_json(study/'bare/provenance.json',record)
            result=arc.analyze(study)
            self.assertLess(result.iloc[0]['exact_ballistic_normalized_error'],1e-12)
            self.assertTrue((study/'bare/analysis.json').exists())

if __name__=='__main__':unittest.main()
