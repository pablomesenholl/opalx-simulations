"""Unit tests use synthetic H5; no OPALX execution or fitted-tune selection."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from reduce_temporal_monitors import fixed_ids, full_diagnostics, reduce, subset_rows, unique_snapshot_keys


class MonitorReductionTest(unittest.TestCase):
    def test_fixed_id_sample_is_order_independent(self):
        ids=np.arange(5000)
        chosen=fixed_ids(ids,1024,123)
        np.testing.assert_array_equal(chosen,fixed_ids(ids[::-1],1024,123))
        self.assertEqual(len(chosen),1024)
        self.assertFalse(np.array_equal(chosen,ids[:1024]))
        np.testing.assert_array_equal(fixed_ids(ids[:4],1024),ids[:4])
        with self.assertRaises(ValueError): fixed_ids([1,1],1)

    def test_missing_ids_are_explicit(self):
        phase=np.arange(18).reshape(3,6)
        observed,present=subset_rows(np.array([2,4,8]),phase,np.array([0,2,7,8,9]))
        np.testing.assert_array_equal(present,[False,True,False,True,False])
        np.testing.assert_array_equal(observed[present],phase[[0,2]])
        self.assertTrue(np.isnan(observed[~present]).all())

    def test_full_diagnostics_uses_all_particles(self):
        rng=np.random.default_rng(717)
        phase=rng.normal(size=(1000,6))
        result=full_diagnostics(phase,np.eye(6))
        self.assertEqual(result['count'],1000)
        np.testing.assert_allclose(result['centroid'],phase.mean(axis=0))
        np.testing.assert_allclose(result['covariance'],np.cov(phase,rowvar=False,bias=True))
        self.assertEqual(result['density_measured']['count'],1000)
        self.assertEqual(result['density_model']['count'],1000)

    def test_terminal_duplicate_keeps_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            with h5py.File(Path(tmp)/'x.h5','w') as data:
                for key,time in [('Step#0',1e-6),('Step#1',2e-6),('Step#2',np.nextafter(2e-6,np.inf))]:
                    data.create_group(key).attrs['TIME']=[time]
                self.assertEqual(unique_snapshot_keys(data),(['Step#0','Step#2'],['Step#1']))
                data.create_group('Step#3').attrs['TIME']=[1e-6]
                with self.assertRaises(ValueError): unique_snapshot_keys(data)

    def test_offline_reduction_retains_raw_endpoints_and_full_moments(self):
        rng=np.random.default_rng(4)
        phase=rng.normal(size=(32,6))*.001
        sections=[dict(centre_m=[0,0,0],theta_rad=0) for _ in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for cell in range(6):
                with h5py.File(root/f'monitor-{cell+1}.h5','w') as data:
                    for step in range(3):
                        group=data.create_group(f'Step#{step}')
                        group.attrs['TIME']=[float(cell+6*min(step,1)+1)]
                        group.attrs['RefPartR']=[0.,0.,0.]
                        group.attrs['RefPartP']=[0.,0.,.073]
                        group['id']=np.arange(32)
                        group['time']=np.full(32,float(cell+6*min(step,1)+1)+.1)
                        for column,name in enumerate(['x','px','y','py','z','pz']):
                            group[name]=phase[:,column] if column%2==0 else .073*(phase[:,column]+int(column==5))
            manifest=reduce(root,sections,.073,root/'reduced',count=8,seed=3)
            self.assertEqual(manifest['passages'],12)
            self.assertFalse(manifest['raw_inputs_deleted'])
            self.assertTrue(all(path.exists() for path in root.glob('monitor-*.h5')))
            observed=np.load(root/'reduced/section-observations.npz')
            self.assertEqual(observed['phase'].shape,(12,8,6))
            np.testing.assert_allclose(observed['centroid'],np.tile(phase.mean(axis=0),(12,1)),atol=1e-16)
            with h5py.File(root/'reduced/raw-first-last.h5','r') as data:
                self.assertEqual(set(data['monitor-1']),{'Step#0','Step#2'})
                self.assertEqual(len(data['monitor-1/Step#0/id']),32)

if __name__=='__main__': unittest.main()
