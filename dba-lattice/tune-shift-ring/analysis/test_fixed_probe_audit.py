"""Tests for independent fixed-step observation audits; no OPALX execution."""
import unittest
import numpy as np
import pandas as pd
from scipy.constants import c
from audit_fixed_step_observations import fitted_map_audit,reference_audit
from summarize_fixed_step_controls import aggregate_primary_modes


class FixedProbeAuditTests(unittest.TestCase):
    def test_summary_preserves_large_opposite_errors_despite_cancelling_mean(self):
        table=pd.DataFrame(dict(amplitude_m=[.001]*4,dt_s=[1e-10]*4,mode=['x']*4,
            probe=['x0+','x0-','x1+','x1-'],analytic_tune=[.75]*4,
            phase_tune_absolute=[.77,.73,.751,.749],
            spectral_tune_absolute=[.78,.72,.751,.749],
            spectral_coherent_fraction=[.5,.6,.8,.9],estimator_difference=[.01,.01,0,0]))
        result=aggregate_primary_modes(table).iloc[0]
        self.assertAlmostEqual(result.mean_phase_error,0)
        self.assertAlmostEqual(result.max_abs_phase_error,.02)
        self.assertAlmostEqual(result.max_abs_spectral_error,.03)
        self.assertAlmostEqual(result.max_opposite_probe_difference,.04)
        self.assertAlmostEqual(result.min_coherence,.5)

    def test_known_linear_modes_are_recovered_without_phase_fitting(self):
        tunes=np.array([.76,1.69,.07]);matrix=np.zeros((6,6))
        for i,tune in enumerate(tunes):
            angle=2*np.pi*tune/6;matrix[2*i:2*i+2,2*i:2*i+2]=[[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]
        initial=np.vstack((np.zeros(6),np.eye(6),-np.eye(6)))
        phase=np.array([initial@np.linalg.matrix_power(matrix,n).T for n in range(193)])
        result=fitted_map_audit(phase,np.eye(6))
        self.assertLess(result['relative_rms_prediction_error'],2e-14)
        np.testing.assert_allclose(np.sort(result['positive_mode_frequencies']),np.sort(tunes),atol=2e-14,rtol=0)
        np.testing.assert_allclose(result['multiplier_moduli'],1,atol=2e-14,rtol=0)

    def test_reference_error_is_distinct_from_particle_reference_error(self):
        times=np.arange(9)*1e-9;beta=.1;p0=beta/np.sqrt(1-beta*beta)
        position=np.column_stack((np.full(9,.02),np.zeros(9),beta*c*(times-4e-9)))
        momentum=np.tile([0,0,p0],(9,1));local=np.zeros((9,1,3));local[:,0,0]=np.arange(9)*1e-5
        result=reference_audit(dict(t=times,r=position,p=momentum,local_r=local),
                               [dict(theta_rad=0.,centre_m=[0.,0.,0.])],p0,8e-9)
        self.assertAlmostEqual(result['max_particle_reference_position_m'],8e-5)
        self.assertAlmostEqual(result['max_reference_abs_section_phase'][0],.02)
        self.assertEqual(result['accepted_reference_sections'],1)

if __name__=='__main__':unittest.main()
