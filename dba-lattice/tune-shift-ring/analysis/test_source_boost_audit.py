"""Regression checks of source boost algebra; no OPALX or PIC execution."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from source_boost_audit import (force_map,comoving_difference,singular_norm_2x2,
                                tight_norms,table_for_ring,write_study)


class SourceBoostAuditTests(unittest.TestCase):
    def test_map_matches_independent_electric_and_cross_product_composition(self):
        rng=np.random.default_rng(4101)
        for beta in [.0728541,.4,.8]:
            gamma=1/np.sqrt(1-beta*beta)
            for _ in range(20):
                n=rng.normal(size=3);n/=np.linalg.norm(n)
                w=.3*rng.normal(size=3);w/=max(1,np.linalg.norm(w)/.9)
                eprime=rng.normal(size=3)
                electric=gamma*eprime-(gamma-1)*np.dot(eprime,n)*n
                c_magnetic=gamma*np.cross(beta*n,eprime)
                force=electric+np.cross(w,c_magnetic)
                np.testing.assert_allclose(force_map(gamma,beta,n,w)@eprime,force,rtol=2e-15,atol=1e-15)

    def test_closed_form_and_zero_lag_reference(self):
        beta=.0728541;gamma=1/np.sqrt(1-beta*beta);u=np.array([0.,0.,1.])
        reference=force_map(gamma,beta,u,beta*u)
        # gamma*(1-beta²) and 1/gamma differ by one binary64 epsilon here;
        # allow two epsilons for the equivalent rounded expressions.
        np.testing.assert_allclose(reference,np.diag([1/gamma,1/gamma,1]),rtol=0,
                                   atol=2*np.finfo(float).eps)
        np.testing.assert_array_equal(comoving_difference(beta,0),np.zeros((3,3)))
        for angle in [1e-5,.001,.05,-.3]:
            n=np.array([np.sin(angle),0,np.cos(angle)])
            np.testing.assert_allclose(comoving_difference(beta,angle),
                force_map(gamma,beta,n,beta*u)-reference,rtol=1e-9,atol=4e-16)

    def test_invariant_norm_matches_svd_and_relative_bound_is_attained(self):
        rng=np.random.default_rng(4102)
        for _ in range(20):
            matrix=rng.normal(size=(2,2))
            self.assertAlmostEqual(singular_norm_2x2(matrix),np.linalg.norm(matrix,2),places=13)
        beta=.0728541;gamma=1/np.sqrt(1-beta*beta)
        reference=np.diag([1/gamma,1/gamma,1])
        for angle in [0,.0003,.003,.1]:
            difference=comoving_difference(beta,angle);norms=tight_norms(beta,angle)
            self.assertAlmostEqual(norms['force_operator_error'],np.linalg.norm(difference,2),delta=1e-16)
            relative=difference@np.linalg.inv(reference)
            _,values,vt=np.linalg.svd(relative)
            eprime=np.linalg.solve(reference,vt[0])
            attained=np.linalg.norm(difference@eprime)/np.linalg.norm(reference@eprime)
            self.assertAlmostEqual(norms['force_relative_error'],attained,delta=1e-16)

    def test_small_angle_linear_and_quadratic_orders(self):
        beta=.0728541;gamma=1/np.sqrt(1-beta*beta);theta=1e-5
        result=tight_norms(beta,theta)
        self.assertAlmostEqual(result['force_operator_error']/theta,gamma-1,delta=1e-10)
        self.assertAlmostEqual(result['force_relative_error']/theta,gamma*(gamma-1),delta=1e-10)
        self.assertAlmostEqual(result['transverse_to_transverse_relative_error']/theta**2,
                               .5*gamma*gamma*beta*beta,delta=1e-12)

    def test_table_and_written_assumptions_are_reproducible(self):
        table=table_for_ring(.0728541,2,[62.5e-12,125e-12,400e-12,1e-9])
        np.testing.assert_allclose(table.h_ps,[62.5,125,400,1000])
        self.assertTrue(np.all(np.diff(table.force_relative_error)>0))
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):write_study(output,.0728541,2.)
            saved=pd.read_csv(output/'composition-only-bounds.csv')
            np.testing.assert_allclose(saved.force_relative_error,table.force_relative_error,rtol=2e-14)
            self.assertIn('PIC',json.loads((output/'assumptions.json').read_text())['excluded'])


if __name__=='__main__':unittest.main()
