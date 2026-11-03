"""Pure numerical tests; no OPALX execution."""
import unittest
import numpy as np
import frozen_midpoint_replay as replay
from edge_average_prototype import exact_slab

class FrozenReplayTests(unittest.TestCase):
    def setUp(self):
        self.a=np.array([[.03,0.,.004],[0.,.02,0.],[.004,0.,.008]])
        self.bs=np.array([.013,.025,-.017]);self.bb=np.array([0.,.8,0.])
        self.normal=np.array([0.,0.,1.]);self.state=np.array([.012,-.006,-.173,0.,0.,1.])
    def test_exact_rhs_jacobian(self):
        state=self.state+np.array([0,0,0,.03,-.02,0])
        flow,jac=replay.rhs_jacobian(state,self.a,self.bs+self.bb)
        finite=replay.numerical_jacobian(lambda s:replay.rhs_jacobian(s,self.a,self.bs+self.bb)[0],state,1e-5)
        np.testing.assert_allclose(jac,finite,rtol=1e-9,atol=2e-11)
        np.testing.assert_allclose(replay.cross_matrix(self.bs)@state[3:],np.cross(self.bs,state[3:]),rtol=1e-15,atol=1e-16)
        gradient=np.r_[-self.a@state[:3],replay.velocity(state[3:])]
        self.assertAlmostEqual(np.dot(gradient,flow),0.,delta=1e-16)
    def test_support_entrance_and_normal_exit(self):
        self.assertTrue(replay.inside(np.array([0.,0.,0.]),.4,self.normal))
        self.assertFalse(replay.inside(np.array([0.,0.,.4]),.4,self.normal))
        n=np.array([-.32,0.,np.sqrt(1-.32**2)])
        self.assertTrue(replay.inside(np.array([1.,0.,.5]),.4,n))
        self.assertFalse(replay.inside(np.array([-1.,0.,.5]),.4,n))
    def test_oracle_analytic_bare_state_and_normal_shift(self):
        state=np.array([0.,0.,-.173,0.,0.,1.])
        actual,jac,info=replay.oracle(state,2.56,np.zeros((3,3)),np.zeros(3),self.bb,.4,self.normal)
        exact=np.r_[exact_slab(state[:3],1.,2.56,.4,.8)]
        np.testing.assert_allclose(actual,exact,rtol=2e-11,atol=2e-12)
        expected=np.array([-.32,0.,np.sqrt(1-.32**2),0.,0.,0.])
        np.testing.assert_allclose(jac[:,2],expected,rtol=2e-10,atol=2e-11)
        self.assertEqual(info['surfaces'],2)
    def test_saltation_full_jacobian_and_tighter_oracle(self):
        args=(2.56,self.a,self.bs,self.bb,.4,self.normal)
        result,jac,_=replay.oracle(self.state,*args)
        tight,tightjac,_=replay.oracle(self.state,*args,rtol=2e-13)
        self.assertLess(np.linalg.norm(result-tight),2e-11)
        self.assertLess(np.linalg.norm(jac-tightjac),2e-10)
        finite=replay.numerical_jacobian(lambda s:replay.oracle(s,*args,rtol=2e-13)[0],self.state)
        self.assertLess(np.linalg.norm(finite-tightjac),2e-8)
        self.assertLess(abs(replay.energy(result,self.a)-replay.energy(self.state,self.a)),2e-11)
    def test_replay_resolves_edges_with_single_macro_self_evaluation(self):
        initial=np.array([0.,0.,-.02,0.,0.,1.])
        final,counts=replay.track(initial,.08,.08,self.a,self.bs,self.bb,.4,self.normal)
        self.assertTrue(np.isfinite(final).all());self.assertEqual(counts['self_evaluations'],1)
        self.assertGreater(counts['rejected'],20);self.assertGreater(counts['maximum_depth'],20)
        self.assertGreater(counts['floor_crossings'],0)
        self.assertEqual(counts['trials'],counts['accepted']+counts['rejected'])
        self.assertEqual(counts['accepted'],counts['macro_steps']+counts['rejected'])
    def test_bare_energy_and_refinement(self):
        initial=np.array([0.,0.,-.173,0.,0.,1.]);exact=np.r_[exact_slab(initial[:3],1.,2.56,.4,.8)]
        errors=[]
        for dt in [.04,.01]:
            final,counts=replay.track(initial,2.56,dt,np.zeros((3,3)),np.zeros(3),self.bb,.4,self.normal)
            self.assertLess(np.linalg.norm(final[3:]-exact[3:]),2e-12)
            self.assertLess(counts['maximum_energy_error'],2e-14)
            errors.append(np.linalg.norm(final[:3]-exact[:3]))
        self.assertLess(errors[1],errors[0]/8)
    def test_ballistic_flow_and_input_clock_guard(self):
        initial=np.array([.1,.2,.1,.03,-.02,1.]);t=.08
        final,counts=replay.track(initial,t,.02,np.zeros((3,3)),np.zeros(3),np.zeros(3),.4,self.normal)
        expected=initial.copy();expected[:3]+=t*replay.velocity(initial[3:])
        np.testing.assert_allclose(final,expected,rtol=1e-15,atol=1e-16)
        self.assertEqual(counts['self_evaluations'],4);self.assertEqual(counts['rejected'],0)
        with self.assertRaises(ValueError):replay.track(initial,t,.03,self.a,self.bs,self.bb,.4,self.normal)
        with self.assertRaises(ValueError):replay.track(initial,t,0,self.a,self.bs,self.bb,.4,self.normal)

if __name__=='__main__':unittest.main()
