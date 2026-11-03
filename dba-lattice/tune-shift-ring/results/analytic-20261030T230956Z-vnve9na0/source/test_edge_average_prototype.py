"""Pure mathematical prototype tests; no OPALX execution or production changes."""
import unittest
import numpy as np
import edge_average_prototype as edge


class EdgeAverageTests(unittest.TestCase):
    def test_straight_residence_endpoints_and_reversal(self):
        for a,b,expected in [(-1,-.1,0),(.1,.9,1),(-1,1,.5),(-1,2,1/3),(.5,1.5,.5),(1,2,0),(0,1,1)]:
            self.assertAlmostEqual(edge.slab_fraction(a,b,1),expected)
            self.assertAlmostEqual(edge.slab_fraction(b,a,1),expected)
        self.assertEqual(edge.slab_fraction(0,0,1),1)
        self.assertEqual(edge.slab_fraction(1,1,1),0)
        with self.assertRaises(ValueError):edge.slab_fraction(0,1,0)

    def test_active_time_centroid_and_endpoint_moment_variant(self):
        self.assertAlmostEqual(edge.residence_centroid(-.3,.2,.7,1.,1.),.65)
        self.assertAlmostEqual(edge.residence_centroid(.7,.2,-.3,1.,1.),.35)
        self.assertEqual(edge.residence_centroid(2.,2.,2.,1.,1.),.5)
        with self.assertRaises(ValueError):edge.residence_centroid(0,1,2,1,0)
        p=np.array([0.,0.,1.]);errors=[]
        for dt in [.08,.04,.02]:
            r=np.array([0.,0.,-.3*dt*edge.velocity(p)[2]])
            original=edge.step(r,p,dt,.4,[0,.8,0])
            corrected=edge.step(r,p,dt,.4,[0,.8,0],method='weighted_moment')
            exact=edge.exact_slab(r,1.,dt,.4,.8)
            np.testing.assert_array_equal(corrected[1],original[1])
            self.assertAlmostEqual(corrected[2]['active_time_centroid'],.65*dt,delta=1e-16)
            errors.append(np.linalg.norm(corrected[0]-exact[0]))
        # Cancels the O(h²) entrance position defect, leaving the third-order
        # single-body Boris phase error. This does not establish full-slab order.
        self.assertTrue(np.all(np.array(errors[:-1])/errors[1:]>7.9))

    def test_sector_clipping_and_normal_exit_geometry(self):
        for a,b in [(-1.,2.),(.1,.9),(.5,1.5),(1.,2.),(0.,1.)]:
            lo,hi=edge.sector_interval([0,0,a],[0,0,b],1.,[0,0,1])
            self.assertAlmostEqual(hi-lo,edge.slab_fraction(a,b,1.))
        self.assertEqual(edge.sector_interval([0,0,1],[1,0,1],1,[0,0,1]),(0.,0.))
        self.assertAlmostEqual(edge.sector_centroid([0,0,-.3],[0,0,.2],[0,0,.7],1.,1.,[0,0,1]),.65)
        b=.8;length=.4;alpha=np.arcsin(b*length)
        normal=np.array([-np.sin(alpha),0.,np.cos(alpha)])
        gamma=np.sqrt(2.);time=.173*gamma+gamma*alpha/b
        r,p=edge.exact_slab([0,0,-.173],1.,time,length,b)
        self.assertAlmostEqual(np.dot(normal,r),length,delta=1e-15)
        np.testing.assert_allclose(p,normal,rtol=1e-15,atol=1e-15)
        with self.assertRaises(ValueError):edge.sector_interval([0,0,0],[0,0,1],0,normal)

    def test_boris_magnetic_norm_and_electric_impulse(self):
        p=np.array([.3,-.1,1.]);b=np.array([.01,.8,.05]);dt=.07
        actual=edge.boris(p,[0,0,0],b,dt)
        self.assertAlmostEqual(np.dot(actual,actual),np.dot(p,p),delta=5e-16)
        electric=np.array([.1,.3,-.04])
        np.testing.assert_allclose(edge.boris(p,electric,[0,0,0],dt),p+dt*electric,rtol=1e-15,atol=1e-16)
        self.assertLess(np.linalg.norm(edge.velocity(p)),1)

    def test_full_body_preserves_combined_midpoint_boris(self):
        r=np.array([.1,.2,.5]);p=np.array([.05,.01,1.])
        kwargs=dict(self_e=[.03,.02,.04],self_b=[.01,.02,.03])
        weighted=edge.step(r,p,.03,1,[0,.8,0],**kwargs)
        original=edge.step(r,p,.03,1,[0,.8,0],method='midpoint',**kwargs)
        np.testing.assert_array_equal(weighted[0],original[0]);np.testing.assert_array_equal(weighted[1],original[1])
        self.assertEqual(weighted[2]['fraction'],1)
        with self.assertRaises(ValueError):edge.track(r,p,.1,.03,1,[0,.8,0])

    def test_single_entrance_exact_momentum_and_position_orders(self):
        p=np.array([0.,0.,1.]);momentum=[];position=[]
        for dt in [.08,.04,.02]:
            r=np.array([0.,0.,-.3*dt*edge.velocity(p)[2]])
            actual=edge.step(r,p,dt,.4,[0,.8,0])
            exact=edge.exact_slab(r,1.,dt,.4,.8)
            momentum.append(np.linalg.norm(actual[1]-exact[1]));position.append(np.linalg.norm(actual[0]-exact[0]))
            self.assertAlmostEqual(actual[2]['fraction'],.7,delta=5e-16)
        self.assertTrue(np.all(np.array(momentum[:-1])/momentum[1:]>7.9))
        self.assertTrue(np.all(np.array(position[:-1])/position[1:]>3.9))

    def test_vanishing_surface_displacement_has_no_finite_impulse_jump(self):
        p=np.array([0.,0.,1.]);dt=.1;z=-.5*dt*edge.velocity(p)[2];eps=1e-10
        jumps={}
        for method in ['midpoint','weighted']:
            hi=edge.step([0,0,z+eps],p,dt,.4,[0,.8,0],method=method)
            lo=edge.step([0,0,z-eps],p,dt,.4,[0,.8,0],method=method)
            jumps[method]=np.linalg.norm(hi[1]-lo[1])
        self.assertGreater(jumps['midpoint'],.05)
        self.assertLess(jumps['weighted'],2e-10)

    def test_single_midpoint_self_field_and_additive_combined_kick(self):
        r=np.array([0.,0.,-.01]);p=np.array([0.,0.,1.]);dt=.08
        calls=[]
        def electric(midpoint):calls.append(midpoint.copy());return [.03,.02,.01]
        self_b=np.array([.01,.02,.03]);body=np.array([0,.8,0])
        actual=edge.step(r,p,dt,.4,body,self_e=electric,self_b=self_b)
        self.assertEqual(len(calls),1)
        np.testing.assert_array_equal(calls[0],r+.5*dt*edge.velocity(p))
        self.assertGreater(actual[2]['iterations'],1)
        np.testing.assert_array_equal(actual[2]['magnetic'],self_b+actual[2]['fraction']*body)
        np.testing.assert_array_equal(actual[1],edge.boris(p,[.03,.02,.01],actual[2]['magnetic'],dt))

    def test_finite_slab_state_refinement_and_derivative_resolution(self):
        errors=[];p=np.array([0.,0.,1.]);duration=2.56
        for dt in [.04,.01]:
            position=[];momentum=[]
            for z in [-.2073,-.1914,-.1755,-.1596,-.1437]:
                r=np.array([0.,0.,z]);actual=edge.track(r,p,duration,dt,.4,[0,.8,0])
                exact=edge.exact_slab(r,1.,duration,.4,.8)
                position.append(np.linalg.norm(actual[0]-exact[0]))
                momentum.append(np.linalg.norm(actual[1]-exact[1]))
                self.assertAlmostEqual(np.linalg.norm(actual[1]),1.,delta=3e-14)
            errors.append([max(position),max(momentum)])
        # A fourfold timestep refinement must resolve both state components;
        # this is a bounded regression, not a proof for arbitrary geometries.
        self.assertTrue(np.all(np.asarray(errors[1])<np.asarray(errors[0])/8))
        r=np.array([0.,0.,-.173]);dt=.01;derivatives=[]
        # Resolve the derivative without subtracting nearly identical long
        # trajectories at epsilon=1e-7; repeated drift rounding is amplified by
        # that quotient. These offsets remain much smaller than this step.
        for eps in [2e-5,1e-5]:
            hi=np.r_[edge.track(r+[0,0,eps],p,duration,dt,.4,[0,.8,0])[:2]]
            lo=np.r_[edge.track(r-[0,0,eps],p,duration,dt,.4,[0,.8,0])[:2]]
            derivatives.append((hi-lo)/(2*eps))
        np.testing.assert_allclose(derivatives[0],derivatives[1],rtol=0,atol=2e-8)
        self.assertGreater(np.linalg.norm(derivatives[1]-edge.exact_full_slab_shift_derivative(1.,.4,.8)),1e-4)

    def test_event_reference_matches_analytic_slab_and_shift_derivative(self):
        r=np.array([0.,0.,-.173]);p=np.array([0.,0.,1.]);duration=2.56
        actual=edge.uniform_sc_reference(r,p,duration,.4,[0,.8,0],[0,0,0],[0,0,0])
        exact=edge.exact_slab(r,1.,duration,.4,.8)
        np.testing.assert_allclose(actual,exact,rtol=2e-11,atol=2e-12)
        epsilon=1e-5
        hi=np.r_[edge.exact_slab(r+[0,0,epsilon],1.,duration,.4,.8)]
        lo=np.r_[edge.exact_slab(r-[0,0,epsilon],1.,duration,.4,.8)]
        np.testing.assert_allclose((hi-lo)/(2*epsilon),edge.exact_full_slab_shift_derivative(1.,.4,.8),rtol=1e-10,atol=1e-12)
        with self.assertRaises(ValueError):edge.exact_slab(r,1.,duration,2.,.8)

if __name__=='__main__':unittest.main()
