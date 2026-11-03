"""Independent mathematical-oracle tests; no OPALX execution."""
import unittest
import numpy as np
from audit_edge_average_math import magnetic_generator,switched_magnetic_error,reversal_error


class EdgeAverageAuditTests(unittest.TestCase):
    def test_magnetic_generator_sign_and_norm(self):
        p=np.array([.2,.3,1.]);b=np.array([.7,0,.1]);gamma=np.sqrt(1+p@p)
        matrix=magnetic_generator(b,gamma)
        np.testing.assert_allclose(matrix@p,np.cross(p,b)/gamma,atol=1e-16,rtol=0)
        np.testing.assert_array_equal(matrix.T,-matrix)

    def test_noncommuting_known_switch_has_quadratic_impulse_error(self):
        p=np.array([.2,.3,1.]);s=np.array([.7,0,.1]);b=np.array([0,.8,0])
        errors=[];remainders=[]
        for dt in [.02,.01,.005]:
            result=switched_magnetic_error(p,s,b,dt,.3)
            errors.append(np.linalg.norm(result['error']))
            remainders.append(np.linalg.norm(result['error']-result['leading']))
        self.assertTrue(np.all(np.array(errors[:-1])/errors[1:]>3.9))
        self.assertTrue(np.all(np.array(errors[:-1])/errors[1:]<4.1))
        self.assertTrue(np.all(np.array(remainders[:-1])/remainders[1:]>7.9))

    def test_original_weighted_step_reverses_while_endpoint_variant_does_not(self):
        p=np.array([.2,.05,1.]);r=np.array([.1,-.03,-.015]);s=np.array([.01,.03,.02]);b=np.array([0,.8,0])
        original=reversal_error(r,p,.08,'weighted',lambda x: -.1*x,s,b)
        corrected=reversal_error(r,p,.08,'weighted_moment',lambda x: -.1*x,s,b)
        self.assertLess(max(original),1e-14)
        self.assertGreater(max(corrected),1e-8)


if __name__=='__main__':unittest.main()
