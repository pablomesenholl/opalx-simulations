"""Bounded independent audit of averaging/endpoint-moment prototype mathematics.

No production OPALX source, integrator, input API or PIC solve is changed.
The switched magnetic oracle uses exact matrix exponentials at a known switch
time; this deliberately removes residence-prediction error from the comparison.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy.linalg import expm
import edge_average_prototype as edge


def magnetic_generator(magnetic, gamma):
    """Matrix for dp/dt = p cross B / gamma in the prototype's units."""
    x,y,z=np.asarray(magnetic)
    return np.array([[0,z,-y],[-z,0,x],[y,-x,0]])/gamma


def switched_magnetic_error(momentum, self_b, body_b, dt, entry_fraction):
    """Compare exact two-field time ordering with one averaged combined kick."""
    gamma=np.sqrt(1+np.dot(momentum,momentum));tau=entry_fraction*dt
    a0=magnetic_generator(self_b,gamma);a1=magnetic_generator(body_b,gamma)
    exact=expm((dt-tau)*(a0+a1))@expm(tau*a0)@momentum
    averaged=edge.boris(momentum,np.zeros(3),self_b+(1-entry_fraction)*body_b,dt)
    leading=.5*tau*(dt-tau)*(a1@a0-a0@a1)@momentum
    return dict(error=exact-averaged,leading=leading,
                exact_momentum=exact,averaged_momentum=averaged)


def reversal_error(position,momentum,dt,method,self_e,self_b,body_b):
    """Time-reverse a step: p and magnetic fields reverse, electric field does not.

    This is an algebraic reversibility check, not an accuracy or symplecticity
    proof. A spatial electric field is sampled once at each algorithmic midpoint.
    """
    forward=edge.step(position,momentum,dt,.4,body_b,self_e=self_e,self_b=self_b,method=method)
    backward=edge.step(forward[0],-forward[1],dt,.4,-body_b,self_e=self_e,self_b=-self_b,method=method)
    return np.linalg.norm(backward[0]-position),np.linalg.norm(backward[1]+momentum)


def study(output):
    """Run known-switch and reversal diagnostics only; no integrator search."""
    output.mkdir(parents=True,exist_ok=True)
    for filename in ['audit_edge_average_math.py','edge_average_prototype.py']:
        shutil.copy2(Path(__file__).with_name(filename),output/filename)
    steps=[.08,.04,.02,.01,.005]
    p=np.array([.2,.3,1.]);self_b=np.array([.7,0,.1]);body_b=np.array([0,.8,0])
    rows=[]
    for dt in steps:
        result=switched_magnetic_error(p,self_b,body_b,dt,.3)
        rows.append(dict(dt=dt,error_norm=np.linalg.norm(result['error']),
                         leading_commutator_norm=np.linalg.norm(result['leading']),
                         remainder_norm=np.linalg.norm(result['error']-result['leading'])))
    pd.DataFrame(rows).to_csv(output/'known-switch-noncommuting-fields.csv',index=False)
    reversals=[];p=np.array([.2,.05,1.]);self_b=np.array([.01,.03,.02])
    for method in ['weighted','weighted_moment']:
        for dt in steps:
            errors=[]
            for fraction in [.2,.5,.8]:
                r=np.array([.1,-.03,-fraction*dt*edge.velocity(p)[2]])
                errors.append(reversal_error(r,p,dt,method,lambda x: -.1*x,self_b,body_b))
            errors=np.array(errors)
            reversals.append(dict(method=method,dt=dt,position_max=errors[:,0].max(),
                                  momentum_max=errors[:,1].max()))
    pd.DataFrame(reversals).to_csv(output/'reversal.csv',index=False)
    (output/'scope.json').write_text(json.dumps(dict(
        scope='Independent dimensionless mathematics only; not production validation.',
        known_switch='Exact relativistic magnetic rotations; constant momentum norm fixes gamma.',
        reversal='Oblique entrance with spatial E=-0.1 r, body By=0.8, self B=(0.01,0.03,0.02).',
        limitation='The scalar plane support and supplied midpoint field omit curved geometry, time-dependent RF, self-consistent PIC and device implementation.'),indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(reversals).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    study(parser.parse_args().output)
