"""Exact composition-only source-direction sensitivity; no OPALX execution.

E' is held fixed in common Cartesian axes. Source gamma and speed are fixed.
The table uses a co-moving test velocity beta*c*u; general maps accept any test
velocity in units c. No rest-frame stretch, Poisson solution or PIC noise enters.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy.constants import c as SPEED_OF_LIGHT


def force_map(gamma, source_beta, direction, test_beta_vector):
    """Return L with E+v_test cross B=L E', for the direct (+) source rule."""
    n=np.asarray(direction,dtype=float);w=np.asarray(test_beta_vector,dtype=float)
    return (gamma*(1-source_beta*np.dot(w,n))*np.eye(3)
            -(gamma-1)*np.outer(n,n)+gamma*source_beta*np.outer(n,w))


def comoving_difference(beta,angle):
    """Cancellation-safe exact Delta L for u=z, n=(sin theta,0,cos theta)."""
    gamma=1/np.sqrt(1-beta*beta)
    b=gamma-1;eta=gamma*beta*beta
    sine=np.sin(angle);cosine=np.cos(angle);one_minus_cosine=2*np.sin(angle/2)**2
    return np.array([[eta*one_minus_cosine-b*sine*sine,0,sine*(eta-b*cosine)],
                     [0,eta*one_minus_cosine,0],
                     [-b*sine*cosine,0,b*sine*sine]])


def singular_norm_2x2(matrix):
    """Exact two-by-two singular norm from trace and determinant invariants."""
    matrix=np.asarray(matrix);trace=np.sum(matrix*matrix);det=np.linalg.det(matrix)
    return np.sqrt(.5*(trace+np.sqrt(max(0.,trace*trace-4*det*det))))


def tight_norms(beta,angle):
    """Return attained operator and force-relative norms, not triangle bounds."""
    gamma=1/np.sqrt(1-beta*beta);difference=comoving_difference(beta,angle)
    block=difference[np.ix_([0,2],[0,2])]
    absolute=max(abs(difference[1,1]),singular_norm_2x2(block))
    relative=max(gamma*abs(difference[1,1]),
                 singular_norm_2x2(block@np.diag([gamma,1.])))
    transverse=gamma*max(abs(difference[0,0]),abs(difference[1,1]))
    return dict(force_operator_error=absolute,force_relative_error=relative,
                transverse_to_transverse_relative_error=transverse,
                electric_operator_error=(gamma-1)*abs(np.sin(angle)),
                c_magnetic_operator_error=2*gamma*beta*abs(np.sin(angle/2)))


def table_for_ring(beta,radius,timesteps):
    """Composition-only bounds for source-direction lag beta*c*h/(2*rho)."""
    rows=[]
    for h in timesteps:
        angle=beta*SPEED_OF_LIGHT*h/(2*radius)
        row=dict(h_ps=h*1e12,source_lag_rad=angle,source_lag_deg=np.degrees(angle))
        row.update(tight_norms(beta,angle));row['force_relative_ppm']=1e6*row['force_relative_error']
        rows.append(row)
    return pd.DataFrame(rows)


def write_study(output,beta,radius):
    """Write source snapshot, exact bounds and narrowly defined assumptions."""
    output.mkdir(parents=True,exist_ok=True)
    for name in ['source_boost_audit.py','test_source_boost_audit.py']:
        shutil.copy2(Path(__file__).with_name(name),output/name)
    table=table_for_ring(beta,radius,np.array([62.5,125,400,1000])*1e-12)
    table.to_csv(output/'composition-only-bounds.csv',index=False)
    (output/'assumptions.json').write_text(json.dumps(dict(beta=beta,gamma=1/np.sqrt(1-beta*beta),
        speed_m_per_s=beta*SPEED_OF_LIGHT,bend_radius_m=radius,
        test_velocity='beta*c times the true source direction',
        held_fixed='Common-axis E prime, gamma, source speed and test velocity',
        excluded='Rest stretch/frame/potential changes, midpoint trajectory error, energy changes, PIC discretization/noise and integrated tune errors',
        source='src/SpaceCharge/CartesianPIC3D/RelativisticFieldComposer.cpp:28-52; direct magneticSign=+1'),indent=2)+'\n')
    print(table.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    parser.add_argument('--beta',type=float,default=.0728541);parser.add_argument('--radius',type=float,default=2.)
    args=parser.parse_args();write_study(args.output,args.beta,args.radius)
