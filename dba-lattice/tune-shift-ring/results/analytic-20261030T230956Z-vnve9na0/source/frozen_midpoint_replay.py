"""Standalone frozen midpoint E=A r, constant self B, boundary-resolved Boris.

No OPALX execution or production/API change. This prescribed self-field model
is not PIC and does not validate source-velocity or deposition effects.
Units are dimensionless as in edge_average_prototype.
"""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from edge_average_prototype import velocity,boris,exact_slab


def cross_matrix(vector):
    x,y,z=vector
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def inside(position,length,normal):
    return position[2]>=0 and np.dot(normal,position)<length


def rhs_jacobian(state,matrix,magnetic):
    """Exact smooth-region Lorentz vector field and 6x6 derivative."""
    r,p=state[:3],state[3:];gamma=np.sqrt(1+np.dot(p,p));v=p/gamma
    dv=(np.eye(3)-np.outer(v,v))/gamma
    jac=np.zeros((6,6));jac[:3,3:]=dv;jac[3:,:3]=matrix
    jac[3:,3:]=-cross_matrix(magnetic)@dv
    return np.r_[v,matrix@r+np.cross(v,magnetic)],jac


def oracle(state,duration,matrix,self_b,body_b,length,normal,rtol=2e-12):
    """Event-resolved DOP853 state+tangent oracle with exact saltation matrices."""
    state=np.asarray(state,dtype=float);matrix=np.asarray(matrix);normal=np.asarray(normal)
    augmented=np.r_[state,np.eye(6).ravel()];time=0.;regions=0;evals=0
    region=0 if state[2]<0 else (1 if np.dot(normal,state[:3])<length else 2)
    for _ in range(3):
        magnetic=np.asarray(self_b)+(np.asarray(body_b) if region==1 else 0.)
        def rhs(t,y):
            flow,jac=rhs_jacobian(y[:6],matrix,magnetic)
            return np.r_[flow,(jac@y[6:].reshape(6,6)).ravel()]
        def surface(t,y):return y[2] if region==0 else np.dot(normal,y[:3])-length
        surface.terminal=True;surface.direction=1
        result=solve_ivp(rhs,(time,duration),augmented,method='DOP853',rtol=rtol,atol=rtol*.01,
                         events=surface if region<2 else None)
        if not result.success:raise RuntimeError(result.message)
        augmented=result.y[:,-1];time=float(result.t[-1]);evals+=result.nfev
        if time>=duration-1e-14:
            return augmented[:6],augmented[6:].reshape(6,6),dict(rhs_evaluations=evals,surfaces=regions)
        state=augmented[:6];event_normal=np.r_[[0.,0.,1.] if region==0 else normal,np.zeros(3)]
        before=rhs_jacobian(state,matrix,magnetic)[0]
        after_b=np.asarray(self_b)+(np.asarray(body_b) if region==0 else 0.)
        after=rhs_jacobian(state,matrix,after_b)[0]
        denominator=np.dot(event_normal,before)
        if denominator<=0:raise ValueError('Oracle requires transverse forward crossing')
        saltation=np.eye(6)+np.outer(after-before,event_normal)/denominator
        augmented[6:]=(saltation@augmented[6:].reshape(6,6)).ravel()
        region+=1;regions+=1
    raise RuntimeError('Oracle did not complete')


def replay_interval(r,p,dt,electric,self_b,body_b,length,normal,counts,depth=0):
    """Dyadic DKD with live point-sampled external B, fixed prescribed self E/B.

    Start/midpoint/end support changes trigger subdivision. Benchmark h and body
    length preclude skipping the whole body. A 64-epsilon displacement floor
    bounds face resolution; no self-field reevaluation occurs on subdivision.
    """
    counts['trials']+=1;counts['maximum_depth']=max(depth,counts['maximum_depth'])
    mid=r+.5*dt*velocity(p)
    magnetic=self_b+(body_b if inside(mid,length,normal) else 0.)
    final_p=boris(p,electric,magnetic,dt);final_r=mid+.5*dt*velocity(final_p)
    scale=max(1.,np.linalg.norm(r),np.linalg.norm(final_r))
    speed=max(np.linalg.norm(velocity(p)),np.linalg.norm(velocity(final_p)))
    floor=64*np.finfo(float).eps*scale/max(speed,np.finfo(float).tiny)
    support=(inside(r,length,normal),inside(mid,length,normal),inside(final_r,length,normal))
    crossed=support[0]!=support[1] or support[1]!=support[2]
    if crossed and dt>floor:
        counts['rejected']+=1
        r,p=replay_interval(r,p,.5*dt,electric,self_b,body_b,length,normal,counts,depth+1)
        return replay_interval(r,p,.5*dt,electric,self_b,body_b,length,normal,counts,depth+1)
    counts['accepted']+=1
    if crossed:counts['floor_crossings']+=1
    return final_r,final_p


def energy(state,matrix):
    """Static symmetric linear electric field invariant gamma-rᵀ A r/2."""
    return np.sqrt(1+np.dot(state[3:],state[3:]))-.5*state[:3]@matrix@state[:3]


def track(state,duration,dt,matrix,self_b,body_b,length,normal):
    """One prescribed-self-field evaluation per macro step, then local replay."""
    if dt<=0:raise ValueError('Positive DT required')
    n=round(duration/dt)
    if not np.isclose(n*dt,duration,rtol=0,atol=1e-14):raise ValueError('DT must divide duration')
    matrix=np.asarray(matrix);self_b=np.asarray(self_b);body_b=np.asarray(body_b);normal=np.asarray(normal)
    state=np.asarray(state,dtype=float);r=state[:3].copy();p=state[3:].copy()
    counts=dict(macro_steps=n,self_evaluations=0,trials=0,rejected=0,accepted=0,maximum_depth=0,floor_crossings=0)
    initial_energy=energy(state,matrix);error=0.
    for _ in range(n):
        predictor=r+.5*dt*velocity(p)
        electric=matrix@predictor;counts['self_evaluations']+=1
        r,p=replay_interval(r,p,dt,electric,self_b,body_b,length,normal,counts)
        error=max(error,abs(energy(np.r_[r,p],matrix)-initial_energy))
    counts['maximum_energy_error']=error
    return np.r_[r,p],counts


def numerical_jacobian(function,state,epsilon=2e-5):
    """Central initial-condition differences; caller checks epsilon refinement."""
    state=np.asarray(state);jac=np.empty((6,6))
    for column in range(6):
        delta=np.zeros(6);delta[column]=epsilon
        jac[:,column]=(function(state+delta)-function(state-delta))/(2*epsilon)
    return jac


def study(output=None):
    root=Path(__file__).resolve().parents[1]
    output=Path(output) if output else Path(tempfile.mkdtemp(prefix='frozen-midpoint-replay-',dir=root/'results'))
    output.mkdir(parents=True,exist_ok=True)
    matrix=np.array([[.03,0.,.004],[0.,.02,0.],[.004,0.,.008]])
    self_b=np.array([.013,.025,-.017]);body_b=np.array([0.,.8,0.]);length=.4;duration=2.56
    state=np.array([.012,-.006,-.173,0.,0.,1.]);alpha=np.arcsin(.8*length)
    geometries={'slab':np.array([0.,0.,1.]),'normal_exit':np.array([-np.sin(alpha),0.,np.cos(alpha)])}
    rows=[];oracles={};detail={}
    for geometry,normal in geometries.items():
        reference,jac,info=oracle(state,duration,matrix,self_b,body_b,length,normal)
        tight,tightjac,tightinfo=oracle(state,duration,matrix,self_b,body_b,length,normal,rtol=2e-13)
        fd=numerical_jacobian(lambda x:oracle(x,duration,matrix,self_b,body_b,length,normal,rtol=2e-13)[0],state)
        oracles[geometry]=dict(state=reference.tolist(),jacobian=jac.tolist(),
            ten_times_tighter_state_difference=float(np.linalg.norm(tight-reference)),
            ten_times_tighter_jacobian_difference=float(np.linalg.norm(tightjac-jac)),
            saltation_vs_finite_difference_norm=float(np.linalg.norm(fd-tightjac)),
            energy_error=float(energy(reference,matrix)-energy(state,matrix)),**info)
        print(geometry,'oracle',oracles[geometry]['ten_times_tighter_jacobian_difference'],flush=True)
        for dt in [.08,.04,.02,.01,.005]:
            actual,counts=track(state,duration,dt,matrix,self_b,body_b,length,normal)
            fn=lambda x:track(x,duration,dt,matrix,self_b,body_b,length,normal)[0]
            actualjac=numerical_jacobian(fn,state)
            finejac=numerical_jacobian(fn,state,epsilon=1e-5)
            error=actualjac-tightjac;key=f'{geometry}-{dt:g}'
            detail[key]=dict(state=actual.tolist(),jacobian=actualjac.tolist(),jacobian_error=error.tolist(),counts=counts)
            row=dict(geometry=geometry,dt=dt,position_error=float(np.linalg.norm(actual[:3]-tight[:3])),
                momentum_error=float(np.linalg.norm(actual[3:]-tight[3:])),jacobian_error=float(np.linalg.norm(error)),
                position_jacobian_error=float(np.linalg.norm(error[:3])),momentum_jacobian_error=float(np.linalg.norm(error[3:])),
                finite_difference_refinement=float(np.linalg.norm(actualjac-finejac)),**counts)
            rows.append(row);print(key,row['position_error'],row['momentum_error'],row['jacobian_error'],counts['trials'],flush=True)
    bare=[];initial=np.array([0.,0.,-.173,0.,0.,1.]);exact=np.r_[exact_slab(initial[:3],1.,duration,length,.8)]
    for geometry,normal in geometries.items():
        for dt in [.08,.04,.02,.01,.005]:
            actual,counts=track(initial,duration,dt,np.zeros((3,3)),np.zeros(3),body_b,length,normal)
            bare.append(dict(geometry=geometry,dt=dt,position_error=float(np.linalg.norm(actual[:3]-exact[:3])),
                momentum_error=float(np.linalg.norm(actual[3:]-exact[3:])),**counts))
    pd.DataFrame(rows).to_csv(output/'convergence.csv',index=False);pd.DataFrame(bare).to_csv(output/'bare.csv',index=False)
    summary=dict(scope='Prescribed linear self-field freeze only; not PIC, no source-velocity validation or OPALX execution.',
        matrix=matrix.tolist(),self_b=self_b.tolist(),body_b=body_b.tolist(),length=length,duration=duration,initial=state.tolist(),
        oracles=oracles,cases=detail,bare=bare)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(output,flush=True);return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    study(parser.parse_args().output)
