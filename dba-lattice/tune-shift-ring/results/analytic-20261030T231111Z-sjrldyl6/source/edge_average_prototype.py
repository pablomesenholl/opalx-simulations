"""Standalone candidate mathematics for hard-edge external-field averaging.

This is not an OPALX implementation, device benchmark, or approved new method.
It keeps drift--combined Boris--drift and one self-field evaluation at the actual
midpoint. Local iterations predict the second half-drift when averaging the
external slab field. Dimensionless units: r/(c*t0), t/t0, p/(mc), E*q*t0/(mc),
B*q*t0/m; thus dr/dt=p/gamma and dp/dt=E+v cross B.
"""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def velocity(momentum):
    """Relativistic velocity in units of c."""
    return np.asarray(momentum)/np.sqrt(1+np.sum(np.asarray(momentum)**2))


def boris(momentum, electric, magnetic, dt):
    """Relativistic electric-half / magnetic Cayley / electric-half kick."""
    minus=np.asarray(momentum)+.5*dt*np.asarray(electric)
    t=.5*dt*np.asarray(magnetic)/np.sqrt(1+np.dot(minus,minus))
    s=2*t/(1+np.dot(t,t))
    plus=minus+np.cross(minus+np.cross(minus,t),s)
    return plus+.5*dt*np.asarray(electric)


def slab_fraction(z0,z1,length):
    """Exact time fraction of a straight constant-speed segment inside [0,L)."""
    if length<=0: raise ValueError('Positive slab length required')
    if z0==z1: return float(0<=z0<length)
    first,second=sorted((-z0/(z1-z0),(length-z0)/(z1-z0)))
    return max(0.,min(1.,second)-max(0.,first))


def residence_centroid(z0,zmid,z1,length,dt):
    """Active-time centroid on the two uncorrected straight half-drift segments."""
    if dt<=0 or length<=0:raise ValueError('Positive time and length required')
    duration=moment=0.
    for offset,a,b in [(0.,z0,zmid),(.5*dt,zmid,z1)]:
        if a==b:
            low,high=(0.,1.) if 0<=a<length else (0.,0.)
        else:
            low,high=sorted((-a/(b-a),(length-a)/(b-a)))
            low=max(0.,low);high=min(1.,high)
        if high>low:
            left=offset+.5*dt*low;right=offset+.5*dt*high
            duration+=right-left;moment+=.5*(right*right-left*left)
    return moment/duration if duration>0 else .5*dt


def sector_interval(start,end,length,exit_normal):
    """Clip a straight segment against z>=0 and n_exit dot r<L."""
    if length<=0:raise ValueError('Positive support length required')
    start,end=np.asarray(start),np.asarray(end);normal=np.asarray(exit_normal)
    low,high=0.,1.
    for a,b,strict in [(start[2],end[2],False),
                       (length-np.dot(normal,start),length-np.dot(normal,end),True)]:
        if a==b:
            if a<0 or (strict and a==0):return 0.,0.
        elif b>a:low=max(low,-a/(b-a))
        else:high=min(high,-a/(b-a))
    return (low,high) if high>low else (0.,0.)


def sector_centroid(start,midpoint,end,length,dt,exit_normal):
    """Active-time centroid for the same two clipped half-drift segments."""
    duration=moment=0.
    for offset,a,b in [(0.,start,midpoint),(.5*dt,midpoint,end)]:
        lo,hi=sector_interval(a,b,length,exit_normal)
        if hi>lo:
            left=offset+.5*dt*lo;right=offset+.5*dt*hi
            duration+=right-left;moment+=.5*(right*right-left*left)
    return moment/duration if duration else .5*dt


def step(position,momentum,dt,length,body_b,self_e=None,self_b=None,method='weighted',exit_normal=None):
    """Averaged candidate or ordinary midpoint; return state and kick diagnostics.

    Both self fields are supplied at the actual ordinary first-drift midpoint.
    They are held fixed while only the external residence fraction is iterated.
    Uniform body B is added to self B before one combined Boris kick. The
    weighted_moment variant adds an endpoint position correction using the
    active-time centroid and v(p_total)-v(p_SC_only). The residence predictor
    and PIC midpoint remain uncorrected: this is an explicit consistency limit.
    The velocity separation neglects O(h²) cross-coupling, hence O(h³) position
    terms for smooth bounded fields; hard-edge residence errors are separate.
    The local
    iteration is a diagnostic prototype, not a guaranteed nonlinear solver for
    arbitrary fields/large dt. No rejected step or extra PIC solve is introduced.
    """
    if dt<=0 or method not in ('weighted','weighted_moment','midpoint'): raise ValueError('Invalid step')
    position=np.asarray(position);momentum=np.asarray(momentum)
    midpoint=position+.5*dt*velocity(momentum)
    se=np.zeros(3) if self_e is None else np.asarray(self_e(midpoint) if callable(self_e) else self_e)
    sb=np.zeros(3) if self_b is None else np.asarray(self_b(midpoint) if callable(self_b) else self_b)
    if exit_normal is None:
        fraction=float(0<=midpoint[2]<length)
        def path_fraction(a,b):return slab_fraction(a[2],b[2],length)
    else:
        fraction=float(midpoint[2]>=0 and np.dot(exit_normal,midpoint)<length)
        def path_fraction(a,b):
            lo,hi=sector_interval(a,b,length,exit_normal);return hi-lo
    iterations=0
    if method!='midpoint':
        first=path_fraction(position,midpoint)
        for iterations in range(1,65):
            trial=boris(momentum,se,sb+fraction*np.asarray(body_b),dt)
            end=midpoint+.5*dt*velocity(trial)
            updated=.5*(first+path_fraction(midpoint,end))
            if abs(updated-fraction)<=4*np.finfo(float).eps:
                fraction=updated
                break
            fraction=updated
        else: raise RuntimeError('Residence iteration failed to converge')
    final_p=boris(momentum,se,sb+fraction*np.asarray(body_b),dt)
    final_r=midpoint+.5*dt*velocity(final_p)
    centroid=(residence_centroid(position[2],midpoint[2],final_r[2],length,dt) if exit_normal is None
              else sector_centroid(position,midpoint,final_r,length,dt,exit_normal))
    correction=np.zeros(3)
    if method=='weighted_moment':
        sc_only=boris(momentum,se,sb,dt)
        correction=(.5*dt-centroid)*(velocity(final_p)-velocity(sc_only))
        final_r=final_r+correction
    return final_r,final_p,dict(fraction=fraction,iterations=iterations,
        electric=se,magnetic=sb+fraction*np.asarray(body_b),midpoint=midpoint,
        active_time_centroid=centroid,position_correction=correction)


def track(position,momentum,duration,dt,length,body_b,**kwargs):
    """Uniform common timesteps; no edge-dependent substeps."""
    count=round(duration/dt)
    if not np.isclose(count*dt,duration,rtol=0,atol=1e-14):raise ValueError('DT must divide duration')
    r,p=np.array(position,dtype=float),np.array(momentum,dtype=float)
    maximum=0
    for _ in range(count):
        r,p,info=step(r,p,dt,length,body_b,**kwargs)
        maximum=max(maximum,info['iterations'])
    return r,p,maximum


def exact_slab(position,pz,duration,length,b):
    """Exact free/constant-B/free relativistic orbit, initially px=py=0.

    Positive pz and B_y, a forward crossing without a turning point (B L < pz).
    The finite slab's exit angle is asin(B L/pz); entry time is exactly ballistic.
    """
    position=np.asarray(position,dtype=float)
    if pz<=0 or b<=0 or length<=0 or b*length>=pz or position[2]>0 or duration<0:
        raise ValueError('Requires initially upstream, forward nonturning orbit')
    gamma=np.sqrt(1+pz*pz);vz=pz/gamma
    entry=-position[2]/vz
    if duration<=entry:return position+np.array([0,0,vz*duration]),np.array([0,0,pz])
    theta_exit=np.arcsin(b*length/pz)
    inside=min(duration-entry,gamma*theta_exit/b)
    theta=b*inside/gamma
    p=np.array([-pz*np.sin(theta),0,pz*np.cos(theta)])
    r=position+np.array([pz*(np.cos(theta)-1)/b,0,-position[2]+pz*np.sin(theta)/b])
    after=max(0.,duration-entry-inside)
    return r+after*velocity(p),p


def exact_full_slab_shift_derivative(pz,length,b):
    """Derivative d(final r,p)/d(initial z) after complete slab exit."""
    px=-b*length;pz_exit=np.sqrt(pz*pz-px*px)
    return np.array([px/pz,0,pz_exit/pz,0,0,0])


def uniform_sc_reference(position,momentum,duration,length,body_b,self_e,self_b):
    """High-accuracy event-resolved DOP853 oracle for constant additive SC fields.

    This resolves the two slab surfaces explicitly; it is not the candidate
    algorithm. Forward z motion is required throughout this bounded benchmark.
    """
    state=np.r_[position,momentum].astype(float);time=0.
    region=0 if state[2]<0 else (1 if state[2]<length else 2)
    for _ in range(3):
        magnetic=np.asarray(self_b)+(np.asarray(body_b) if region==1 else 0)
        def rhs(t,y):
            v=velocity(y[3:]);return np.r_[v,np.asarray(self_e)+np.cross(v,magnetic)]
        def surface(t,y):return y[2]-(0. if region==0 else length)
        surface.terminal=True;surface.direction=1
        result=solve_ivp(rhs,(time,duration),state,method='DOP853',rtol=2e-12,atol=2e-14,
                         events=surface if region<2 else None)
        if not result.success:raise RuntimeError(result.message)
        state=result.y[:,-1];time=float(result.t[-1])
        if time>=duration-1e-14:return state[:3],state[3:]
        if state[5]<=0:raise ValueError('Reference ceased forward motion')
        region+=1
    raise RuntimeError('Reference failed to finish')


def study(output=None,geometry="slab"):
    """Measure state, launch-offset derivative, crossing continuity and additive SC."""
    root=Path(__file__).resolve().parents[1]
    output=Path(output) if output else Path(tempfile.mkdtemp(prefix='edge-average-prototype-',dir=root/'results'))
    output.mkdir(parents=True,exist_ok=True)
    b=.8;length=.4;p=np.array([0.,0.,1.]);duration=2.56
    if geometry not in ('slab','sector'):raise ValueError('Unknown geometry')
    alpha=np.arcsin(b*length)
    support={} if geometry=='slab' else dict(exit_normal=np.array([-np.sin(alpha),0.,np.cos(alpha)]))
    steps=[.08,.04,.02,.01,.005]
    offsets=np.linspace(-.2073,-.1437,17)
    rows=[]
    for method in ['midpoint','weighted','weighted_moment']:
        for dt in steps:
            positions=[];momenta=[];derivatives=[];derivatives_r=[];derivatives_p=[];iteration=0
            for z in offsets:
                r=np.array([0.,0.,z]);er,ep=exact_slab(r,1.,duration,length,b)
                ar,ap,it=track(r,p,duration,dt,length,[0,b,0],method=method,**support)
                positions.append(np.linalg.norm(ar-er));momenta.append(np.linalg.norm(ap-ep));iteration=max(iteration,it)
                # Small but resolved relative to timestep; no cancellation-based
                # tolerance is used to infer derivative convergence.
                epsilon=1e-5*dt
                hi=np.r_[track(r+[0,0,epsilon],p,duration,dt,length,[0,b,0],method=method,**support)[:2]]
                lo=np.r_[track(r-[0,0,epsilon],p,duration,dt,length,[0,b,0],method=method,**support)[:2]]
                numerical=(hi-lo)/(2*epsilon)
                error=numerical-exact_full_slab_shift_derivative(1.,length,b)
                derivatives.append(np.linalg.norm(error));derivatives_r.append(np.linalg.norm(error[:3]));derivatives_p.append(np.linalg.norm(error[3:]))
            rows.append(dict(method=method,dt=dt,position_max=max(positions),momentum_max=max(momenta),
                position_rms=float(np.sqrt(np.mean(np.array(positions)**2))),momentum_rms=float(np.sqrt(np.mean(np.array(momenta)**2))),
                derivative_max=max(derivatives),derivative_rms=float(np.sqrt(np.mean(np.array(derivatives)**2))),
                derivative_position_max=max(derivatives_r),derivative_momentum_max=max(derivatives_p),max_iterations=iteration))
    frame=pd.DataFrame(rows);frame.to_csv(output/'slab-convergence.csv',index=False)
    # Cross the discontinuity of ordinary midpoint field sampling by a vanishing
    # normal displacement. Residence weighting should have a vanishing response.
    continuity=[];dt=.1;z=-.5*dt*velocity(p)[2]
    for method in ['midpoint','weighted','weighted_moment']:
        for epsilon in [.001,1e-5,1e-7,1e-9,1e-11]:
            hi=step([0,0,z+epsilon],p,dt,length,[0,b,0],method=method,**support)
            lo=step([0,0,z-epsilon],p,dt,length,[0,b,0],method=method,**support)
            continuity.append(dict(method=method,offset=epsilon,momentum_jump=float(np.linalg.norm(hi[1]-lo[1])),
                position_jump=float(np.linalg.norm(hi[0]-lo[0])),fraction_jump=hi[2]['fraction']-lo[2]['fraction']))
    pd.DataFrame(continuity).to_csv(output/'surface-continuity.csv',index=False)
    sc_e=np.array([.06,.02,0.]);sc_b=np.array([.01,.03,.02]);r=np.array([0,0,-.173])
    exact=uniform_sc_reference(r,p,duration,length,[0,b,0],sc_e,sc_b)
    sc=[]
    for method in (['midpoint','weighted','weighted_moment'] if geometry=='slab' else []):
        for dt in steps:
            actual=track(r,p,duration,dt,length,[0,b,0],self_e=sc_e,self_b=sc_b,method=method,**support)
            sc.append(dict(method=method,dt=dt,position_error=float(np.linalg.norm(actual[0]-exact[0])),
                           momentum_error=float(np.linalg.norm(actual[1]-exact[1]))))
    pd.DataFrame(sc).to_csv(output/'additive-sc-convergence.csv',index=False)
    entrance=[]
    for method in ['midpoint','weighted','weighted_moment']:
        for dt in steps:
            r_errors=[];p_errors=[]
            for phase in np.linspace(.11,.89,17):
                r=np.array([0.,0.,-phase*dt*velocity(p)[2]]);epsilon=1e-6*dt
                hi=np.r_[step(r+[0,0,epsilon],p,dt,length,[0,b,0],method=method,**support)[:2]]
                lo=np.r_[step(r-[0,0,epsilon],p,dt,length,[0,b,0],method=method,**support)[:2]]
                theta=b*(1-phase)*dt/np.sqrt(2.)
                exact_derivative=np.array([-np.sin(theta),0,np.cos(theta),-b*np.cos(theta),0,-b*np.sin(theta)])
                error=(hi-lo)/(2*epsilon)-exact_derivative
                r_errors.append(np.linalg.norm(error[:3]));p_errors.append(np.linalg.norm(error[3:]))
            entrance.append(dict(method=method,dt=dt,derivative_position_max=max(r_errors),derivative_momentum_max=max(p_errors)))
    pd.DataFrame(entrance).to_csv(output/'entrance-derivative-convergence.csv',index=False)
    summary=dict(scope='Standalone dimensionless candidate only; no OPALX production method/API changed.',
        equations='dr/dt=p/gamma, dp/dt=E+(p/gamma) cross B; gamma=sqrt(1+p^2)',
        settings=dict(geometry=geometry,exit_normal=support.get('exit_normal',np.array([0.,0.,1.])).tolist(),B_y=b,slab_length=length,pz=1.,duration=duration,launch_offsets=offsets.tolist(),dts=steps),
        slab=rows,continuity=continuity,additive_sc=sc,entrance=entrance,
        limitation='Plane slab with uniform fields, forward motion and converged local residence iteration. No arbitrary curved support, field map, PIC redistribution, RF time integration or device performance validation.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(output);print(frame.to_string(index=False));print(pd.DataFrame(sc).to_string(index=False))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    parser.add_argument('--geometry',choices=['slab','sector'],default='slab')
    args=parser.parse_args();study(args.output,args.geometry)
