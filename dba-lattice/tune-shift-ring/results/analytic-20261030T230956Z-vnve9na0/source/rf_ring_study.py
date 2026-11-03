"""One-rank RF-ring calibration and bare normal-mode probes.

Six 0.1 m VARIABLE_RF_CAVITY elements replace each D2 centre, preserving all
poses and the circumference. Constant polynomial models specify MV/m, MHz and
radians. A zero-amplitude COF-launched particle measures RF centre arrival times;
live-RF passes refine synchronization. All particles track in ordinary Kokkos
Boris kernels. FROMFILE positions are absolute [m], momenta are p/(mc).
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import h5py
import numpy as np
import pandas as pd
from scipy.constants import c
from scipy.optimize import brentq

from large_ellipsoid_benchmark import write_json, sha256, write_particles

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'results/ellipsoid-match-stable-long-d1center'


def read_pose(command):
    """Parse this study's explicit six-dimensional planar lab pose."""
    def value(key):
        match = re.search(r'(?<![A-Z_])'+key+r'\s*=\s*([^,;]+)', command)
        if match is None:
            raise ValueError(f'Missing {key} in element pose')
        return float(match[1])
    origin = np.array([value(key) for key in ['X','Y','Z']])
    theta, phi, psi = [value(key) for key in ['THETA','PHI','PSI']]
    if abs(phi)+abs(psi)>1e-14:
        raise ValueError('Only planar unrolled DBA elements are supported')
    return origin, theta


def pose_string(origin, theta):
    """Serialize a complete lab pose without hiding placement defaults."""
    return ', '.join(f'{key}={value:.17g}' for key,value in zip(
        ['X','Y','Z','THETA','PHI','PSI'], [*origin,theta,0,0]))



def cyclic_d1_lattice(base):
    """Start at the first D1 centre while retaining each unbroken physical drift.

    The ring element order begins with C1_D1; the explicit COF SECTION and
    FROMFILE mean place the reference at its middle. This avoids an artificial
    passive support boundary through the initial bunch. Analytic half-drifts
    describe the chosen section, not additional physical elements.
    """
    poses=[]
    for cell in range(1,7):
        match=re.search(r'^C'+str(cell)+r'_D1:\s*DRIFT\b[^;]*;',base,re.MULTILINE)
        if match is None: raise ValueError(f'Missing C{cell}_D1')
        origin,theta=read_pose(match[0])
        centre=origin+.5*np.array([np.sin(theta),0,np.cos(theta)])
        poses.append(dict(cell=cell,centre_m=centre.tolist(),theta_rad=theta))
    line=re.search(r'^LATTICE:\s*RING\s*=\s*\(([^)]*)\);',base,re.MULTILINE)
    names=[value.strip() for value in line[1].split(',')]
    first=names.index('C1_D1')
    replacement='LATTICE: RING = ('+', '.join(names[first:]+names[:first])+');'
    base=base[:line.start()]+replacement+base[line.end():]
    return base,poses


def straight_particles(phase,reference_r,reference_p,theta):
    """Convert analytic straight-section coordinates to absolute lab FROMFILE."""
    p0=np.linalg.norm(reference_p)
    rotation=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    position=np.asarray(phase)[:,[0,2,4]]@rotation.T+reference_r
    momentum=p0*np.asarray(phase)[:,[1,3,5]]@rotation.T+reference_p
    return np.column_stack((position,momentum))

def rf_lattice(base, frequency_hz, amplitude_mvm, phases, dump=8):
    """Split each 1 m D2 into .45 m drift/.1 m RF/.45 m drift."""
    if len(phases)!=6 or frequency_hz<=0:
        raise ValueError('Six RF phases and a positive frequency are required')
    lattice = base.split('DIST: DISTRIBUTION')[0]
    lattice = re.sub(r'PSDUMPFREQ=\d+',f'PSDUMPFREQ={dump}',lattice)
    lattice = re.sub(r'STATDUMPFREQ=\d+',f'STATDUMPFREQ={dump}',lattice)
    lattice = 'OPTION, STEPINFOFQ=10000, EBDUMP=TRUE, REPARTFREQ=0;\n'+lattice
    models = [f'RFA: POLYNOMIAL_TIME_DEPENDENCE, P0={amplitude_mvm:.17g};',
              f'RFF: POLYNOMIAL_TIME_DEPENDENCE, P0={frequency_hz/1e6:.17g};']
    poses=[]
    for cell in range(1,7):
        name=f'C{cell}_D2'
        match=re.search(r'^'+name+r':\s*DRIFT\b[^;]*;',lattice,re.MULTILINE)
        if match is None:
            raise ValueError(f'Missing {name}')
        origin,theta=read_pose(match[0])
        tangent=np.array([np.sin(theta),0,np.cos(theta)])
        models.append(f'RFP{cell}: POLYNOMIAL_TIME_DEPENDENCE, P0={phases[cell-1]:.17g};')
        replacement='\n'.join([
            f'{name}A: DRIFT, L=.45, APERTURE="RECTANGLE(1.0,0.2)", {pose_string(origin,theta)};',
            f'C{cell}_RF: VARIABLE_RF_CAVITY, L=.1, WIDTH=1, HEIGHT=.2, '+
                f'AMPLITUDE_MODEL="RFA", FREQUENCY_MODEL="RFF", PHASE_MODEL="RFP{cell}", '+pose_string(origin+.45*tangent,theta)+';',
            f'{name}B: DRIFT, L=.45, APERTURE="RECTANGLE(1.0,0.2)", '+pose_string(origin+.55*tangent,theta)+';'])
        lattice=lattice[:match.start()]+replacement+lattice[match.end():]
        line=re.search(r'^LATTICE:\s*RING\b[^;]*;',lattice,re.MULTILINE)
        changed=re.sub(r'\b'+name+r'\b',f'{name}A, C{cell}_RF, {name}B',line[0])
        lattice=lattice[:line.start()]+changed+lattice[line.end():]
        poses.append(dict(cell=cell,centre_m=(origin+.5*tangent).tolist(),origin_m=(origin+.45*tangent).tolist(),theta_rad=theta))
    return '\n'.join(models)+'\n'+lattice,poses


def tracking_deck(lattice, count, dt, turns=None, solver='NONE', charge=1e-18, mesh=32, steps=None):
    """Build an absolute FROMFILE launch with no BEAM.PC or INITIALORBIT."""
    if (turns is None)==(steps is None): raise ValueError('Specify exactly one of turns or nominal steps')
    limit=10000000 if steps is None else int(steps)
    stop='' if turns is None else f', TURNS={turns}'
    return lattice+f'''\nDIST: DISTRIBUTION, TYPE=FROMFILE, FNAME="particles.txt", NPARTDIST={count};
SOURCE: EMISSIONSOURCE, DISTRIBUTION=DIST;
SOURCES: EMISSIONSOURCELIST=(SOURCE);
ONE: BINNING, PARAMETER="GAMMAZ", MAXBINS=1, ADAPTIVEBINNING=FALSE, DUMPBINSFILE="NONE", TABLEPRINTFREQ=0;
FS: FIELDSOLVER, TYPE={solver}, BINS=ONE, NX={mesh}, NY={mesh}, NZ={mesh},
    PARFFTX=TRUE, PARFFTY=TRUE, PARFFTZ=TRUE, BCFFTX=OPEN, BCFFTY=OPEN, BCFFTZ=OPEN;
BEAM1: BEAM, PARTICLE=PROTON, CHARGE=1, NALLOC={count}, SOURCES=SOURCES, BCHARGE={charge:.17g};
TRACK, LINE=LATTICE, BEAM=BEAM1, DT={dt:.17g}, MAXSTEPS={limit};
RUN, METHOD="PARALLEL", FIELDSOLVER=FS{stop};
ENDTRACK;
QUIT;
'''


def frenet_to_cartesian(phase, p0, radius=2.):
    """Map small canonical deviations at the common B2/B1 circular section.

    phase=(x,px/p0,y,py/p0,z_head,delta). The exact circular geometry supplies
    the first-order lab px=-p0*z/rho correlation of an ahead-of-reference particle.
    z_head is represented by reference arc length rho*theta. This spatial/time
    longitudinal identification is paraxial and used here only for tiny probes.
    """
    phase=np.atleast_2d(np.asarray(phase,float))
    angle=phase[:,4]/radius
    cs,sn=np.cos(angle),np.sin(angle)
    rad=radius+phase[:,0]
    out=np.empty_like(phase)
    out[:,:3]=np.column_stack((rad*cs-radius,phase[:,2],rad*sn))
    out[:,3:]=p0*np.column_stack((phase[:,1]*cs-(1+phase[:,5])*sn,phase[:,3],phase[:,1]*sn+(1+phase[:,5])*cs))
    return out


def cartesian_to_frenet(position,momentum,p0,radius=2.):
    """Inverse circular coordinate map, including the radial momentum rotation."""
    position,momentum=np.asarray(position),np.asarray(momentum)
    angle=np.arctan2(position[...,2],position[...,0]+radius)
    cs,sn=np.cos(angle),np.sin(angle)
    result=np.empty(position.shape[:-1]+(6,))
    result[...,0]=np.hypot(position[...,0]+radius,position[...,2])-radius
    result[...,1]=(momentum[...,0]*cs+momentum[...,2]*sn)/p0
    result[...,2]=position[...,1]
    result[...,3]=momentum[...,1]/p0
    result[...,4]=radius*angle
    result[...,5]=(-momentum[...,0]*sn+momentum[...,2]*cs)/p0-1
    return result


def read_history(path):
    """Read ordered reference and ID-stable particle snapshots with SI fields."""
    with h5py.File(path,'r') as data:
        keys=sorted((k for k in data if re.fullmatch(r'Step#\d+',k)),key=lambda k:int(k[5:]))
        times=[]; refs=[]; refps=[]; positions=[]; momenta=[]; electric=[]
        ids=None
        for key in keys:
            group=data[key]
            order=np.argsort(group['id'][:])
            this=group['id'][:][order]
            if ids is None: ids=this
            if not np.array_equal(this,ids): raise ValueError('Particle loss or changed IDs')
            times.append(float(np.asarray(group.attrs['TIME']).ravel()[0]))
            refs.append(np.asarray(group.attrs['RefPartR']))
            refps.append(np.asarray(group.attrs['RefPartP']))
            positions.append(np.column_stack([group[a][:][order] for a in ['x','y','z']]))
            momenta.append(np.column_stack([group[a][:][order] for a in ['px','py','pz']]))
            electric.append(np.column_stack([group[a][:][order] for a in ['Ex','Ey','Ez']]))
    result=dict(t=np.array(times),r=np.array(refs),p=np.array(refps),local_r=np.array(positions),local_p=np.array(momenta),electric=np.array(electric),ids=ids)
    if not len(times) or not all(np.isfinite(v).all() for v in result.values()): raise ValueError('Missing/nonfinite snapshots')
    # The final writer can repeat a configured dump at the same physical time.
    keep=np.r_[True,np.diff(result['t'])>0]
    return {key:(value[keep] if key!='ids' else value) for key,value in result.items()}


def plane_crossings(history, origin, theta, aperture=.1):
    """Find forward reference crossings using cubic Hermite position and velocity.

    Root coordinates are normalized to each bracket, so SI-second tolerances
    cannot erase a crossing. Transverse acceptance rejects the opposite ring side.
    """
    t,r,p=history['t'],history['r'],history['p']
    ez=np.array([np.sin(theta),0,np.cos(theta)])
    ex=np.array([np.cos(theta),0,-np.sin(theta)])
    z=(r-origin)@ez
    x=(r-origin)@ex
    velocity=c*p/np.sqrt(1+np.sum(p*p,axis=1))[:,None]
    speed=velocity@ez
    candidates=np.flatnonzero((z[:-1]<=0)&(z[1:]>0)&(speed[:-1]>0)&(abs(x[:-1])<aperture)&(abs(x[1:])<aperture))
    events=[]
    for i in candidates:
        dt=t[i+1]-t[i]
        def hermite(u):
            return (2*u**3-3*u*u+1)*z[i]+(u**3-2*u*u+u)*dt*speed[i]+(-2*u**3+3*u*u)*z[i+1]+(u**3-u*u)*dt*speed[i+1]
        fraction=brentq(hermite,0,1,xtol=1e-14)
        events.append(dict(time_s=float(t[i]+fraction*dt),index=int(i),fraction=float(fraction),bracket_dt_s=float(dt),distance_ends_m=[float(z[i]),float(z[i+1])]))
    return events


def reference_summary(history,cof,poses):
    """Measure one-turn synchronous timing, energy and particle-reference errors."""
    p0=np.linalg.norm(cof['momentum_mc'])
    mass=cof['mass_eV']
    energies=(np.sqrt(1+np.sum(history['p']**2,axis=1))-1)*mass
    centres=[]
    for pose in poses:
        events=plane_crossings(history,np.asarray(pose['centre_m']),pose['theta_rad'])
        if len(events)!=1 or max(abs(np.asarray(events[0]['distance_ends_m'])))>=.05:
            raise ValueError('Need one cavity-centre crossing bracketed inside its straight gap')
        centres.append(events[0])
    return dict(period_s=float(history['t'][-1]),centre_times_s=[v['time_s'] for v in centres],centre_crossings=centres,
                reference_return_m=float(np.linalg.norm(history['r'][-1]-cof['position_m'])),
                reference_momentum_return_mc=float(np.linalg.norm(history['p'][-1]-cof['momentum_mc'])),
                reference_final_energy_error_eV=float(energies[-1]-(np.sqrt(1+p0*p0)-1)*mass),
                reference_energy_range_eV=[float(energies.min()),float(energies.max())],
                max_particle_reference_position_m=float(np.max(np.linalg.norm(history['local_r'],axis=2))),
                max_particle_reference_momentum_mc=float(np.max(np.linalg.norm(history['local_p']-np.column_stack((np.zeros(len(history['p'])),np.zeros(len(history['p'])),np.linalg.norm(history['p'],axis=1)))[:,None,:],axis=2))),
                max_electric_V_per_m=float(np.max(np.linalg.norm(history['electric'],axis=2))), snapshots=len(history['t']))


def rf_parameters(period,centre_times,p0,mass_ev,g=.001,harmonic=4,length=.1):
    """Sinc-corrected uniform-gap amplitude and head-focusing RF phases."""
    gamma=np.sqrt(1+p0*p0); beta=p0/gamma
    frequency=harmonic/period
    k=2*np.pi*frequency/(beta*c)
    transit=np.sinc(k*length/(2*np.pi))
    amplitude=g*beta**2*gamma*mass_ev/(k*length*transit)/1e6
    phases=(np.pi-2*np.pi*frequency*np.asarray(centre_times)+np.pi)%(2*np.pi)-np.pi
    return dict(frequency_hz=float(frequency),amplitude_mvm=float(amplitude),phases_rad=phases.tolist(),
                g_per_m=g,harmonic=harmonic,transit_factor=float(transit),k_per_m=float(k),length_m=length)



def rf_body_intervals(logtext):
    """Merge RF support intervals, including splits caused by passive monitors."""
    intervals=set(); current=None
    for line in logtext.splitlines():
        match=re.search(r'Key:\s*\(\s*([-+0-9.eE]+)\s*-\s*([-+0-9.eE]+)\)',line)
        if match: current=(float(match[1]),float(match[2]))
        if current and 'VARIABLE_RF_CAVITY' in line:
            intervals.add(current)
    merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo<=merged[-1][1]+2e-6:
            merged[-1]=(merged[-1][0],max(merged[-1][1],hi))
        else: merged.append((lo,hi))
    return [interval for interval in merged if abs(interval[1]-interval[0]-.1)<2e-6]


def execute(directory,binary,timeout=180,expected_boundary_control=True):
    """Run one rank with immutable input/binary hashes and bounded child lifetime."""
    command=[str(binary),'--info','1','case.in']
    meta=dict(command=command,binary_sha256=sha256(binary),input_sha256=sha256(directory/'case.in'),
              particle_sha256=sha256(directory/'particles.txt'),mpi_ranks=1,omp_threads=1,status='running',timeout_s=timeout,expected_boundary_control=expected_boundary_control)
    started=time.monotonic()
    with (directory/'run.log').open('w') as log:
        child=subprocess.Popen(command,cwd=directory,env={**os.environ,'OMP_NUM_THREADS':'1','OMP_PROC_BIND':'false'},stdout=log,stderr=subprocess.STDOUT)
        meta['pid']=child.pid; write_json(directory/'provenance.json',meta)
        try:
            child.wait(timeout=timeout)
            meta['status']='completed' if child.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:
            child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            meta['status']='timed_out'
    meta.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-started)
    logtext=(directory/'run.log').read_text()
    meta['boundary_controller_active']='Ring boundary control: synchronized ordinary Boris/PIC substeps.' in logtext
    requested_rf=re.findall(r'^(C[1-6]_RF):', (directory/'case.in').read_text(), re.MULTILINE)
    meta['active_rf_names']=[name for name in requested_rf if name in logtext]
    meta['rf_body_intervals_m']=rf_body_intervals(logtext)
    meta['rf_registration_verified']=(len(meta['active_rf_names'])==len(requested_rf) or len(meta['rf_body_intervals_m'])==len(requested_rf))
    if meta['status']=='completed' and not meta['rf_registration_verified']:
        meta['status']='missing_rf_elements'
    write_json(directory/'provenance.json',meta)
    if meta['status']!='completed' or meta['boundary_controller_active'] != expected_boundary_control:
        raise RuntimeError(f'{directory}: {meta["status"]}, controller={meta["boundary_controller_active"]}')
    print(f'{directory.name}: {meta["elapsed_seconds"]:.2f}s',flush=True)
    return meta


def prepare(binary,dt=6.25e-11):
    """Snapshot source and obtain a fresh bare-magnet COF before RF calibration."""
    repo=ROOT.parents[2]
    study=Path(tempfile.mkdtemp(prefix='rf-ring-',dir=ROOT/'results'))
    print(study,flush=True)
    shutil.copy2(__file__,study/'rf_ring_study.py')
    for filename in ['config.json','bare-gaussian-cof.in']:
        shutil.copy2(ROOT/filename,study/filename)
    rotated,poses=cyclic_d1_lattice((study/'bare-gaussian-cof.in').read_text())
    (study/'bare-gaussian-cof.in').write_text(rotated)
    write_json(study/'sections.json',poses)
    model=study/'analytic'; model.mkdir()
    for filename in ['summary.json','match-0pc.npz']:
        shutil.copy2(MODEL/filename,model/filename)
    (study/'source.diff').write_text(subprocess.check_output(['git','diff'],cwd=repo,text=True))
    source=study/'source'; source.mkdir()
    files=subprocess.check_output(['git','ls-files','--modified','--others','--exclude-standard','src','unit_tests'],cwd=repo,text=True).splitlines()
    for name in files:
        if (repo/name).is_file():
            target=source/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(repo/name,target)
    base,_=cyclic_d1_lattice((ROOT/'bare-cof.in').read_text())
    base=base.replace('ENABLELINEARTRANSFERMAPS=TRUE','ENABLELINEARTRANSFERMAPS=FALSE')
    base=re.sub(r'\bDT\s*=\s*[^,;]+',f'DT={dt:.17g}',base)
    base=re.sub(r'TIMEINTEGRATOR="[^"]+"','TIMEINTEGRATOR="BORIS"',base)
    section=[*poses[0]['centre_m'],poses[0]['theta_rad'],0,0]
    base=re.sub(r'(ic:\s*COF\s*,)',lambda match:match[0]+' SECTION={'+','.join(f'{v:.17g}' for v in section)+'},',base)
    (study/'cof.in').write_text(base)
    with (study/'cof.log').open('w') as log:
        subprocess.run([str(binary),'--info','1','cof.in'],cwd=study,env={**os.environ,'OMP_NUM_THREADS':'1','OMP_PROC_BIND':'false'},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    cof=json.loads((study/'bare-cof.json').read_text())
    if not cof['converged'] or max(abs(np.asarray(cof['residual'])))>1e-9:
        raise ValueError('DBA closed orbit did not converge')
    write_json(study/'study.json',dict(binary=str(binary),binary_sha256=sha256(binary),dt_s=dt,mpi_ranks=1,omp_threads=1,revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),model=str(MODEL)))
    return study


def calibrate(study,iterations=3):
    """Measure zero-RF timing, then iteratively synchronize actual RF passes."""
    meta=json.loads((study/'study.json').read_text()); binary=Path(meta['binary'])
    if sha256(binary)!=meta['binary_sha256']: raise ValueError('Binary changed after preparation')
    cof=json.loads((study/'bare-cof.json').read_text())
    model=json.loads((study/'analytic/summary.json').read_text())
    p0=np.linalg.norm(cof['momentum_mc']); gamma=np.sqrt(1+p0*p0)
    period=model['circumference_m']/(c*p0/gamma)
    parameters=rf_parameters(period,np.zeros(6),p0,cof['mass_eV'])
    parameters['amplitude_mvm']=0
    records=[]
    for iteration in range(iterations+1):
        directory=study/('zero-amplitude' if iteration==0 else f'synchronous-{iteration}')
        directory.mkdir()
        lattice,poses=rf_lattice((study/'bare-gaussian-cof.in').read_text(),parameters['frequency_hz'],parameters['amplitude_mvm'],parameters['phases_rad'],dump=8)
        (directory/'case.in').write_text(tracking_deck(lattice,1,meta['dt_s'],1))
        write_particles(directory/'particles.txt',np.array([[*cof['position_m'],*cof['momentum_mc']]]))
        write_json(directory/'rf.json',parameters)
        execute(directory,binary)
        summary=reference_summary(read_history(directory/'case.h5'),cof,poses)
        if iteration and summary['max_electric_V_per_m'] < 1e-8:
            write_json(directory/'reference-summary.json',dict(**summary,valid_rf=False))
            raise RuntimeError('Nonzero RF amplitude produced no electric field')
        write_json(directory/'reference-summary.json',summary)
        records.append(dict(case=directory.name,**summary,rf=parameters))
        parameters=rf_parameters(summary['period_s'],summary['centre_times_s'],p0,cof['mass_eV'])
        write_json(study/'calibration.json',dict(passes=records,next_parameters=parameters))
        print(json.dumps({k:v for k,v in summary.items() if k not in ['centre_crossings']}),flush=True)
    return records



def normal_mode_probes(basis,amplitude=1e-5):
    """A central particle and twelve antithetic pure-mode quadrature probes."""
    basis=np.asarray(basis,float)
    if basis.shape!=(6,6) or amplitude<=0: raise ValueError('6x6 mode basis and positive amplitude required')
    phase=[np.zeros(6)]; labels=['centre']
    for mode,axis in enumerate('xyz'):
        scale=amplitude/np.max(abs(basis[np.ix_([0,2,4],[2*mode,2*mode+1])]))
        for quadrature in range(2):
            vector=scale*basis[:,2*mode+quadrature]
            phase.extend([vector,-vector]); labels.extend([f'{axis}{quadrature}+',f'{axis}{quadrature}-'])
    return np.asarray(phase),labels


def section_observations(history,sections,p0,initial_phase=None):
    """Strobe every particle at reference D1-centre events in static straight axes.

    Both bracketing samples must lie within 0.4 m of the section, so the bare
    probe interpolation never crosses a magnet/RF face. In this force-free
    straight lab momentum is constant and linear position interpolation is exact.
    This helper is for small bare-probe HDF5 histories, not full PIC phase dumps.
    """
    direction=history['p']/np.linalg.norm(history['p'],axis=1)[:,None]
    if np.max(abs(direction[:,1]))>1e-12: raise ValueError('Only planar references supported')
    horizontal=np.column_stack((direction[:,2],np.zeros(len(direction)),-direction[:,0]))
    vertical=np.array([0,1,0])
    lr=history['local_r']; lp=history['local_p']
    lab_r=history['r'][:,None,:]+lr[:,:,0,None]*horizontal[:,None,:]+lr[:,:,1,None]*vertical+lr[:,:,2,None]*direction[:,None,:]
    lab_p=lp[:,:,0,None]*horizontal[:,None,:]+lp[:,:,1,None]*vertical+lp[:,:,2,None]*direction[:,None,:]
    events=[]
    if initial_phase is not None: events.append((0.,0,np.asarray(initial_phase)))
    for cell,section in enumerate(sections):
        origin=np.asarray(section['centre_m']); theta=section['theta_rad']
        rotation=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        for event in plane_crossings(history,origin,theta):
            if max(abs(np.asarray(event['distance_ends_m'])))>.4:
                raise ValueError('Section bracket lies outside the straight')
            i=event['index']; f=event['fraction']
            r=((1-f)*lab_r[i]+f*lab_r[i+1]-origin)@rotation
            p=((1-f)*lab_p[i]+f*lab_p[i+1])@rotation/p0
            phase=np.column_stack((r[:,0],p[:,0],r[:,1],p[:,1],r[:,2],p[:,2]-1))
            events.append((event['time_s'],cell,phase))
    events.sort(key=lambda event:event[0])
    times=np.array([e[0] for e in events]); cells=np.array([e[1] for e in events]); phase=np.array([e[2] for e in events])
    if len(cells)<12 or np.any(np.diff(cells)%6!=1):
        raise ValueError('Missing or unordered equivalent-cell sections')
    return dict(t_s=times,cells=cells,turn_coordinate=np.arange(len(events))/6,phase=phase,ids=history['ids'])



def primary_mode_table(table,labels,signed_cell_phases):
    """Select deliberately excited modes; retain the full table as leakage evidence."""
    result=table.copy()
    result['launch_label']=result['id'].map(dict(enumerate(labels)))
    result=result[result['mode']==result['launch_label'].str[0]].copy()
    if set(result['mode'])!=set('xyz') or not result['active'].all():
        raise ValueError('All three deliberately excited modes must have active fits')
    expected=dict(zip('xyz',6*np.abs(signed_cell_phases)/(2*np.pi)))
    result['analytic_tune']=result['mode'].map(expected)
    result['phase_minus_analytic']=result['phase_tune_absolute']-result['analytic_tune']
    result['spectral_minus_analytic']=result['spectral_tune_absolute']-result['analytic_tune']
    return result.sort_values('id')


def analyze_probes(study,directory):
    """Write independent spectral/phase fits plus a primary-mode-only summary."""
    from analyze_ellipsoid_modes import analyze
    destination=directory/'mode-analysis'
    model_path=study/'analytic/match-0pc.npz'
    analyze(directory/'section-observations.npz',model_path,destination,central_id=0)
    model=np.load(model_path)
    labels=json.loads((directory/'launch.json').read_text())['labels']
    table=primary_mode_table(pd.read_csv(destination/'mode-tunes.csv'),labels,model['signed_cell_phases'])
    table.to_csv(destination/'primary-mode-tunes.csv',index=False)
    rows=[]
    for axis,group in table.groupby('mode'):
        rows.append(dict(mode=axis,probes=len(group),analytic_tune=float(group.analytic_tune.iloc[0]),
            tracking_phase_mean=float(group.phase_tune_absolute.mean()),tracking_spectral_mean=float(group.spectral_tune_absolute.mean()),
            maximum_phase_error=float(abs(group.phase_minus_analytic).max()),maximum_spectral_error=float(abs(group.spectral_minus_analytic).max()),
            maximum_estimator_difference=float(group.estimator_difference.max()),maximum_phase_residual_rad=float(group.phase_residual_rad.max()),
            minimum_spectral_coherent_fraction=float(group.spectral_coherent_fraction.min())))
    summary=dict(selection='Each source particle is compared only in its deliberately excited launch mode; all cross-mode fits remain in mode-tunes.csv',
                 observations=int(np.load(directory/'section-observations.npz')['phase'].shape[0]),modes=rows)
    write_json(destination/'primary-summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary

def run_probes(study,turns=32,amplitude=1e-5):
    """Run small-amplitude zero-charge normal-mode probes with the calibrated RF."""
    meta=json.loads((study/'study.json').read_text()); binary=Path(meta['binary'])
    if sha256(binary)!=meta['binary_sha256']: raise ValueError('Binary changed after preparation')
    calibration=json.loads((study/'calibration.json').read_text())
    last=calibration['passes'][-1]
    if not last.get('max_electric_V_per_m',0)>1e-8: raise ValueError('RF has not been physically calibrated')
    rf=last['rf']
    cof=json.loads((study/'bare-cof.json').read_text()); sections=json.loads((study/'sections.json').read_text())
    model=np.load(study/'analytic/match-0pc.npz')
    phase,labels=normal_mode_probes(model['normal_mode_basis'],amplitude)
    particles=straight_particles(phase,cof['position_m'],cof['momentum_mc'],sections[0]['theta_rad'])
    directory=study/f'bare-probes-{turns}turns-{amplitude*1e6:g}um'
    directory.mkdir()
    lattice,_=rf_lattice((study/'bare-gaussian-cof.in').read_text(),rf['frequency_hz'],rf['amplitude_mvm'],rf['phases_rad'],dump=48)
    (directory/'case.in').write_text(tracking_deck(lattice,len(phase),meta['dt_s'],turns))
    write_particles(directory/'particles.txt',particles)
    write_json(directory/'launch.json',dict(labels=labels,phase=phase.tolist(),amplitude_m=amplitude,turns=turns,rf=rf))
    execute(directory,binary,timeout=max(180,turns*10))
    observed=section_observations(read_history(directory/'case.h5'),sections,np.linalg.norm(cof['momentum_mc']),phase)
    np.savez(directory/'section-observations.npz',**observed)
    # Analysis helpers deliberately remain separate: the matched-ellipsoid
    # model owns signed normal-mode conventions and independent spectral fits.
    analyze_probes(study,directory)
    print(directory,flush=True)
    return directory

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path)
    parser.add_argument('--binary',type=Path,default=ROOT.parents[2]/'omp-build/src/opalx')
    parser.add_argument('--dt',type=float,default=6.25e-11)
    parser.add_argument('--calibrate',action='store_true')
    parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--run-probes',action='store_true')
    parser.add_argument('--probe-turns',type=int,default=32)
    parser.add_argument('--probe-amplitude',type=float,default=1e-5)
    parser.add_argument('--analyze-probes',type=Path)
    args=parser.parse_args()
    study=args.study.resolve() if args.study else prepare(args.binary.resolve(),args.dt)
    if args.calibrate: calibrate(study,args.iterations)
    if args.run_probes: run_probes(study,args.probe_turns,args.probe_amplitude)
    if args.analyze_probes: analyze_probes(study,args.analyze_probes.resolve())


if __name__=='__main__': main()
