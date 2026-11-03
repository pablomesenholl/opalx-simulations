"""Bounded shared-boundary control-subset experiment; full-bunch PIC/Boris.

No public input API is changed. This runner exercises the approved experimental
process environment switch and compares one common 40 ns endpoint. It does not
claim an equilibrium ellipsoid or measure a tune. Run only explicitly via CLI.
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
from large_ellipsoid_benchmark import generate_ellipsoid, write_particles, write_json, sha256
from rf_ring_study import cyclic_d1_lattice, rf_lattice, tracking_deck, straight_particles, read_pose
from check_temporal_monitors import local_to_lab, phase_at_section
from short_arc_ellipsoid import ROOT, REPO, CALIBRATION, MATCH, CHARGE, read_final
from ellipsoid_observables import uniform_density_diagnostics
from match_ellipsoid_ring import eigenemittances

DURATION = 40e-9
ENVIRONMENT = 'OPALX_TEST_BOUNDARY_CONTROL_STRIDE'
CASES = (
    ('control4-dt62p5ps', 16384, 62.5e-12, 600),
    ('control32-dt62p5ps', 4096, 62.5e-12, 600),
    ('control139-dt62p5ps', 1024, 62.5e-12, 600),
    ('control32-dt31p25ps', 4096, 31.25e-12, 600),
    ('full-dt62p5ps', 0, 62.5e-12, 90),
)


def selected_ids(ids, stride):
    """Exact C++ SplitMix64 predicate, operating on actual persistent particle IDs."""
    ids = np.asarray(ids, dtype=np.int64)
    if stride is None or stride in (0, 1):
        return ids.copy()
    if stride < 0 or stride & (stride-1):
        raise ValueError('Stride must be zero or a positive power of two')
    z = ids.astype(np.uint64)+np.uint64(0x9e3779b97f4a7c15)
    z = (z ^ (z >> 30))*np.uint64(0xbf58476d1ce4e5b9)
    z = (z ^ (z >> 27))*np.uint64(0x94d049bb133111eb)
    z ^= z >> 31
    return ids[(z & np.uint64(stride-1)) == 0]


def whitening(covariance):
    """Symmetric C0^-1/2; this fixed metric is shared by every comparison."""
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values <= 0):
        raise ValueError('Positive-definite covariance required')
    return (vectors/np.sqrt(values))@vectors.T


def geometry(lattice, section, p0, phase):
    """Actual posed first-Q faces and ballistic arrival estimates (not tracked times)."""
    def pose(name):
        command = re.search(r'(?m)^'+name+r'\s*:[^;]+;', lattice)
        if command is None:
            raise ValueError(f'Missing {name}')
        origin, theta = read_pose(command[0])
        length = float(re.search(r'(?<![A-Z_])L\s*=\s*([^,;]+)',command[0])[1])
        return origin, theta, length
    theta = section['theta_rad']
    tangent = np.array([np.sin(theta), 0., np.cos(theta)])
    q, qt, qlength = pose('C1_Q')
    rf, _, _ = pose('C1_RF')
    start = np.asarray(section['centre_m'])
    entry = float((q-start)@tangent)
    end = entry+qlength
    rf_start = float((rf-start)@tangent)
    if abs(qt-theta)>1e-12 or entry<=0 or end<=entry:
        raise ValueError('Expected downstream straight quadrupole')
    momentum = p0*np.asarray(phase)[:,[1,3,5]].copy(); momentum[:,2] += p0
    vz = c*momentum[:,2]/np.sqrt(1+np.sum(momentum**2,axis=1))
    speed = c*p0/np.sqrt(1+p0*p0)
    if not np.all(abs(phase[:,4])<.5):
        raise ValueError('Initial bunch is not contained in the unbroken D1')
    if not speed*DURATION > end or np.max(phase[:,4]+vz*DURATION)>=rf_start:
        raise ValueError('Arc must cross Q and stop before RF')
    return dict(q_entry_m=entry,q_exit_m=end,rf_entry_m=rf_start,
        reference_entry_s=entry/speed,reference_exit_s=end/speed,
        reference_endpoint_m=speed*DURATION,
        ballistic_entry_s=(entry-phase[:,4])/vz,ballistic_exit_s=(end-phase[:,4])/vz)


def coverage(phase, ids, control, inverse, geom):
    """Keep the selected sample as-is, recording phase and mode coverage without reseeding."""
    positions = {int(value): i for i,value in enumerate(ids)}
    index = np.array([positions[int(value)] for value in control],dtype=int)
    if len(index)==0:
        return dict(count=0, ids=[])
    values = phase[index]
    mode = values@inverse.T
    amplitudes = np.sqrt(mode[:,::2]**2+mode[:,1::2]**2)
    fullstd = np.std(phase,axis=0)
    return dict(count=len(index),ids=control.tolist() if len(index)<=1024 else None,
        mean_over_full_rms=(values.mean(axis=0)/fullstd).tolist(),
        rms_over_full_rms=(np.sqrt(np.mean(values**2,axis=0))/fullstd).tolist(),
        phase_quantiles=np.quantile(values,[0,.05,.5,.95,1],axis=0).tolist(),
        mode_amplitude_quantiles=np.quantile(amplitudes,[0,.05,.5,.95,1],axis=0).tolist(),
        ballistic_q_entry_quantiles_s=np.quantile(geom['ballistic_entry_s'][index],[0,.05,.5,.95,1]).tolist(),
        ballistic_q_exit_quantiles_s=np.quantile(geom['ballistic_exit_s'][index],[0,.05,.5,.95,1]).tolist())


def make_deck(lattice,count,dt,solver='OPEN',mesh=32):
    """Exactly 40 ns nominal clock, full N charge, final H5 only."""
    steps = round(DURATION/dt)
    if not np.isclose(steps*dt,DURATION,rtol=0,atol=1e-24):
        raise ValueError('DT must divide the common endpoint')
    lattice = re.sub(r'(PSDUMPFREQ|STATDUMPFREQ)=\d+',r'\1=1000000000',lattice)
    lattice = re.sub(r'STEPINFOFQ=\d+','STEPINFOFQ=100',lattice)
    return tracking_deck(lattice,count,dt,solver=solver,charge=CHARGE,mesh=mesh,steps=steps)


def prepare(binary,output=None,count=131072,cases=CASES,smoke=True):
    """Freeze launch, sources, binary, control sets and immutable input hashes."""
    study = Path(output) if output else Path(tempfile.mkdtemp(prefix='subset-boundary-arc-',dir=ROOT/'results'))
    study.mkdir(parents=True,exist_ok=True)
    if (study/'study.json').exists(): raise ValueError('Study already exists')
    cof=json.loads((CALIBRATION/'bare-cof.json').read_text())
    sections=json.loads((CALIBRATION/'sections.json').read_text())
    rf=json.loads((CALIBRATION/'synchronous-3/rf.json').read_text())
    base,derived=cyclic_d1_lattice((ROOT/'bare-gaussian-cof.in').read_text())
    if not np.allclose(sections[0]['centre_m'],derived[0]['centre_m'],atol=1e-14):
        raise ValueError('Calibration section mismatch')
    lattice,_=rf_lattice(base,rf['frequency_hz'],rf['amplitude_mvm'],rf['phases_rad'])
    model=np.load(MATCH/'match-160pc.npz'); covariance=model['covariance']; inverse=model['normal_mode_inverse']
    p0=float(np.linalg.norm(cof['momentum_mc']))
    source=study/'source';source.mkdir()
    changed=subprocess.check_output(['git','ls-files','--modified','--others','--exclude-standard','--','src','unit_tests'],cwd=REPO,text=True).splitlines()
    for relative in dict.fromkeys(changed):
        path=REPO/relative
        if path.is_file():
            target=source/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    (source/'working-tree.diff').write_text(subprocess.check_output(['git','diff'],cwd=REPO,text=True))
    (source/'analysis').mkdir()
    for path in Path(__file__).parent.glob('*.py'): shutil.copy2(path,source/'analysis'/path.name)
    for path in [CALIBRATION/'bare-cof.json',CALIBRATION/'sections.json',CALIBRATION/'synchronous-3/rf.json',MATCH/'match-160pc.npz',ROOT/'bare-gaussian-cof.in']:
        shutil.copy2(path,study/path.name)
    # Keep an immutable executable, so later unrelated builds do not change this study.
    executable=study/'opalx';shutil.copy2(binary,executable)
    entries=[(name,stride,dt,limit,count,'OPEN') for name,stride,dt,limit in cases]
    if smoke: entries += [('smoke-unset',None,62.5e-12,90,64,'NONE'),('smoke-stride1',1,62.5e-12,90,64,'NONE')]
    cached={}; checks={}
    for name,stride,dt,limit,n,solver in entries:
        if n not in cached:
            phase=generate_ellipsoid(covariance,n,seed=20260911)
            particles=straight_particles(phase,cof['position_m'],cof['momentum_mc'],sections[0]['theta_rad'])
            file=study/f'initial-n{n}.txt';write_particles(file,particles)
            # FROMFILE stores a particle count line followed by the named
            # column header; the numeric rows start on line three.
            loaded=np.loadtxt(file,skiprows=2)
            static=dict(centre_m=cof['position_m'],theta_rad=sections[0]['theta_rad'])
            restored=phase_at_section(loaded[:,:3],loaded[:,3:],static,p0)
            if np.max(abs(restored-phase))>1e-12: raise ValueError('FROMFILE phase round trip failed')
            ids=np.arange(n,dtype=np.int64)
            diagnostic=np.sort(np.random.default_rng(90217).choice(ids,min(n,1024),replace=False))
            geom=geometry(lattice,sections[0],p0,phase)
            np.savez(study/f'initial-n{n}.npz',phase=phase,covariance_ideal=covariance,covariance_sample=np.cov(phase,rowvar=False,bias=True),ids=ids,diagnostic_ids=diagnostic)
            checks[str(n)]=dict(count=n,coordinate_order=['x_m','px/p0','y_m','py/p0','z_head_m','pz/p0-1'],
                charge_C=CHARGE,charge_per_particle_C=CHARGE/n,mean=phase.mean(axis=0).tolist(),
                covariance=np.cov(phase,rowvar=False,bias=True).tolist(),
                roundtrip_max_abs=float(abs(restored-phase).max()),
                density=uniform_density_diagnostics(phase[:,[0,2,4]],covariance[np.ix_([0,2,4],[0,2,4])]),
                geometry={key:value for key,value in geom.items() if not key.startswith('ballistic')},
                full_coverage=coverage(phase,ids,ids,inverse,geom))
            cached[n]=(phase,ids,geom,file)
        phase,ids,geom,file=cached[n]
        chosen=selected_ids(ids,stride)
        directory=study/name;directory.mkdir()
        shutil.copy2(file,directory/'particles.txt')
        (directory/'case.in').write_text(make_deck(lattice,n,dt,solver))
        np.save(directory/'control-ids.npy',chosen)
        write_json(directory/'coverage.json',coverage(phase,ids,chosen,inverse,geom))
        write_json(directory/'provenance.json',dict(status='prepared',count=n,mesh=32,solver=solver,dt_s=dt,
            steps=round(DURATION/dt),duration_s=DURATION,charge_C=CHARGE,stride=stride,actual_control_count=len(chosen),
            timeout_s=limit,initial_npz=f'initial-n{n}.npz',input_sha256=sha256(directory/'case.in'),
            particles_sha256=sha256(directory/'particles.txt'),control_ids_sha256=sha256(directory/'control-ids.npy')))
    write_json(study/'launch-checks.json',checks)
    write_json(study/'study.json',dict(binary=str(executable),binary_sha256=sha256(executable),original_binary=str(Path(binary).resolve()),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        duration_s=DURATION,p0_mc=p0,charge_C=CHARGE,mpi_ranks=1,omp_threads=8,seed=20260911,
        cases=[row[0] for row in entries],scope='Shared boundary sample sensitivity across Q; full-bunch PIC/Boris. No equilibrium or tune claim.',
        id_mapping='Single FROMFILE source on MPI1: IPPL initial IDs 0..N-1 in file order; run validates actual H5 IDs and startup selected count.',
        source_sha256={str(path.relative_to(study)):sha256(path) for path in source.rglob('*') if path.is_file()}))
    print(study,flush=True)
    return study


def parse_log(log):
    """Capture live predicate evidence, accepted/rejected counts and last reported time."""
    sample=re.search(r'EXPERIMENTAL boundary sample: stride=(\d+), selected=(\d+), reference=always',log)
    trials=re.search(r'Boris boundary trials=(\d+), rejected=(\d+)',log)
    solves=re.search(r'Total FieldSolver calls:\s*(\d+)',log)
    times=re.findall(r'\bt=\s*([\d.eE+-]+)\s*\[(ps|ns|us|s)\]',log)
    factors=dict(ps=1e-12,ns=1e-9,us=1e-6,s=1)
    return dict(boundary_controller_active='synchronized ordinary Boris/PIC substeps' in log,
        logged_sample=list(map(int,sample.groups())) if sample else None,
        boundary_counts=list(map(int,trials.groups())) if trials else None,
        field_solver_calls=int(solves[1]) if solves else None,
        last_reported_time_s=max((float(value)*factors[unit] for value,unit in times),default=None))


def run_case(study,name):
    """Run only a prepared immutable case, owning its PID and bounded cancellation."""
    study=Path(study);meta=json.loads((study/'study.json').read_text())
    if name not in meta['cases']: raise ValueError('Unknown case')
    directory=study/name;record=json.loads((directory/'provenance.json').read_text())
    if record['status']!='prepared': raise ValueError('Refusing to overwrite run')
    for path,expected in [(Path(meta['binary']),meta['binary_sha256']),(directory/'case.in',record['input_sha256']),
                          (directory/'particles.txt',record['particles_sha256']),(directory/'control-ids.npy',record['control_ids_sha256'])]:
        if sha256(path)!=expected: raise ValueError(f'Changed immutable input: {path}')
    env={key:value for key,value in os.environ.items() if not key.startswith('OPALX_TEST_')}
    env.update(OMP_NUM_THREADS='8',OMP_PROC_BIND='false')
    if record['stride'] is not None: env[ENVIRONMENT]=str(record['stride'])
    command=[meta['binary'],'--info','2','case.in'];started=time.monotonic()
    with (directory/'run.log').open('w') as log:
        child=subprocess.Popen(command,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT)
        record.update(status='running',pid=child.pid,command=command,mpi_ranks=1,omp_threads=8,
            environment={key:env.get(key) for key in ['OMP_NUM_THREADS','OMP_PROC_BIND',ENVIRONMENT]},
            start_unix_s=time.time())
        write_json(directory/'provenance.json',record)
        try:
            child.wait(timeout=record['timeout_s'])
            record['status']='completed' if child.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:
            child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill();child.wait()
            record['status']='timed_out'
    record.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-started,
        **parse_log((directory/'run.log').read_text()))
    expected=[record['stride'],record['actual_control_count']]
    record['selection_verified']=record['logged_sample']==expected if (record['stride'] or 0)>1 else record['logged_sample'] is None
    write_json(directory/'provenance.json',record)
    if not record['selection_verified'] or not record['boundary_controller_active']:
        raise RuntimeError(f'{name}: controller or sample validation failed')
    print(f'{name}: {record["status"]}; {record["elapsed_seconds"]:.2f}s',flush=True)
    return record


def final_frame(directory,meta,record,cof,section):
    """Final phase in one common ballistic reference frame, with no centroid subtraction."""
    duration = float(meta['duration_s'])
    final=read_final(directory/'case.h5',record['count'],meta['p0_mc'],duration)
    if not np.isclose(final['charge_C'],CHARGE,rtol=1e-12,atol=0): raise ValueError('Charge changed')
    with h5py.File(directory/'case.h5','r') as data:
        key=max((key for key in data if re.fullmatch(r'Step#\d+',key)),key=lambda key:int(key[5:]))
        ref_r=np.asarray(data[key].attrs['RefPartR'])
    phase=final['phase'];p=meta['p0_mc']*phase[:,[1,3,5]].copy();p[:,2]+=meta['p0_mc']
    rlab,plab=local_to_lab(phase[:,[0,2,4]],p,ref_r,final['refp'])
    reference=np.asarray(cof['position_m'])+duration*c*np.asarray(cof['momentum_mc'])/np.sqrt(1+meta['p0_mc']**2)
    result=phase_at_section(rlab,plab,dict(centre_m=reference,theta_rad=section['theta_rad']),meta['p0_mc'])
    return result,dict(reference_r_m=ref_r.tolist(),reference_p_mc=final['refp'].tolist(),
        charge_C=final['charge_C'],time_s=final['time_s'],snapshots=final['snapshots'])


def compare_phase(a,b,covariance,diagnostic_ids,selected):
    """Full-bunch moments and per-particle errors with a common, fixed C0 metric."""
    if a.shape!=b.shape: raise ValueError('Phase arrays must share IDs and shape')
    w=whitening(covariance);delta=a-b
    ca=np.cov(a,rowvar=False,bias=True);cb=np.cov(b,rowvar=False,bias=True)
    metric=np.linalg.norm(delta@w.T,axis=1)/np.sqrt(6)
    selections=dict(full=np.arange(len(a)),diagnostic=np.asarray(diagnostic_ids),selected=np.asarray(selected),
                    unselected=np.setdiff1d(np.arange(len(a)),selected))
    return dict(whitened_covariance_error=float(np.linalg.norm(w@(ca-cb)@w.T)/np.sqrt(6)),
        centroid_difference_over_initial_rms=(delta.mean(axis=0)/np.sqrt(np.diag(covariance))).tolist(),
        component_rms_difference=np.sqrt(np.mean(delta**2,axis=0)).tolist(),
        particle_error_quantiles={name:np.quantile(metric[index],[.5,.95,.99,1]).tolist() if len(index) else None for name,index in selections.items()})


def analyze(study):
    """Retain all raw endpoints and compare completed cases; largest sample is not truth."""
    study=Path(study);meta=json.loads((study/'study.json').read_text())
    cof=json.loads((study/'bare-cof.json').read_text());section=json.loads((study/'sections.json').read_text())[0]
    results={};rows=[];pairs=[]
    for name in meta['cases']:
        directory=study/name;record=json.loads((directory/'provenance.json').read_text())
        row=dict(case=name,status=record['status'],control_count=record['actual_control_count'],dt_ps=record['dt_s']*1e12,
            elapsed_seconds=record.get('elapsed_seconds'),last_reported_time_s=record.get('last_reported_time_s'),
            field_solver_calls=record.get('field_solver_calls'))
        if record['status']=='completed':
            if not record.get('selection_verified'): raise ValueError('Unverified runtime sample')
            phase,info=final_frame(directory,meta,record,cof,section)
            launch=np.load(study/record['initial_npz']);ids=launch['ids']
            if not np.array_equal(selected_ids(ids,record['stride']),np.load(directory/'control-ids.npy')): raise ValueError('Control ID inconsistency')
            cov=np.cov(phase,rowvar=False,bias=True)
            density=uniform_density_diagnostics(phase[:,[0,2,4]])
            write_json(directory/'analysis.json',dict(**info,mean=phase.mean(axis=0).tolist(),covariance=cov.tolist(),
                rms=np.std(phase,axis=0).tolist(),eigenemittances=eigenemittances(cov).tolist(),density=density))
            np.savez(directory/'diagnostics.npz',ids=launch['diagnostic_ids'],phase=phase[launch['diagnostic_ids']],mean=phase.mean(axis=0),covariance=cov)
            results[name]=(phase,record,launch['covariance_sample'],launch['diagnostic_ids'])
            row.update(density)
        rows.append(row)
    intended=[('control4-dt62p5ps','control32-dt62p5ps'),('control32-dt62p5ps','control139-dt62p5ps'),
        ('control32-dt62p5ps','control32-dt31p25ps'),('full-dt62p5ps','control139-dt62p5ps'),('smoke-unset','smoke-stride1')]
    for a,b in intended:
        if a not in results or b not in results: continue
        pa,ra,cov,diagnostic=results[a];pb,rb,_,_=results[b]
        stats=compare_phase(pa,pb,cov,diagnostic,np.load(study/a/'control-ids.npy'))
        stats.update(case_a=a,case_b=b,phase_bitwise_equal=bool(np.array_equal(pa,pb)),
            boundary_counts_equal=ra.get('boundary_counts')==rb.get('boundary_counts'))
        pairs.append(stats)
    frame=pd.DataFrame(rows);frame.to_csv(study/'cases.csv',index=False)
    write_json(study/'comparisons.json',pairs)
    pd.DataFrame([dict(case_a=p['case_a'],case_b=p['case_b'],whitened_covariance_error=p['whitened_covariance_error'],
        diagnostic_p99=p['particle_error_quantiles']['diagnostic'][2],all_particle_p99=p['particle_error_quantiles']['full'][2]) for p in pairs]).to_csv(study/'comparisons.csv',index=False)
    if pairs:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axis=plt.subplots(figsize=(9,4.5),layout='constrained')
        labels=[p['case_a']+' / '+p['case_b'] for p in pairs]
        axis.barh(labels,[p['whitened_covariance_error'] for p in pairs],color='#266a96')
        axis.set_xlabel(r'$\|C_0^{-1/2}(C_a-C_b)C_0^{-1/2}\|_F/\sqrt{6}$ at 40 ns')
        axis.grid(axis='x',alpha=.25);axis.set_axisbelow(True)
        fig.savefig(study/'covariance-comparisons.png',dpi=220);fig.savefig(study/'covariance-comparisons.pdf');plt.close(fig)
    return frame


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--binary',type=Path,default=REPO/'omp-build/src/opalx');p.add_argument('--output',type=Path)
    p=sub.add_parser('run');p.add_argument('study',type=Path);p.add_argument('case')
    p=sub.add_parser('analyze');p.add_argument('study',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.binary,args.output)
    elif args.command=='run':run_case(args.study,args.case)
    else:analyze(args.study)
