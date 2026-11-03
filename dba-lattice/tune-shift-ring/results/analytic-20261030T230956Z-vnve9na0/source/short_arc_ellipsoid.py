"""Finite-emittance ellipsoid PIC convergence in an unbroken DBA straight.

One MPI rank, eight OpenMP threads, 6.25 ns entirely inside the first D1.
This tests short-arc covariance evolution, not periodic matching or tunes.
The uniform-density rms closure is an approximation to the sampled kinetic beam.
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
from rf_ring_study import cyclic_d1_lattice, rf_lattice, tracking_deck, straight_particles
from match_ellipsoid_ring import cell_pass, eigenemittances

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
CALIBRATION = ROOT/'results/rf-ring-s3a01yyz'
MATCH = ROOT/'results/ellipsoid-match-stable-long-d1center'
DURATION = 6.25e-9
CHARGE = 160e-12
CASES = (
    ('bare-n131072-m32', 131072, 32, 6.25e-11, 0),
    ('sc-n131072-m32', 131072, 32, 6.25e-11, 160),
    ('sc-n131072-m32-fine', 131072, 32, 3.125e-11, 160),
    ('sc-n131072-m64', 131072, 64, 6.25e-11, 160),
    ('sc-n524288-m64', 524288, 64, 6.25e-11, 160),
)


def phase_from_local(position, momentum, p0):
    """H5 straight reference-frame coordinates to (x,px/p0,y,py/p0,z,delta)."""
    if p0 <= 0 or np.shape(position) != np.shape(momentum) or np.shape(position)[-1] != 3:
        raise ValueError('Require matching Cartesian arrays and positive reference momentum')
    result = np.empty(np.shape(position)[:-1]+(6,))
    result[..., [0, 2, 4]] = position
    result[..., [1, 3, 5]] = np.asarray(momentum)/p0
    result[..., 5] -= 1
    return result


def ballistic_phase(phase, p0, duration):
    """Exact relativistic zero-field motion at common time, with a straight reference."""
    if p0 <= 0 or duration < 0 or np.shape(phase)[-1] != 6:
        raise ValueError('Invalid ballistic phase-space propagation')
    phase = np.asarray(phase)
    result = phase.copy()
    momentum = p0*phase[:, [1, 3, 5]]
    momentum[:, 2] += p0
    velocity = c*momentum/np.sqrt(1+np.sum(momentum**2, axis=1))[:, None]
    velocity[:, 2] -= c*p0/np.sqrt(1+p0*p0)
    result[:, [0, 2, 4]] += duration*velocity
    return result


def predictions(covariance, charge, p0, duration):
    """Uniform-ellipsoid closure and exact paraxial drift of the same covariance."""
    gamma = np.sqrt(1+p0*p0)
    beta = p0/gamma
    distance = beta*c*duration
    drift = np.eye(6)
    drift[0, 1] = drift[2, 3] = distance
    drift[4, 5] = distance/gamma**2
    element = dict(name='D1-short', length_m=distance, kx_m2=0., ky_m2=0.)
    closure = cell_pass(covariance, [element], charge, beta, gamma,
                        rf_slope=0., rtol=2e-11, with_map=False)[0]
    return closure, drift@covariance@drift.T


def covariance_comparison(actual, expected):
    """Dimensionless full covariance error, rms errors and sorted eigenemittances."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    scale = np.sqrt(np.diag(expected))
    if actual.shape != (6, 6) or expected.shape != (6, 6) or np.any(scale <= 0):
        raise ValueError('Positive 6x6 covariance required')
    normalized = (actual-expected)/np.outer(scale, scale)
    return dict(normalized_covariance_error=normalized.tolist(),
                normalized_frobenius_error=float(np.linalg.norm(normalized)),
                maximum_normalized_error=float(np.max(abs(normalized))),
                rms_relative_error=(np.sqrt(np.diag(actual))/scale-1).tolist(),
                eigenemittance_relative_error=(eigenemittances(actual)/eigenemittances(expected)-1).tolist())


def make_deck(lattice, count, mesh, dt, charged):
    """Final H5 only; initial state is separately preserved exactly in NPZ/FROMFILE."""
    steps = round(DURATION/dt)
    if not np.isclose(steps*dt, DURATION, rtol=0, atol=1e-24):
        raise ValueError('Timestep must divide the common physical duration')
    lattice = re.sub(r'(PSDUMPFREQ|STATDUMPFREQ)=\d+', r'\1=1000000000', lattice)
    return tracking_deck(lattice, count, dt, solver='OPEN' if charged else 'NONE',
                         charge=CHARGE, mesh=mesh, steps=steps)


def prepare(binary, output=None, cases=CASES):
    """Freeze calibrated lattice, input samples, source snapshots, and checksums."""
    study = Path(output) if output else Path(tempfile.mkdtemp(prefix='short-arc-ellipsoid-', dir=ROOT/'results'))
    study.mkdir(parents=True, exist_ok=True)
    if (study/'study.json').exists():
        raise ValueError('Study already exists')
    cof = json.loads((CALIBRATION/'bare-cof.json').read_text())
    rf = json.loads((CALIBRATION/'synchronous-3/rf.json').read_text())
    sections = json.loads((CALIBRATION/'sections.json').read_text())
    base, derived_sections = cyclic_d1_lattice((ROOT/'bare-gaussian-cof.in').read_text())
    if not np.allclose(sections[0]['centre_m'], derived_sections[0]['centre_m'], atol=1e-14):
        raise ValueError('Calibration and input straight sections differ')
    lattice, _ = rf_lattice(base, rf['frequency_hz'], rf['amplitude_mvm'], rf['phases_rad'], dump=1000000000)
    p0 = float(np.linalg.norm(cof['momentum_mc']))
    gamma = np.sqrt(1+p0*p0)
    source = study/'source'
    source.mkdir()
    changed = subprocess.check_output(['git', 'ls-files', '--modified', '--others', '--exclude-standard', '--', 'src', 'unit_tests'], cwd=REPO, text=True).splitlines()
    for relative in dict.fromkeys(changed):
        path = REPO/relative
        if path.is_file():
            target = source/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    (source/'working-tree.diff').write_text(subprocess.check_output(['git', 'diff'], cwd=REPO, text=True))
    for path in Path(__file__).parent.glob('*.py'):
        target = source/'analysis'/path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    for path in [CALIBRATION/'bare-cof.json', CALIBRATION/'sections.json', CALIBRATION/'synchronous-3/rf.json',
                 ROOT/'bare-gaussian-cof.in', MATCH/'match-0pc.npz', MATCH/'match-160pc.npz', MATCH/'config.json']:
        shutil.copy2(path, study/path.name)
    cache = {}
    for name, count, mesh, dt, model_charge in cases:
        key = (count, model_charge)
        if key not in cache:
            covariance = np.load(MATCH/f'match-{model_charge}pc.npz')['covariance']
            phase = generate_ellipsoid(covariance, count)
            particles = straight_particles(phase, cof['position_m'], cof['momentum_mc'], sections[0]['theta_rad'])
            prefix = study/f'initial-{model_charge}pc-n{count}'
            write_particles(prefix.with_suffix('.txt'), particles)
            sample = np.cov(phase, rowvar=False, ddof=0)
            np.savez(prefix.with_suffix('.npz'), phase=phase, covariance_ideal=covariance, covariance_sample=sample)
            cache[key] = prefix
        directory = study/name
        directory.mkdir()
        shutil.copy2(cache[key].with_suffix('.txt'), directory/'particles.txt')
        (directory/'case.in').write_text(make_deck(lattice, count, mesh, dt, model_charge != 0))
        write_json(directory/'provenance.json', dict(status='prepared', count=count, mesh=mesh, dt_s=dt,
            steps=round(DURATION/dt), model_charge_C=model_charge*1e-12, charge_C=CHARGE,
            initial_npz=str(cache[key].with_suffix('.npz').relative_to(study)),
            input_sha256=sha256(directory/'case.in'), particles_sha256=sha256(directory/'particles.txt')))
    write_json(study/'study.json', dict(binary=str(Path(binary).resolve()), binary_sha256=sha256(binary),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        duration_s=DURATION, distance_m=float(c*p0/gamma*DURATION), p0_mc=p0,
        charge_C=CHARGE, mpi_ranks=1, omp_threads=8, seed=20260911, cases=[row[0] for row in cases],
        scope='Short unbroken first-D1 arc only: no one-cell, matched-state, or tune validation.',
        initial_snapshot='Exact NPZ/FROMFILE at t=0; tracker writes final H5 only.',
        source_sha256={str(path.relative_to(study)):sha256(path) for path in source.rglob('*') if path.is_file()}))
    print(study, flush=True)
    return study


def run_case(study, name, timeout=600):
    """Run a prepared immutable one-rank OMP8 case; keep failed artifacts."""
    study = Path(study)
    meta = json.loads((study/'study.json').read_text())
    if name not in meta['cases']:
        raise ValueError('Unknown case')
    directory = study/name
    record = json.loads((directory/'provenance.json').read_text())
    if record['status'] != 'prepared':
        raise ValueError('Refusing to overwrite a run')
    for path, expected in [(Path(meta['binary']),meta['binary_sha256']),
                           (directory/'case.in',record['input_sha256']),
                           (directory/'particles.txt',record['particles_sha256'])]:
        if sha256(path) != expected:
            raise ValueError(f'Changed immutable input: {path}')
    command = [meta['binary'], '--info', '1', 'case.in']
    started = time.monotonic()
    with (directory/'run.log').open('w') as log:
        child = subprocess.Popen(command, cwd=directory, env={**os.environ,'OMP_NUM_THREADS':'8','OMP_PROC_BIND':'false'}, stdout=log, stderr=subprocess.STDOUT)
        record.update(status='running', command=command, pid=child.pid, mpi_ranks=1, omp_threads=8, timeout_s=timeout)
        write_json(directory/'provenance.json',record)
        try:
            child.wait(timeout=timeout)
            record['status'] = 'completed' if child.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            record['status']='timed_out'
    record.update(returncode=child.returncode, elapsed_seconds=time.monotonic()-started)
    log = (directory/'run.log').read_text()
    record['boundary_controller_active'] = 'synchronized ordinary Boris/PIC substeps' in log
    counts = re.search(r'Boris boundary trials=(\d+), rejected=(\d+)',log)
    record['boundary_counts'] = list(map(int,counts.groups())) if counts else None
    write_json(directory/'provenance.json',record)
    if record['status'] != 'completed' or not record['boundary_controller_active']:
        raise RuntimeError(f'{name}: {record["status"]}, controller={record["boundary_controller_active"]}')
    print(f'{name}: {record["elapsed_seconds"]:.2f}s', flush=True)
    return record


def read_final(path, count, p0, duration=DURATION):
    """Read final ID-sorted fields/phase; reject loss, nonfinite data and wrong clock."""
    with h5py.File(path,'r') as data:
        keys = sorted((key for key in data if re.fullmatch(r'Step#\d+',key)),key=lambda key:int(key[5:]))
        if not keys: raise ValueError('No H5 particle snapshots')
        group = data[keys[-1]]
        order = np.argsort(group['id'][:])
        ids = group['id'][:][order]
        if not np.array_equal(ids,np.arange(count)):
            raise ValueError('Particle loss or unexpected IDs')
        arrays = [np.column_stack([group[key][:][order] for key in names]) for names in
                  [('x','y','z'),('px','py','pz'),('Ex','Ey','Ez'),('Bx','By','Bz')]]
        timestamp = float(np.asarray(group.attrs['TIME']).ravel()[0])
        charge = float(np.sum(group['q'][:]))
        refp = np.asarray(group.attrs['RefPartP'])
        if not all(np.isfinite(array).all() for array in arrays+[refp]) or not np.isfinite(charge):
            raise ValueError('Nonfinite phase or fields')
        if not np.isclose(timestamp,duration,rtol=0,atol=1e-20):
            raise ValueError('Incorrect final physical time')
        return dict(phase=phase_from_local(arrays[0],arrays[1],p0), charge_C=charge,
                    refp=refp, snapshots=len(keys), electric=arrays[2], magnetic=arrays[3], time_s=timestamp)


def analyze(study):
    """Preserve full sample covariances and compare both ideal and sampled closures."""
    study = Path(study)
    meta = json.loads((study/'study.json').read_text())
    rows = []
    results = {}
    for name in meta['cases']:
        directory=study/name
        record=json.loads((directory/'provenance.json').read_text())
        if record['status'] != 'completed': continue
        # Also parse existing logs when analyzing a study prepared by an earlier
        # version of the runner's diagnostic extraction.
        counts = re.search(r'Boris boundary trials=(\d+), rejected=(\d+)', (directory/'run.log').read_text()) if (directory/'run.log').exists() else None
        boundary_counts = list(map(int, counts.groups())) if counts else record.get('boundary_counts')
        if boundary_counts and boundary_counts != [record['steps'], 0]:
            raise ValueError('Short straight unexpectedly required boundary subdivision')
        final=read_final(directory/'case.h5',record['count'],meta['p0_mc'])
        if not np.isclose(final['charge_C'],CHARGE,rtol=1e-12,atol=0):
            raise ValueError('Fixed total bunch charge changed')
        initial=np.load(study/record['initial_npz'])
        actual=np.cov(final['phase'],rowvar=False,ddof=0)
        ideal=predictions(initial['covariance_ideal'],record['model_charge_C'],meta['p0_mc'],DURATION)
        sample=predictions(initial['covariance_sample'],record['model_charge_C'],meta['p0_mc'],DURATION)
        ballistic=np.cov(ballistic_phase(initial['phase'],meta['p0_mc'],DURATION),rowvar=False,ddof=0)
        comparisons={key:covariance_comparison(actual,value) for key,value in
            [('ideal_closure',ideal[0]),('sample_closure',sample[0]),('ideal_paraxial_drift',ideal[1]),
             ('sample_paraxial_drift',sample[1]),('sample_exact_ballistic',ballistic)]}
        z=.5+meta['distance_m']+final['phase'][:,4]
        if not np.all((z>0)&(z<1)):
            raise ValueError('Bunch did not remain inside the unbroken D1')
        eigen_initial=eigenemittances(initial['covariance_sample'])
        eigen_final=eigenemittances(actual)
        output=dict(covariance_final=actual.tolist(), covariance_initial_sample=initial['covariance_sample'].tolist(),
            covariance_initial_ideal=initial['covariance_ideal'].tolist(),
            covariance_closure_ideal=ideal[0].tolist(), covariance_closure_sample=sample[0].tolist(),
            covariance_ballistic_sample=ballistic.tolist(), comparisons=comparisons,
            centroid_final=np.mean(final['phase'],axis=0).tolist(), rms_final=np.sqrt(np.diag(actual)).tolist(),
            eigenemittance_initial=eigen_initial.tolist(),eigenemittance_final=eigen_final.tolist(),
            eigenemittance_relative_change=(eigen_final/eigen_initial-1).tolist(),
            total_charge_C=final['charge_C'], particles=record['count'], all_finite=True,
            final_time_s=final['time_s'], h5_snapshots=final['snapshots'], d1_z_bounds_m=[float(z.min()),float(z.max())],
            boundary_counts=boundary_counts,
            max_electric_Vpm=float(np.max(np.linalg.norm(final['electric'],axis=1))),
            max_magnetic_T=float(np.max(np.linalg.norm(final['magnetic'],axis=1))),
            limitation=meta['scope'])
        if record['model_charge_C'] and output['max_electric_Vpm']==0:
            raise ValueError('Charged case has zero self-field')
        write_json(directory/'analysis.json',output)
        results[name] = output
        row=dict(case=name, count=record['count'], mesh=record['mesh'], dt_s=record['dt_s'],
                 elapsed_s=record['elapsed_seconds'], charge_C=final['charge_C'],
                 sample_closure_normalized_error=comparisons['sample_closure']['normalized_frobenius_error'],
                 exact_ballistic_normalized_error=comparisons['sample_exact_ballistic']['normalized_frobenius_error'])
        for i,axis in enumerate(['x','px','y','py','z','delta']):
            row[f'{axis}_rms_relative_closure_error']=comparisons['sample_closure']['rms_relative_error'][i]
        rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(study/'summary.csv',index=False)
    write_json(study/'summary.json',dict(scope=meta['scope'],cases=rows))
    convergence = {}
    for label, coarse, fine in [
            ('timestep','sc-n131072-m32','sc-n131072-m32-fine'),
            ('mesh','sc-n131072-m32','sc-n131072-m64'),
            ('particles','sc-n131072-m64','sc-n524288-m64')]:
        if coarse not in results or fine not in results: continue
        left, right = results[coarse], results[fine]
        lc, rc = np.asarray(left['covariance_final']), np.asarray(right['covariance_final'])
        lp, rp = np.asarray(left['covariance_closure_sample']), np.asarray(right['covariance_closure_sample'])
        scale = np.sqrt(np.diag(rp))
        residual_difference = ((lc-lp)-(rc-rp))/np.outer(scale,scale)
        convergence[label] = dict(coarse=coarse, fine=fine, final_difference=covariance_comparison(lc,rc),
            closure_residual_difference_normalized=residual_difference.tolist(),
            closure_residual_difference_norm=float(np.linalg.norm(residual_difference)))
    write_json(study/'convergence.json',convergence)
    print(frame.to_string(index=False),flush=True)
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=REPO/'omp-build/src/opalx')
    parser.add_argument('--study',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run',default=None)
    parser.add_argument('--analyze',action='store_true')
    parser.add_argument('--timeout',type=float,default=600)
    args=parser.parse_args()
    study=args.study or prepare(args.binary)
    if args.prepare_only: return
    if args.run:
        names=json.loads((study/'study.json').read_text())['cases'] if args.run=='all' else [args.run]
        for name in names: run_case(study,name,args.timeout)
    if args.analyze: analyze(study)


if __name__=='__main__': main()
