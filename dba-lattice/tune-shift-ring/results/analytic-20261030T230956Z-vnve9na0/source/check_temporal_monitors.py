"""Validate existing temporal monitors against dense same-run straight snapshots.

One MPI rank/one OpenMP thread. MONITOR has a mandatory .01 m nominal length;
each D1 is replaced by .495 m drift + .01 m monitor + .495 m drift. All magnet
and RF poses, total nominal length, and the physical monitor-centre section stay
fixed. Monitor positions are reference-relative at group TIME, not row `time`.
Only the field-free, planar D1 section is validated here.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import h5py
import numpy as np
import pandas as pd

from large_ellipsoid_benchmark import sha256, write_json, write_particles
from rf_ring_study import (ROOT, read_pose, pose_string, tracking_deck, execute,
    rf_lattice, straight_particles, normal_mode_probes, read_history,
    plane_crossings)

DEFAULT_STUDY = ROOT/'results/rf-ring-s3a01yyz'


def insert_monitors(lattice):
    """Replace each D1 by equal-total-length passive pieces centred on its plane."""
    for cell in range(1, 7):
        name = f'C{cell}_D1'
        match = re.search(r'^'+name+r':\s*DRIFT\b[^;]*;', lattice, re.MULTILINE)
        if match is None:
            raise ValueError(f'Missing {name}')
        if abs(float(re.search(r'\bL=([^,;]+)', match[0])[1])-1) > 1e-14:
            raise ValueError('Monitor placement assumes a 1 m D1 drift')
        origin, theta = read_pose(match[0])
        tangent = np.array([np.sin(theta), 0., np.cos(theta)])
        replacement = '\n'.join([
            f'{name}A: DRIFT, L=.495, APERTURE="RECTANGLE(1.0,0.2)", '+pose_string(origin, theta)+';',
            f'C{cell}_MON: MONITOR, L=.01, TYPE=TEMPORAL, OUTFN="monitor-{cell}", '+pose_string(origin+.5*tangent, theta)+';',
            f'{name}B: DRIFT, L=.495, APERTURE="RECTANGLE(1.0,0.2)", '+pose_string(origin+.505*tangent, theta)+';'])
        lattice = lattice[:match.start()]+replacement+lattice[match.end():]
        line = re.search(r'^LATTICE:\s*RING\b[^;]*;', lattice, re.MULTILINE)
        changed = re.sub(r'\b'+name+r'\b', f'{name}A, C{cell}_MON, {name}B', line[0])
        lattice = lattice[:line.start()]+changed+lattice[line.end():]
    return lattice


def nominal_length(lattice):
    """Sum explicitly numeric body lengths of the members used by this RING deck."""
    line = re.search(r'^LATTICE:\s*RING\s*=\s*\(([^)]*)\);', lattice, re.MULTILINE)
    result = 0.
    for name in line[1].split(','):
        command = re.search(r'^'+name.strip()+r':[^;]*;', lattice, re.MULTILINE)[0]
        if ': MARKER' in command:
            continue
        result += float(re.search(r'\bL\s*=\s*([^,;]+)', command)[1])
    return result


def local_to_lab(position, momentum, reference_r, reference_p):
    """Transform planar reference-following H5 coordinates to lab arrays."""
    direction = np.asarray(reference_p)/np.linalg.norm(reference_p)
    if abs(direction[1]) > 1e-12:
        raise ValueError('This check supports planar reference trajectories')
    rotation = np.column_stack(([direction[2], 0., -direction[0]], [0., 1., 0.], direction))
    return np.asarray(position)@rotation.T+reference_r, np.asarray(momentum)@rotation.T


def phase_at_section(position, momentum, section, p0):
    """Static section coordinates (x,px/p0,y,py/p0,zhead,delta)."""
    theta = section['theta_rad']
    rotation = np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]])
    r = (position-section['centre_m'])@rotation
    p = momentum@rotation/p0
    return np.column_stack((r[:,0], p[:,0], r[:,1], p[:,1], r[:,2], p[:,2]-1))


def dense_phase_at(history, time_s, section, p0):
    """Interpolate lab particles only within a single field-free D1 straight."""
    i = np.searchsorted(history['t'], time_s)-1
    if i < 0 or i+1 >= len(history['t']):
        raise ValueError('Dense history does not bracket monitor time')
    fraction = (time_s-history['t'][i])/(history['t'][i+1]-history['t'][i])
    arrays = []
    for j in [i, i+1]:
        r, p = local_to_lab(history['local_r'][j], history['local_p'][j], history['r'][j], history['p'][j])
        arrays.append(phase_at_section(r,p,section,p0))
    if max(abs(arrays[0][:,4]).max(), abs(arrays[1][:,4]).max()) > .4:
        raise ValueError('Particle interpolation extends outside field-free D1')
    return (1-fraction)*arrays[0]+fraction*arrays[1]


def ordered_keys(data):
    """Return only numeric H5Part snapshot groups in stored-passage order."""
    return sorted((key for key in data if re.fullmatch(r'Step#\d+',key)),key=lambda key:int(key[5:]))


def read_monitor_group(group, section, p0):
    """Decode monitor rows in ID order using reference attributes and group TIME."""
    order = np.argsort(group['id'][:])
    ids = group['id'][:][order]
    position = np.column_stack([group[key][:][order] for key in ['x','y','z']])
    momentum = np.column_stack([group[key][:][order] for key in ['px','py','pz']])
    r,p = local_to_lab(position,momentum,group.attrs['RefPartR'],group.attrs['RefPartP'])
    return dict(ids=ids,phase=phase_at_section(r,p,section,p0),
        time_s=float(np.asarray(group.attrs['TIME']).ravel()[0]),row_times_s=group['time'][:][order])


def compare(directory, study):
    """Compare all monitor particles, empirical moments and primary mode phase."""
    history = read_history(directory/'case.h5')
    sections = json.loads((study/'sections.json').read_text())
    cof = json.loads((study/'bare-cof.json').read_text())
    p0 = np.linalg.norm(cof['momentum_mc'])
    inverse = np.load(study/'analytic/match-0pc.npz')['normal_mode_inverse']
    launch = json.loads((directory/'launch.json').read_text())
    rows=[]; events=[]
    for cell,section in enumerate(sections):
        crossings = plane_crossings(history,np.array(section['centre_m']),section['theta_rad'])
        theta=section['theta_rad']; tangent=np.array([np.sin(theta),0.,np.cos(theta)])
        # A final directed return may stop exactly on the plane, without a
        # strict sign-changing bracket. Include that independently verified end.
        if abs((history['r'][-1]-section['centre_m'])@tangent)<1e-10 and history['p'][-1]@tangent>0:
            crossings.append(dict(time_s=float(history['t'][-1])))
        with h5py.File(directory/f'monitor-{cell+1}.h5','r') as data:
            for key in ordered_keys(data):
                monitor = read_monitor_group(data[key],section,p0)
                if not np.array_equal(monitor['ids'],history['ids']):
                    raise ValueError('Monitor does not contain the full particle ID set')
                time_s = monitor['time_s']
                if time_s > history['t'][-1]+1e-15:
                    raise ValueError('Monitor extrapolated beyond the completed tracking interval')
                dense = dense_phase_at(history,time_s,section,p0)
                delta = monitor['phase']-dense
                monitor_modes=(monitor['phase']-monitor['phase'][0])@inverse.T
                dense_modes=(dense-dense[0])@inverse.T
                phase_errors=[]
                for particle,label in enumerate(launch['labels']):
                    if label=='centre': continue
                    mode='xyz'.index(label[0])
                    a=monitor_modes[particle,2*mode]+1j*monitor_modes[particle,2*mode+1]
                    b=dense_modes[particle,2*mode]+1j*dense_modes[particle,2*mode+1]
                    phase_errors.append(abs(np.angle(a/b)))
                moment_delta=np.cov(monitor['phase'],rowvar=False,bias=True)-np.cov(dense,rowvar=False,bias=True)
                rows.append(dict(cell=cell+1,step=key,time_s=time_s,particles=len(monitor['ids']),
                    max_position_error_m=float(abs(delta[:,[0,2,4]]).max()),
                    max_momentum_error_p0=float(abs(delta[:,[1,3,5]]).max()),
                    max_centroid_error=float(abs(delta.mean(axis=0)).max()),
                    max_covariance_error=float(abs(moment_delta).max()),
                    max_primary_phase_error_rad=float(max(phase_errors)),
                    row_minus_group_time_min_s=float((monitor['row_times_s']-time_s).min()),
                    row_minus_group_time_max_s=float((monitor['row_times_s']-time_s).max()),
                    nearest_dense_crossing_error_s=float(min(abs(e['time_s']-time_s) for e in crossings))))
                events.append((time_s,cell,monitor['phase']))
    frame=pd.DataFrame(rows); frame.to_csv(directory/'monitor-comparison.csv',index=False)
    events.sort(key=lambda event:event[0])
    np.savez(directory/'monitor-observations.npz',t_s=np.array([e[0] for e in events]),cells=np.array([e[1] for e in events]),phase=np.array([e[2] for e in events]),ids=history['ids'])
    raw_bytes=sum((directory/f'monitor-{i}.h5').stat().st_size for i in range(1,7))
    summary=dict(particles=int(len(history['ids'])),passages=len(frame),monitor_bytes=raw_bytes,
        max_electric_V_per_m=float(np.linalg.norm(history['electric'],axis=-1).max()),
        max_position_error_m=float(frame.max_position_error_m.max()),
        max_momentum_error_p0=float(frame.max_momentum_error_p0.max()),
        max_primary_phase_error_rad=float(frame.max_primary_phase_error_rad.max()),
        max_row_time_error_s=float(max(abs(frame.row_minus_group_time_min_s).max(),abs(frame.row_minus_group_time_max_s).max())),
        max_crossing_time_error_s=float(frame.nearest_dense_crossing_error_s.max()),
        final_time_s=float(history['t'][-1]),binary_sha256=json.loads((directory/'provenance.json').read_text())['binary_sha256'],
        group_time_required=True,validated_geometry='Planar field-free D1; position shifted to group TIME. Curved sections and PIC kicks require separate validation.')
    write_json(directory/'summary.json',summary)
    return summary


def run(study=DEFAULT_STUDY, dt=6.25e-11, fixed=False):
    """Prepare and execute a bounded two-turn thirteen-probe monitor study."""
    binary=ROOT.parents[2]/'omp-build/src/opalx'
    directory=Path(tempfile.mkdtemp(prefix='temporal-monitor-',dir=ROOT/'results'))
    print(directory,flush=True)
    rf=json.loads((study/'synchronous-3/rf.json').read_text())
    lattice,_=rf_lattice((study/'bare-gaussian-cof.in').read_text(),rf['frequency_hz'],rf['amplitude_mvm'],rf['phases_rad'],dump=16)
    monitored=insert_monitors(lattice)
    if abs(nominal_length(lattice)-nominal_length(monitored))>1e-12:
        raise ValueError('Nominal circumference changed')
    if fixed: monitored=monitored.replace('REPARTFREQ=0','REPARTFREQ=10')
    phase,labels=normal_mode_probes(np.load(study/'analytic/match-0pc.npz')['normal_mode_basis'],1e-5)
    cof=json.loads((study/'bare-cof.json').read_text())
    sections=json.loads((study/'sections.json').read_text())
    write_particles(directory/'particles.txt',straight_particles(phase,cof['position_m'],cof['momentum_mc'],sections[0]['theta_rad']))
    write_json(directory/'launch.json',dict(labels=labels,phase=phase.tolist(),rf=rf))
    (directory/'case.in').write_text(tracking_deck(monitored,len(phase),dt,turns=2))
    repo=ROOT.parents[2]
    write_json(directory/'study.json',dict(source_study=str(study),dt_s=dt,nominal_length_m=nominal_length(monitored),nominal_unmonitored_length_m=nominal_length(lattice),
        fixed_step=fixed,revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),mpi_ranks=1,omp_threads=1))
    (directory/'source.diff').write_text(subprocess.check_output(['git','diff'],cwd=repo,text=True))
    for name in ['src/AbsBeamline/Monitor.cpp','src/Algorithms/TrackReferenceStep.cpp','src/Structure/LossDataSink.cpp','src/Elements/OpalMonitor.cpp','src/Lines/Ring.cpp']:
        target=directory/'source'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(repo/name,target)
    shutil.copy2(__file__,directory/Path(__file__).name)
    execute(directory,binary,timeout=90,expected_boundary_control=not fixed)
    print(json.dumps(compare(directory,study),indent=2),flush=True)
    return directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=DEFAULT_STUDY)
    parser.add_argument('--dt',type=float,default=6.25e-11)
    parser.add_argument('--fixed',action='store_true')
    parser.add_argument('--analyze',type=Path)
    args=parser.parse_args()
    if args.analyze: print(json.dumps(compare(args.analyze,args.study),indent=2))
    else: run(args.study,args.dt,args.fixed)

if __name__=='__main__': main()
