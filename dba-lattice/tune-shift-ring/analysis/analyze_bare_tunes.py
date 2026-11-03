"""Analyze completed device bare-tune runs and compare analytic/COF optics."""
import argparse
import json
import re
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.constants import c
from bare_tune_tools import cell_optics, laboratory_coordinates, cell_crossing_tune, matched_phase_difference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('runs', nargs='+', type=Path)
args = parser.parse_args()
for run in args.runs:
    provenance = json.loads((run/'provenance.json').read_text())
    if not provenance.get('completed'):
        raise RuntimeError(f'Run not complete: {run}')
    config = json.loads((run/'config.json').read_text())
    cof = json.loads((run/'bare-cof.json').read_text())
    if provenance['nominal_period_s'] <= 0:
        gamma = 1 + cof['energy_MeV']*1e6/cof['mass_eV']
        beta = np.sqrt(1-gamma**-2)
        provenance['nominal_period_s'] = config['cells']*sum(el['length_m'] for el in config['elements'])/(beta*c)
        provenance['period_metadata_correction'] = 'COF time_s is launch time; use circumference/speed. Original simulation dumped each step; particle transport unchanged.'
        (run/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    gaussian = provenance.get('distribution') == 'GAUSS'
    particle_count = provenance['particle_count'] if gaussian else len(provenance['amplitudes_m'])
    records, reference = [], []
    with h5py.File(run/'case.h5') as data:
        names = sorted(data, key=lambda k: int(k.split('#')[1]))
        raw_snapshots = len(names)
        stride = max(1, raw_snapshots//(provenance['requested_turns']*192))
        selected = names[::stride]
        if selected[-1] != names[-1]: selected.append(names[-1])
        for name in selected:
            step = data[name]
            ids = step['id'][:]
            if len(ids) != particle_count:
                raise RuntimeError('Unexpected particle loss')
            time = float(step.attrs['TIME'][0])
            order = np.argsort(ids)
            columns = {key: step[key][:][order] for key in ('x','y','z','px','py','pz')}
            for i, particle in enumerate(ids[order]):
                records.append(dict(time_s=time, id=int(particle), **{k:v[i] for k,v in columns.items()}))
            reference.append(dict(time_s=time, **{f'r{k}':step.attrs['RefPartR'][i] for i,k in enumerate('xyz')}, **{f'p{k}':step.attrs['RefPartP'][i] for i,k in enumerate('xyz')}))
    frame, ref = pd.DataFrame(records), pd.DataFrame(reference)
    frame.to_csv(run/'particles.csv', index=False)
    ref.to_csv(run/'reference.csv', index=False)
    poses = []
    for cell in range(1, config['cells']+1):
        line = re.search(rf'C{cell}_B1:.*?;', (run/'case.in').read_text()).group()
        values = {key:float(re.search(rf'\b{key}=([^,;]+)', line).group(1)) for key in ['X','Y','Z','THETA']}
        poses.append((np.array([values[k] for k in ['X','Y','Z']]), values['THETA']))
    internal_dt = provenance['dt_s']
    cap = min(el['length_m'] for el in config['elements'])/(4*c)
    while internal_dt > cap: internal_dt /= 2
    summary = dict(dt_s=provenance['dt_s'], internal_dt_s=internal_dt,
                   requested_turns=provenance['requested_turns'], raw_snapshots=raw_snapshots,
                   analysis_stride=stride, measurements=[])
    if not gaussian:
        central = frame[frame.id == 0]
        summary['maximum_center_offset_m'] = float(np.linalg.norm(central[['x','y','z']],axis=1).max())
    else:
        summary['distribution'] = 'GAUSS'
        summary['particle_count'] = particle_count
        summary['first_snapshot_statistics'] = frame[frame.time_s == frame.time_s.min()][['x','y','z','px','py','pz']].describe().to_dict()
        summary['excluded_small_signal'] = []
    summary['maximum_reference_y_m'] = float(ref.ry.abs().max())
    phases = []
    for particle in (sorted(frame.id.unique()) if gaussian else sorted(frame.id.unique())[1:]):
        samples = frame[frame.id == particle].reset_index(drop=True)
        lab_r, lab_p = laboratory_coordinates(ref[['rx','ry','rz']].to_numpy(), ref[['px','py','pz']].to_numpy(),
                    samples[['x','y','z']].to_numpy(), samples[['px','py','pz']].to_numpy())
        kinetic = np.sqrt(1+np.sum(lab_p**2,axis=1))-1
        energy_change = float(np.max(np.abs(kinetic/kinetic[0]-1)))
        for axis, j in [('x',0), ('y',2)]:
            amplitude = (float(samples[axis].iloc[0]) if gaussian else provenance['amplitudes_m'][int(particle)])
            if gaussian and abs(amplitude) < provenance['minimum_analysis_amplitude_m']:
                summary['excluded_small_signal'].append(dict(particle_id=int(particle),axis=axis,first_snapshot_position_m=amplitude))
                continue
            optics = cell_optics(config, axis)
            result = cell_crossing_tune(samples.time_s.to_numpy(), lab_r, lab_p, poses,
                        provenance['nominal_period_s'], optics['beta'], optics['alpha'], axis, bend_radius=2.)
            low_order = cell_crossing_tune(samples.time_s.to_numpy(), lab_r, lab_p, poses,
                        provenance['nominal_period_s'], optics['beta'], optics['alpha'], axis,
                        interpolation_order=3, bend_radius=2.)
            interpolation_difference, common_crossings = matched_phase_difference(
                    result, low_order, provenance['nominal_period_s'], config['cells'])
            block = np.asarray(cof['matrix'])[j:j+2,j:j+2]
            eigenvalues = np.linalg.eigvals(block)
            eigenvalue = eigenvalues[np.sign(eigenvalues.imag) == np.sign(block[0,1])][0]
            cof_tune = (np.angle(eigenvalue)/(2*np.pi))%1 + np.floor(optics['tune'])
            summary['measurements'].append(dict(axis=axis, particle_id=int(particle),
                position_label='first snapshot' if gaussian else 'initial',
                initial_position_m=amplitude, amplitude_m=abs(amplitude), analytic_tune=float(optics['tune']),
                interpolation_order_difference=interpolation_difference,
                interpolation_common_crossings=common_crossings,
                cof_eigenvalue_modulus_error=float(abs(eigenvalue)-1), cof_stability_flag=cof['stability'],
                cof_tune=float(cof_tune), tracking_tune=result['tune'], half_tunes=result['half_tunes'],
                tracking_minus_analytic=float(result['tune']-optics['tune']),
                tracking_minus_cof=float(result['tune']-cof_tune), crossings=len(result['time']),
                phase_residual_rms_turns=result['phase_residual_rms_turns'],
                independent_tune=result['independent_tune'],
                frequency_estimator_difference=result['independent_tune']-result['tune'],
                relative_energy_variation=energy_change))
            for n,t,phase,cell,x,slope in zip(result['turns'], result['time'], result['phase'], result['cells'], result['coordinate'], result['slope']):
                phases.append(dict(turn=n,time_s=t,axis=axis,particle_id=int(particle),cell=int(cell),coordinate_m=x,slope=slope,phase_turns=phase))
    pd.DataFrame(phases).to_csv(run/'section_phases.csv', index=False)
    (run/'tunes.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(str(run), json.dumps(summary, indent=2), flush=True)
