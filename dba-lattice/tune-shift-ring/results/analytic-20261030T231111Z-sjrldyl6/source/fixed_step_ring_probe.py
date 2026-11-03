"""Check the existing fixed-step Boris mode against the calibrated RF-ring model.

REPARTFREQ=10 is OPALX's ordinary setting. It prevents midpoint retries;
spatial ring field selection and midpoint PIC are retained. Domain redistribution
returns immediately on one MPI rank. No new input parameter or solver is used.
The zero-charge probes establish timestep error before any large PIC run.
"""
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

from large_ellipsoid_benchmark import ROOT, sha256, write_json, write_particles
from rf_ring_study import (execute, normal_mode_probes, read_history, rf_lattice,
                           section_observations, straight_particles, tracking_deck)
from ellipsoid_observables import individual_mode_tunes


if __name__ == '__main__':
    source = ROOT / 'results/rf-ring-s3a01yyz'
    binary = ROOT.parents[2] / 'omp-build/src/opalx'
    study = Path(tempfile.mkdtemp(prefix='fixed-step-rf-', dir=ROOT / 'results'))
    for name in ['fixed_step_ring_probe.py','rf_ring_study.py','ellipsoid_observables.py']:
        shutil.copy2(Path(__file__).with_name(name), study / name)
    cof = json.loads((source / 'bare-cof.json').read_text())
    sections = json.loads((source / 'sections.json').read_text())
    rf = json.loads((source / 'synchronous-3/rf.json').read_text())
    model = np.load(source / 'analytic/match-0pc.npz')
    phase, labels = normal_mode_probes(model['normal_mode_basis'], 1e-5)
    particles = straight_particles(phase, cof['position_m'], cof['momentum_mc'], sections[0]['theta_rad'])
    expected = abs(model['signed_cell_phases']) * 6 / (2*np.pi)
    write_json(study / 'study.json', dict(binary=str(binary), binary_sha256=sha256(binary),
              source=str(source), mpi_ranks=1, omp_threads=1, turns=32,
              repartfreq=10, model_tunes_xyz=expected.tolist(),
              note='Existing fixed-step Boris mode; no public API or numerical kernel change'))
    print(study, flush=True)
    rows=[]
    for dt in [4e-10, 2e-10, 1e-10]:
        directory=study / f'dt{dt*1e12:g}ps'
        directory.mkdir()
        lattice,_=rf_lattice((source / 'bare-gaussian-cof.in').read_text(),
                              rf['frequency_hz'],rf['amplitude_mvm'],rf['phases_rad'],dump=4)
        lattice=re.sub(r'REPARTFREQ\s*=\s*0\b', 'REPARTFREQ=10', lattice)
        (directory / 'case.in').write_text(tracking_deck(lattice,len(phase),dt,32))
        write_particles(directory / 'particles.txt',particles)
        write_json(directory / 'launch.json',dict(phase=phase.tolist(),labels=labels,rf=rf))
        execute(directory,binary,timeout=600,expected_boundary_control=False)
        history=read_history(directory / 'case.h5')
        observed=section_observations(history,sections,np.linalg.norm(cof['momentum_mc']),phase)
        np.savez(directory / 'section-observations.npz',**observed)
        result=individual_mode_tunes(observed['phase'],observed['turn_coordinate'],
                 model['normal_mode_inverse'],model['signed_cell_phases'],observed['phase'][:,0])
        np.savez(directory / 'mode-tunes.npz',ids=observed['ids'],**result)
        for i,label in enumerate(labels):
            if label=='centre': continue
            mode='xyz'.index(label[0])
            row=dict(dt_s=dt,probe=label,mode=label[0],analytic_tune=expected[mode],
                     phase_tune=result['phase_tune_absolute'][i,mode],
                     spectral_tune=result['spectral_tune_absolute'][i,mode],
                     phase_residual_rad=result['phase_residual_rad'][i,mode],
                     spectral_coherence=result['spectral_coherent_fraction'][i,mode])
            row['phase_minus_model']=row['phase_tune']-row['analytic_tune']
            rows.append(row)
        table=pd.DataFrame(rows)
        table.to_csv(study / 'tune-comparison.csv',index=False)
        print(table[table.dt_s==dt].groupby('mode')[['phase_tune','spectral_tune','phase_minus_model','spectral_coherence']].mean().to_string(),flush=True)
