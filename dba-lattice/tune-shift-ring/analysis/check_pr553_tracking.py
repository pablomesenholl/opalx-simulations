"""Recheck the proton baseline on PR553, with legacy and named-COF launches.

One rank/one thread. Fresh output directories retain exact input and binary hash.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import h5py
import numpy as np
import pandas as pd
from kv_model import beam_parameters

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dt', type=float, default=1e-10)
parser.add_argument('--cof-integrator', choices=['DOP853', 'BORIS'], default='DOP853')
parser.add_argument('--launch', choices=['both', 'cof_initialorbit'], default='both')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=root, text=True).strip())
binary = repo/'omp-build/src/opalx'
out = Path(tempfile.mkdtemp(prefix='pr553-tracking-', dir=root/'results'))
shutil.copy2(__file__, out/'check_pr553_tracking.py')
revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
(out/'provenance.json').write_text(json.dumps(dict(revision=revision, dt_s=args.dt,
    cof_integrator=args.cof_integrator, cof_dt_s=args.dt, mpi_ranks=1,
    omp_threads=1, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()), indent=2)+'\n')
bare = (root/'bare-cof.in').read_text()
lattice = bare.split('BEAM1: BEAM')[0].replace('PSDUMPFREQ=0, STATDUMPFREQ=0', 'PSDUMPFREQ=100, STATDUMPFREQ=100').replace('ENABLELINEARTRANSFERMAPS=TRUE', 'ENABLELINEARTRANSFERMAPS=FALSE')
cof = bare[bare.index('ic: COF'):bare.index('QUIT;')]
cof = re.sub(r'\bDT\s*=\s*[^,;]+', f'DT={args.dt:.17g}', cof)
cof = re.sub(r'TIMEINTEGRATOR\s*=\s*"[^"]+"', f'TIMEINTEGRATOR="{args.cof_integrator}"', cof)
cof = re.sub(r'MAXSTEPS\s*=\s*\d+', 'MAXSTEPS=200000', cof)
beta, gamma, _, _ = beam_parameters(2.5)
summary = []
print(out, flush=True)
for mode in (['legacy_fromfile', 'cof_initialorbit'] if args.launch == 'both' else [args.launch]):
    case = out/mode
    case.mkdir()
    source = ('DIST: DISTRIBUTION, TYPE=FROMFILE, FNAME="reference.txt", NPARTDIST=1;\n'
              if mode == 'legacy_fromfile' else
              'DIST: DISTRIBUTION, TYPE=GAUSS, SIGMAX=1e-12, SIGMAY=1e-12, SIGMAZ=1e-12, SIGMAPX=0, SIGMAPY=0, SIGMAPZ=0, NPARTDIST=1;\n')
    source += 'SOURCE: EMISSIONSOURCE, DISTRIBUTION=DIST;\nSOURCES: EMISSIONSOURCELIST=(SOURCE);\n'
    source += 'FS: FIELDSOLVER, TYPE=NONE, NX=16, NY=16, NZ=16, PARFFTX=TRUE, PARFFTY=TRUE, PARFFTZ=TRUE, BCFFTX=OPEN, BCFFTY=OPEN, BCFFTZ=OPEN;\n'
    source += 'BEAM1: BEAM, PARTICLE=PROTON, CHARGE=1, NALLOC=1, SOURCES=SOURCES, BCHARGE=1.602176634e-19'
    source += ', PC=P0;\n'+cof if mode == 'cof_initialorbit' else ';\n'
    source += f'TRACK, LINE=LATTICE, BEAM=BEAM1, DT={args.dt:.17g}, MAXSTEPS=200000'
    source += ', INITIALORBIT=ic;\n' if mode == 'cof_initialorbit' else ';\n'
    source += 'RUN, METHOD="PARALLEL", FIELDSOLVER=FS, TURNS=1;\nENDTRACK;\nQUIT;\n'
    (case/'case.in').write_text(lattice+source)
    (case/'reference.txt').write_text(f'1\nx px y py z pz\n0 0 0 0 0 {beta*gamma:.17g}\n')
    with (case/'run.log').open('w') as log:
        subprocess.run([str(binary), '--info', '2', 'case.in'], cwd=case, stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, 'OMP_NUM_THREADS':'1', 'OMP_PROC_BIND':'false'}, check=True)
    rows = []
    with h5py.File(case/'case.h5') as f:
        for key in sorted(f, key=lambda k:int(k.split('#')[1])):
            step = f[key]
            if len(step['x']) != 1:
                raise RuntimeError('Expected one surviving particle')
            rows.append(dict(s_m=float(step.attrs['SPOS'][0]),time_s=float(step.attrs['TIME'][0]),
                             x_m=float(step['x'][0]), y_m=float(step['y'][0]), z_m=float(step['z'][0]),
                             reference_x_m=float(step.attrs['RefPartR'][0]), reference_z_m=float(step.attrs['RefPartR'][2])))
    frame = pd.DataFrame(rows)
    frame.to_csv(case/'trajectory.csv', index=False)
    result = dict(mode=mode, dt_s=args.dt, snapshots=len(frame),
                  max_abs_x_m=float(frame.x_m.abs().max()),
                  final_local_position_m=frame[['x_m','y_m','z_m']].iloc[-1].tolist(),
                  final_reference_xz_m=frame[['reference_x_m','reference_z_m']].iloc[-1].tolist(),
                  final_s_m=float(frame.s_m.iloc[-1]))
    if mode == 'cof_initialorbit':
        orbit = json.loads((case/'bare-cof.json').read_text())
        result.update(cof_integrator=args.cof_integrator, cof_dt_s=args.dt,
                      cof_coordinates=orbit['coordinates'], cof_residual=orbit['residual'],
                      cof_converged=orbit['converged'], cof_map_stability=orbit['stability'])
    summary.append(result)
    print(json.dumps(result), flush=True)
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
