"""Run a reproducible bare-tune experiment through actual device particle tracking.

One MPI rank / one OpenMP thread. Output directory is printed immediately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from scipy.constants import c

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dt', type=float, required=True)
parser.add_argument('--turns', type=int, default=128)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True, cwd=root).strip())
binary = repo / 'omp-build/src/opalx'
out = Path(tempfile.mkdtemp(prefix=f'bare-tunes-{args.dt*1e12:g}ps-', dir=root/'results'))
print(out, flush=True)
shutil.copy2(__file__, out/'run_bare_tunes.py')
shutil.copy2(root/'config.json', out/'config.json')
(out/'source.diff').write_text(subprocess.check_output(['git', 'diff'], text=True, cwd=repo))
for name in ('src/Algorithms/DeviceExternalField.h', 'src/Algorithms/DeviceExternalField.cpp'):
    shutil.copy2(repo/name, out/Path(name).name)
env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OMP_PROC_BIND': 'false'}
bare = (root/'bare-cof.in').read_text()
bare += '\n'  # Keep input terminators separate from later generated commands.
bare = 'OPTION, REPARTFREQ=0, STEPINFOFQ=10000;\n' + bare
bare = re.sub(r'\bDT\s*=\s*[^,;]+', f'DT={args.dt:.17g}', bare)
bare = re.sub(r'(?<![A-Z])TIMEINTEGRATOR="[^"]+"', 'TIMEINTEGRATOR="BORIS"', bare)
bare = bare.replace('ENABLELINEARTRANSFERMAPS=TRUE', 'ENABLELINEARTRANSFERMAPS=FALSE')
(out/'cof.in').write_text(bare)
started = time.time()
with (out/'cof.log').open('w') as log:
    subprocess.run([str(binary), '--info', '1', 'cof.in'], cwd=out, env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
cof = json.loads((out/'bare-cof.json').read_text())
assert cof['converged'] and max(abs(v) for v in cof['coordinates']) < 1e-12
bg = ((1 + cof['energy_MeV']*1e6/cof['mass_eV'])**2 - 1)**0.5
config = json.loads((root/'config.json').read_text())
circumference = config['cells']*sum(el['length_m'] for el in config['elements'])
# COF time_s is the launch-state time, not the revolution period.
period = circumference / (c*bg/(1+bg*bg)**0.5)
dump = max(1, round(period / args.dt / 96))
lattice = bare.split('BEAM1: BEAM')[0]
lattice = lattice.replace('PSDUMPFREQ=0, STATDUMPFREQ=0', f'PSDUMPFREQ={dump}, STATDUMPFREQ={dump}')
source = f'''
DIST: DISTRIBUTION, TYPE=FROMFILE, FNAME="particles.txt", NPARTDIST=5;
SOURCE: EMISSIONSOURCE, DISTRIBUTION=DIST;
SOURCES: EMISSIONSOURCELIST=(SOURCE);
FS: FIELDSOLVER, TYPE=NONE, NX=16, NY=16, NZ=16, PARFFTX=TRUE, PARFFTY=TRUE, PARFFTZ=TRUE, BCFFTX=OPEN, BCFFTY=OPEN, BCFFTZ=OPEN;
BEAM1: BEAM, PARTICLE=PROTON, CHARGE=1, NALLOC=5, SOURCES=SOURCES, BCHARGE=8.01088317e-19;
TRACK, LINE=LATTICE, BEAM=BEAM1, DT={args.dt:.17g}, MAXSTEPS=200000;
RUN, METHOD="PARALLEL", FIELDSOLVER=FS, TURNS={args.turns};
ENDTRACK;
QUIT;
'''
(out/'case.in').write_text(lattice + source)
(out/'particles.txt').write_text('5\nx px y py z pz\n' + ''.join(
    f'{amplitude:.17g} 0 {amplitude:.17g} 0 0 {bg:.17g}\n' for amplitude in (0., 1e-5, -1e-5, 5e-5, -5e-5)))
provenance = dict(dt_s=args.dt, requested_turns=args.turns, amplitudes_m=[0., 1e-5, -1e-5, 5e-5, -5e-5],
                  nominal_period_s=period, dump_steps=dump, mpi_ranks=1, omp_threads=1,
                  backend='Kokkos OpenMP', binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True, cwd=repo).strip())
(out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
print('COF complete; starting device tracking', flush=True)
with (out/'run.log').open('w') as log:
    subprocess.run([str(binary), '--info', '1', 'case.in'], cwd=out, env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
if 'Ring boundary control: synchronized ordinary Boris/PIC substeps.' not in (out/'run.log').read_text():
    raise RuntimeError('The intended boundary controller was not active; inspect input and log')
provenance['elapsed_seconds'] = time.time()-started
provenance['completed'] = True
(out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
print(json.dumps(provenance), flush=True)
