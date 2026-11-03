"""One-rank regression: central particle independent of four spectators (FS=NONE).

Pass a completed check_pr553_tracking.py output directory. Retains inputs, logs,
HDF5, executable hash, source diff, and a numerical assertion summary.
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

import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('baseline', type=Path)
args = parser.parse_args()
base = args.baseline.resolve() / 'cof_initialorbit'
repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
binary = repo / 'omp-build/src/opalx'
out = Path(tempfile.mkdtemp(prefix='resolved-spectators-', dir=base.parent.parent))
shutil.copy2(__file__, out / Path(__file__).name)
(out / 'source.diff').write_text(subprocess.check_output(['git', 'diff'], text=True, cwd=repo))
orbit = json.loads((base / 'bare-cof.json').read_text())
assert orbit['converged'] and np.max(np.abs(orbit['coordinates'])) < 1e-12
momentum = np.sqrt((1 + orbit['energy_MeV'] * 1e6 / orbit['mass_eV']) ** 2 - 1)
template = (base / 'case.in').read_text()
template = re.sub(r'ic: COF,.*?;', '', template, flags=re.S)
template = template.replace(', INITIALORBIT=ic', '').replace(', PC=P0', '')
template = re.sub(r'DIST: DISTRIBUTION,.*?;',
                  'DIST: DISTRIBUTION, TYPE=FROMFILE, FNAME="particles.txt", NPARTDIST=COUNT;', template)
tables = {}
print(out, flush=True)
for count in (1, 5):
    case = out / f'n{count}'
    case.mkdir()
    (case / 'case.in').write_text(template.replace('COUNT', str(count)).replace('NALLOC=1', f'NALLOC={count}'))
    offsets = [(0., 0.), (1e-4, 0.), (-1e-4, 0.), (0., 1e-4), (0., -1e-4)][:count]
    (case / 'particles.txt').write_text(f'{count}\nx px y py z pz\n' + ''.join(
        f'{x:.17g} 0 0 0 {z:.17g} {momentum:.17g}\n' for x, z in offsets))
    with (case / 'run.log').open('w') as log:
        subprocess.run([str(binary), '--info', '2', 'case.in'], cwd=case,
                       env={**os.environ, 'OMP_NUM_THREADS': '1', 'OMP_PROC_BIND': 'false'},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = []
    with h5py.File(case / 'case.h5') as data:
        for name in sorted(data, key=lambda k: int(k.split('#')[1])):
            step = data[name]
            assert len(step['id']) == count, 'Unexpected loss'
            for i in range(count):
                rows.append(dict(time_s=step.attrs['TIME'][0], id=int(step['id'][i]),
                                 **{axis: step[axis][i] for axis in ('x', 'y', 'z', 'px', 'py', 'pz')}))
    tables[count] = pd.DataFrame(rows)
    tables[count].to_csv(case / 'particles.csv', index=False)
central = {n: df[df.id == df.id.min()].reset_index(drop=True) for n, df in tables.items()}
np.testing.assert_allclose(central[1].time_s, central[5].time_s, rtol=0, atol=1e-18)
delta = central[1][['x', 'y', 'z']].to_numpy() - central[5][['x', 'y', 'z']].to_numpy()
maximum_difference = float(np.max(np.linalg.norm(delta, axis=1)))
maximum_offset = {n: float(np.max(np.linalg.norm(df[['x', 'y', 'z']], axis=1))) for n, df in central.items()}
spectator_offset = float(tables[5][tables[5].id != tables[5].id.min()].x.abs().max())
# 1 nm is > rounding accumulated through ~12000 frame changes, yet 10^6 below
# the original millimetre error. Finite offsets must remain real trajectories.
assert maximum_difference < 1e-9, maximum_difference
assert max(maximum_offset.values()) < 1e-9, maximum_offset
assert spectator_offset > 1e-5, 'Spectators must not be projected onto the closed orbit'
summary = dict(baseline=str(base), mpi_ranks=1, omp_threads=1,
               binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
               central_difference_m=maximum_difference, central_maximum_offset_m=maximum_offset,
               spectator_maximum_x_m=spectator_offset, passed=True)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
