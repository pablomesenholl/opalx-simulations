"""Profile one-rank OPEN PIC in a field-free drift without boundary subdivision.

Uses the already verified cold ellipsoid particles; wall times include startup
and final output. Internal OPALX timers distinguish repeated solve/kernel cost.
This is a timing experiment, not a matched-ring or tune-shift result.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import h5py
import numpy as np
import pandas as pd

from large_ellipsoid_benchmark import ROOT, drift_deck, sha256, write_json


if __name__ == '__main__':
    source = ROOT / 'results/large-ellipsoid-rjg8zxr_'
    binary = ROOT.parents[2] / 'omp-build/src/opalx'
    metadata = json.loads((source / 'study.json').read_text())
    study = Path(tempfile.mkdtemp(prefix='pic-thread-profile-', dir=ROOT / 'results'))
    shutil.copy2(__file__, study / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('large_ellipsoid_benchmark.py'), study)
    write_json(study / 'study.json', dict(binary=str(binary), binary_sha256=sha256(binary),
               mpi_ranks=1, count=131072, mesh=32, steps=100, dt_s=6.25e-11,
               charge_C=metadata['charge_C'], source=str(source),
               scope='OpenMP timing only, field-free drift; no ring matching claim'))
    print(study, flush=True)
    rows, control = [], None
    for threads in [1, 2, 4, 8]:
        directory = study / f'omp{threads}'
        directory.mkdir()
        (directory / 'particles.txt').symlink_to(source / 'n131072-m32/particles.txt')
        deck = drift_deck(131072, 32, metadata['charge_C'], dt=6.25e-11, steps=100)
        deck = deck.replace('PSDUMPFREQ=1, STATDUMPFREQ=1', 'PSDUMPFREQ=100, STATDUMPFREQ=100')
        (directory / 'case.in').write_text(deck)
        command = [str(binary), '--info', '1', 'case.in']
        record = dict(command=command, omp_threads=threads, mpi_ranks=1,
                      input_sha256=sha256(directory / 'case.in'),
                      particle_sha256=sha256(directory / 'particles.txt'), status='running')
        write_json(directory / 'provenance.json', record)
        start = time.monotonic()
        with (directory / 'run.log').open('w') as stream:
            try:
                result = subprocess.run(command, cwd=directory, timeout=180,
                          env={**os.environ, 'OMP_NUM_THREADS':str(threads), 'OMP_PROC_BIND':'false'},
                          stdout=stream, stderr=subprocess.STDOUT)
                record.update(returncode=result.returncode,
                              status='completed' if result.returncode == 0 else 'failed')
            except subprocess.TimeoutExpired:
                record.update(status='timed_out')
        record['elapsed_seconds'] = time.monotonic() - start
        write_json(directory / 'provenance.json', record)
        print(f'{directory.name}: {record}', flush=True)
        if record['status'] != 'completed':
            raise RuntimeError(f'{directory}: {record["status"]}')
        with h5py.File(directory / 'case.h5') as output:
            group = output[max(output, key=lambda name:int(name.split('#')[-1]))]
            ordering = np.argsort(group['id'][:])
            if not np.array_equal(group['id'][:][ordering], np.arange(131072)):
                raise RuntimeError('Particle loss or unexpected particle IDs')
            values = np.column_stack([group[name][:][ordering]
                                      for name in ['x','y','z','px','py','pz']])
            if not np.isfinite(values).all():
                raise RuntimeError('Nonfinite final particle state')
            final_time = float(np.asarray(group.attrs['TIME']).ravel()[0])
            if not np.isclose(final_time, 100*6.25e-11, rtol=1e-12, atol=0):
                raise RuntimeError('Incomplete timing run')
        if control is None:
            control = values
        rows.append(dict(omp_threads=threads, wall_seconds=record['elapsed_seconds'],
                    max_delta_R_m=float(np.max(abs(values[:,:3]-control[:,:3]))),
                    max_delta_P_mc=float(np.max(abs(values[:,3:]-control[:,3:])))))
        pd.DataFrame(rows).to_csv(study / 'thread-comparison.csv', index=False)
