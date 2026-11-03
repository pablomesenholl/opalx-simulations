"""Check a completed one-particle COF/GAUSS launch against its tracked reference."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
run = parser.parse_args().run
metadata = json.loads((run/'provenance.json').read_text())
assert metadata['completed'] and metadata['particle_count'] == 1
cof = json.loads((run/'bare-cof.json').read_text())
with h5py.File(run/'case.h5') as output:
    names = sorted(output, key=lambda key: int(key.split('#')[1]))
    position = np.array([[output[n][k][0] for k in ('x','y','z')] for n in names])
    momentum = np.array([[output[n][k][0] for k in ('px','py','pz')] for n in names])
    last = output[names[-1]]
    ref_r, ref_p = np.array(last.attrs['RefPartR']), np.array(last.attrs['RefPartP'])
    end_time = float(last.attrs['TIME'][0])
launch_r, launch_p = np.array(cof['position_m']), np.array(cof['momentum_mc'])
result = dict(maximum_particle_reference_distance_m=float(np.linalg.norm(position,axis=1).max()),
              maximum_local_transverse_momentum_mc=float(np.linalg.norm(momentum[:,:2],axis=1).max()),
              reference_return_distance_m=float(np.linalg.norm(ref_r-launch_r)),
              reference_return_momentum_error_mc=float(np.linalg.norm(ref_p-launch_p)),
              final_particle_local_position_m=position[-1].tolist(),
              final_particle_local_momentum_mc=momentum[-1].tolist(), end_time_s=end_time,
              cof_residual=cof['residual'], binary_sha256=metadata['binary_sha256'])
result['checks'] = dict(finite=bool(np.isfinite(position).all() and np.isfinite(momentum).all()),
                       particle_reference_below_nm=result['maximum_particle_reference_distance_m'] < 1e-9,
                       reference_return_below_nm=result['reference_return_distance_m'] < 1e-9,
                       transverse_momentum_below_1e_10=result['maximum_local_transverse_momentum_mc'] < 1e-10)
result['passed'] = all(result['checks'].values())
(run/'orbit-agreement.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2,allow_nan=False))
if not result['passed']:
    raise RuntimeError('Orbit comparison failed')
