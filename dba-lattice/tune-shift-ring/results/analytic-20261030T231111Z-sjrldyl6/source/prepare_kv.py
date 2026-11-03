"""Create a matched observation section and longitudinally stratified coasting ring."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from scipy.constants import c, elementary_charge
from scipy.optimize import brentq
from kv_model import beam_parameters, matched
from kv_distribution import quiet_sphere, transverse_coordinates, section_fromfile, envelopes_at, shell_invariant, moments

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'opalx-kv-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    (root/"results").mkdir(exist_ok=True)
    cfg = json.loads((root/'config.json').read_text())
    ex, ey = 4*cfg['rms_emittance_x_m_rad'], 4*cfg['rms_emittance_y_m_rad']
    els, cells = cfg['elements'], cfg['cells']
    beta, gamma, rigidity, ia = beam_parameters(cfg['kinetic_energy_MeV'])
    _, zero, _ = matched(els, ex, ey, 0.)
    K = brentq(lambda k: cells*(matched(els, ex, ey, k)[1][1]-zero[1])+.2, 0., 3e-6, xtol=1e-16)
    initial, tunes, error = matched(els, ex, ey, K)
    current = K*ia*beta**3*gamma**3/2
    circumference = cells*sum(el['length_m'] for el in els)
    count, seed = 65536, 546544
    unit = quiet_sphere(count, seed)
    section = transverse_coordinates(unit, initial, ex, ey)
    particle_file = section_fromfile(section, beta*gamma)
    # 1024 longitudinal slices, each with exact transverse second moments.
    s = np.repeat((np.arange(count//64)+.5)*circumference/(count//64), 64)
    envelopes = envelopes_at(s, initial, els, ex, ey, K)
    ring = transverse_coordinates(unit, envelopes, ex, ey)
    out = Path(tempfile.mkdtemp(prefix='matched-kv-', dir=root/'results'))
    shutil.copy2(root/'config.json', out/'config.json')
    shutil.copytree(Path(__file__).parent, out/'source', ignore=shutil.ignore_patterns('__pycache__'))
    np.savetxt(out/'kv-section.txt', particle_file, fmt='%.17e', header=f'{count}\nx px y py z pz', comments='')
    pd.DataFrame(dict(s_m=s, x_m=ring[:, 0], xp=ring[:, 1], y_m=ring[:, 2], yp=ring[:, 3],
                      macro_charge_C=np.full(count, current*circumference/(beta*c*count)))).to_csv(out/'kv-ring.csv', index=False)
    report = dict(seed=seed, macroparticles=count, longitudinal_slices=count//64,
                  distribution='KV S3 shell, signed/cyclic quiet sampling; not waterbag',
                  current_A=current, perveance=K, circumference_m=circumference,
                  number_line_density_per_m=current/(elementary_charge*beta*c),
                  ring_charge_C=current*circumference/(beta*c),
                  protons_in_ring=current*circumference/(elementary_charge*beta*c),
                  initial_edge_envelope=initial.tolist(), bare_tunes=(cells*zero).tolist(),
                  matched_tunes=(cells*tunes).tolist(), section_moments=moments(section),
                  max_section_shell_error=float(max(abs(shell_invariant(section, initial, ex, ey)-1))),
                  max_ring_shell_error=float(max(abs(shell_invariant(ring, envelopes, ex, ey)-1))),
                  max_normalized_momentum_error=float(max(abs(np.linalg.norm(particle_file[:, [1, 3, 5]], axis=1)-beta*gamma))),
                  section_file_scope='Transverse observation plane at z=0; NOT a charged 3D bunch',
                  ring_file_scope='Local curvilinear coasting coordinates; NOT a Cartesian OPALX FROMFILE input',
                  longitudinal_assumption='Periodic uniform coasting density; finite stratified quadrature in s',
                  opalx_limitation='Current field solver rejects mixed open/open/periodic boundaries')
    report['sha256'] = {name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['kv-section.txt', 'kv-ring.csv']}
    report['provenance'] = dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                                command=[sys.executable, *sys.argv], mpi_ranks=1,
                                execution='serial Python; no OPALX/PIC run',
                                config_sha256=hashlib.sha256((root/'config.json').read_bytes()).hexdigest())
    (out/'distribution.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'run.log').write_text(json.dumps(report['provenance'], indent=2)+'\n')
    # Stable root entry points: reproducible generated files, archived originals above.
    shutil.copy2(out/'kv-section.txt', root/'kv-section.txt')
    shutil.copy2(out/'distribution.json', root/'matched-kv.json')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for ax, (i, j), labels in zip(axes, [(0, 2), (0, 1), (2, 3)],
                                  [('x [mm]', 'y [mm]'), ('x [mm]', "x′ [mrad]"), ('y [mm]', "y′ [mrad]")]):
        ax.hist2d(section[:, i]*1000, section[:, j]*1000, bins=60, cmap='viridis', cmin=1)
        ax.set(xlabel=labels[0], ylabel=labels[1])
    fig.suptitle('Matched KV observation section · 2.5 MeV protons · 10.399 mA')
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'kv-section.{ext}', dpi=220)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
