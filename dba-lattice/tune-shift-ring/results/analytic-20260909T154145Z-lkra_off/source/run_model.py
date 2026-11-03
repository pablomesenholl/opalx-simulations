"""Reproduce the KV benchmark; immutable results, tables and scientific figures."""
import argparse
from datetime import datetime, timezone
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
import scipy
from scipy.constants import c, elementary_charge
from scipy.optimize import brentq

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'opalx-kv-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kv_model import beam_parameters, matched, propagate, envelope_multipliers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'config.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root_dir = Path(__file__).resolve().parents[1]
    out = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('analytic-%Y%m%dT%H%M%SZ-'), dir=root_dir/'results'))
    shutil.copy2(args.config, out/'config.json')
    shutil.copytree(Path(__file__).parent, out/'source', ignore=shutil.ignore_patterns('__pycache__'))
    els, cells = config['elements'], config['cells']
    ex, ey = 4*config['rms_emittance_x_m_rad'], 4*config['rms_emittance_y_m_rad']
    beta, gamma, rigidity, ia = beam_parameters(config['kinetic_energy_MeV'])
    circumference = cells*sum(el['length_m'] for el in els)
    _, qcell0, _ = matched(els, ex, ey, 0.)
    q0 = cells*qcell0
    cache = {}
    def solution(K):
        if K not in cache:
            cache[K] = matched(els, ex, ey, K)
        return cache[K]
    target_K = []
    for shift in config['target_vertical_shifts']:
        if shift == 0:
            target_K.append(0.)
            continue
        upper = 1e-7
        while cells*solution(upper)[1][1]-q0[1] > shift:
            upper *= 2
            if upper > .001:
                raise RuntimeError('Requested tune depression not bracketed')
        target_K.append(brentq(lambda K: cells*solution(K)[1][1]-q0[1]-shift,
                              0., upper, xtol=1e-15))
    rows, profiles = [], []
    for K in sorted(set(np.linspace(0, max(target_K), 31)) | set(target_K)):
        initial, q, error = solution(K)
        end, s, y = propagate(initial, els, ex, ey, K, samples=80)
        current = K*ia*beta**3*gamma**3/2
        multiplier = max(abs(envelope_multipliers(initial, els, ex, ey, K)))
        row = dict(perveance=K, current_mA=current*1e3,
                   line_density_per_m=current/(elementary_charge*beta*c),
                   particles_in_ring=current*circumference/(elementary_charge*beta*c),
                   qx=cells*q[0], qy=cells*q[1], dqx=cells*q[0]-q0[0], dqy=cells*q[1]-q0[1],
                   sigma_x0_mm=initial[0]*500, sigma_y0_mm=initial[2]*500,
                   max_sigma_x_mm=max(y[:, 0])*500, max_sigma_y_mm=max(y[:, 2])*500,
                   max_envelope_multiplier=multiplier, scaled_closure_error=error,
                   a_m=initial[0], ap=initial[1], b_m=initial[2], bp=initial[3])
        rows.append(row)
        if K in target_K:
            profiles.append((row, pd.DataFrame(dict(s_m=s, sigma_x_mm=y[:, 0]*500,
                                                   sigma_y_mm=y[:, 2]*500, phase_x_rad=y[:, 4], phase_y_rad=y[:, 5]))))
    table = pd.DataFrame(rows)
    table.to_csv(out/'current_scan.csv', index=False)
    targets = table[table.perveance.isin(target_K)]
    targets.to_csv(out/'targets.csv', index=False)
    for n, (row, frame) in enumerate(profiles):
        frame.to_csv(out/f'envelope-{n}.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for axis in ['x', 'y']:
        axes[0].plot(table.current_mA, table[f'dq{axis}'], label=rf'$\Delta Q_{axis}$', lw=2)
        axes[0].scatter(targets.current_mA, targets[f'dq{axis}'], s=20)
    axes[0].set(xlabel='Coasting current [mA]', ylabel='Incoherent tune shift', title='Matched KV beam, 2.5 MeV protons')
    axes[0].legend()
    axes[1].plot(table.current_mA, table.max_envelope_multiplier, color='black')
    axes[1].axhline(1, color='gray', ls='--')
    axes[1].set(xlabel='Coasting current [mA]', ylabel='Largest envelope multiplier modulus', title='Cell envelope stability')
    for ax in axes:
        ax.grid(alpha=.2)
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'tune_shift.{ext}')
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    for row, frame in profiles:
        label = rf'$\Delta Q_y={row["dqy"]:.2f}$; {row["current_mA"]:.3f} mA'
        for axis, ax in zip(['x', 'y'], axes):
            ax.plot(frame.s_m, frame[f'sigma_{axis}_mm'], label=label)
            ax.set_ylabel(rf'$\sigma_{axis}$ [mm]')
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title('Periodic rms envelopes — one of six identical cells')
    axes[1].set_xlabel('Distance along cell [m]')
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'envelopes.{ext}')
    plt.close(fig)
    # A tighter integration repeats all target matches, independently of the cache.
    convergence = []
    for K in target_K:
        _, fine, _ = matched(els, ex, ey, K, rtol=2e-12)
        convergence.append(float(max(abs(cells*(fine-solution(K)[1])))))
    if max(convergence) > 1e-7:
        raise RuntimeError(f'Tune convergence failed: {convergence}')
    summary = dict(circumference_m=circumference, beta_rel=beta, gamma=gamma,
                   rigidity_T_m=rigidity, bare_tunes=q0.tolist(),
                   max_tune_change_tighter_integration=max(convergence),
                   max_scaled_closure_error=float(table.scaled_closure_error.max()),
                   max_envelope_multiplier=float(table.max_envelope_multiplier.max()),
                   scope='Coasting KV, zero momentum spread, direct transverse space charge; no OPALX run')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root_dir, text=True).strip()
    provenance = dict(revision=revision, command=sys.argv, python=sys.version,
                      numpy=np.__version__, scipy=scipy.__version__, mpi_ranks=1,
                      execution='serial Python, MPI not invoked',
                      config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    (out/'run.log').write_text(json.dumps(provenance, indent=2)+'\n'+json.dumps(summary, indent=2)+'\n')
    print(targets[['current_mA', 'qx', 'qy', 'dqx', 'dqy', 'max_envelope_multiplier']].to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
