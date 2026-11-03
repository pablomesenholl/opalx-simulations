"""Run frozen-length matching and coupled free/RF ellipsoid moment experiments."""
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
from scipy.constants import c, elementary_charge, proton_mass
from scipy.optimize import brentq
from kv_model import beam_parameters
from ellipsoid_model import (match_transverse, transverse_pass, axis_coefficients,
                             initial_covariance, ring_slip, evolve_covariance)
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'opalx-kv-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'config.json').read_text())
    ecfg = json.loads((root/'ellipsoid-config.json').read_text())
    beta, gamma, _, _ = beam_parameters(cfg['kinetic_energy_MeV'])
    els, cells = cfg['elements'], cfg['cells']
    ex, ey = cfg['rms_emittance_x_m_rad'], cfg['rms_emittance_y_m_rad']
    length = ecfg['longitudinal_semiaxis_m']
    circumference = cells*sum(el['length_m'] for el in els)
    cache = {}
    def match(charge):
        if charge not in cache:
            cache[charge] = match_transverse(els, ex, ey, length, charge, beta, gamma)
        return cache[charge]
    bare = match(0.)[1]*cells
    target = ecfg['target_vertical_shift']
    upper = 1e-12
    while match(upper)[1][1]*cells-bare[1] > target:
        upper *= 2
        if upper > 1e-8:
            raise RuntimeError('Charge target not bracketed')
    charge = brentq(lambda q: match(q)[1][1]*cells-bare[1]-target, 0, upper, xtol=1e-22)
    initial, tunes, error = match(charge)
    _, s, u = transverse_pass(initial, els, 5*ex, 5*ey, length, charge, beta, gamma, samples=100)
    coefficients = np.array([axis_coefficients(v[0], v[2], length, charge, beta, gamma) for v in u])
    cancel_slope = -np.trapezoid(coefficients[:, 2], s)
    rf_wave = 2*np.pi*ecfg['rf_harmonic']/circumference
    voltage = abs(cancel_slope)*beta**2*gamma*proton_mass*c*c/(elementary_charge*rf_wave)
    out = Path(tempfile.mkdtemp(prefix='ellipsoid-', dir=root/'results'))
    for name in ['config.json', 'ellipsoid-config.json']:
        shutil.copy2(root/name, out/name)
    shutil.copytree(Path(__file__).parent, out/'source', ignore=shutil.ignore_patterns('__pycache__'))
    scan = []
    for q in np.linspace(0, charge, 21):
        env, tune, residual = match(q)
        scan.append(dict(charge_pC=q*1e12, qx=cells*tune[0], qy=cells*tune[1],
                         dqx=cells*tune[0]-bare[0], dqy=cells*tune[1]-bare[1],
                         sigma_x0_mm=env[0]*1000/np.sqrt(5), sigma_y0_mm=env[2]*1000/np.sqrt(5)))
    pd.DataFrame(scan).to_csv(out/'frozen_length_scan.csv', index=False)
    pd.DataFrame(dict(s_m=s, a_m=u[:, 0], b_m=u[:, 2], ksc_x_m2=coefficients[:, 0],
                      ksc_y_m2=coefficients[:, 1], ksc_z_m2=coefficients[:, 2])).to_csv(out/'frozen_fields.csv', index=False)
    covariance = initial_covariance(initial, ex, ey, length, ecfg['rms_relative_momentum_spread'])
    summaries, frames = {}, {}
    for label, q, slope in [('no_sc_no_rf', 0., 0.), ('sc_no_rf', charge, 0.),
                             ('sc_rf_cancel', charge, cancel_slope), ('sc_rf_opposite', charge, -cancel_slope)]:
        rows, reason = evolve_covariance(covariance, els, cells, ecfg['turns'], q, beta, gamma,
                                         rf_slope=slope, samples=ecfg['samples_per_element'],
                                         maximum_axis=ecfg['maximum_spatial_semiaxis_m'])
        frame = pd.DataFrame([dict(s_m=ss, turn=ss/circumference,
                                   sigma_x_mm=np.sqrt(v[0, 0])*1000, sigma_y_mm=np.sqrt(v[2, 2])*1000,
                                   sigma_z_mm=np.sqrt(v[4, 4])*1000, sigma_delta=np.sqrt(v[5, 5]),
                                   correlation_xz=v[0, 4]/np.sqrt(v[0, 0]*v[4, 4]),
                                   det_ratio=np.linalg.det(v)/np.linalg.det(covariance)) for ss, v in rows])
        frame.to_csv(out/f'{label}.csv', index=False)
        frames[label] = frame
        summaries[label] = dict(status=reason, turns=float(frame.turn.iloc[-1]),
                                final_sigma_z_mm=float(frame.sigma_z_mm.iloc[-1]),
                                min_sigma_z_mm=float(frame.sigma_z_mm.min()), max_sigma_z_mm=float(frame.sigma_z_mm.max()),
                                max_det_relative_error=float(max(abs(frame.det_ratio-1))))
        fine_rows, fine_reason = evolve_covariance(covariance, els, cells, ecfg['turns'], q, beta, gamma,
                                         rf_slope=slope, samples=1, rtol=2e-12,
                                         maximum_axis=ecfg['maximum_spatial_semiaxis_m'])
        final_fine_sizes = np.sqrt(np.diag(fine_rows[-1][1])[[0, 2, 4]])
        final_sizes = np.sqrt(np.diag(rows[-1][1])[[0, 2, 4]])
        convergence = float(max(abs(final_fine_sizes/final_sizes-1)))
        summaries[label]['tighter_integration_max_relative_size_change'] = convergence
        if convergence > 1e-6 or fine_reason != reason:
            raise RuntimeError(f'Covariance integration convergence failed: {label}, {convergence}')
    _, fine, _ = match_transverse(els, ex, ey, length, charge, beta, gamma, rtol=2e-12)
    tune_error = float(max(abs(cells*(fine-tunes))))
    if tune_error > 1e-7:
        raise RuntimeError('Frozen tune integration convergence failed')
    summary = dict(model='Uniform ellipsoid rms closure, not exact kinetic equilibrium',
                   bunch_charge_pC=charge*1e12, protons=charge/elementary_charge,
                   full_length_m=2*length, rms_length_m=length/np.sqrt(5),
                   peak_current_mA=3*charge*beta*c/(4*length)*1000,
                   average_single_bunch_current_mA=charge*beta*c/circumference*1000,
                   bare_tunes=bare.tolist(), frozen_length_tunes=(cells*tunes).tolist(),
                   frozen_length_shifts=(cells*tunes-bare).tolist(), matched_semiaxes_m=[initial[0], initial[2], length],
                   scaled_matching_residual=error, tighter_integration_tune_difference=tune_error,
                   ring=ring_slip(els, cells, gamma), rf_frequency_Hz=ecfg['rf_harmonic']*beta*c/circumference,
                   trial_voltage_per_cell_V=voltage, number_cavities=cells,
                   rf_cancel_slope_per_cell_m1=cancel_slope,
                   maximum_rf_phase_rad=rf_wave*length,
                   rf_scope='Trial linear kicks at cell exits, both polarities; NOT an optimized cavity match',
                   evolving_cases=summaries)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    provenance = dict(revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
                      command=[sys.executable, *sys.argv], mpi_ranks=1, execution='serial Python; no OPALX run',
                      config_sha256={name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['config.json', 'ellipsoid-config.json']})
    (out/'run.log').write_text(json.dumps(provenance, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    tab = pd.DataFrame(scan)
    for axis in ['x', 'y']:
        axes[0].plot(tab.charge_pC, tab[f'dq{axis}'], label=rf'$\Delta Q_{axis}$', lw=2)
    axes[0].set(xlabel='Bunch charge [pC]', ylabel='Conditional tune shift', title='Length held fixed: 10 cm full length')
    for label, frame in frames.items():
        axes[1].plot(frame.turn, frame.sigma_z_mm, label=label.replace('_', ' '))
    axes[1].set(xlabel='Turns', ylabel='Rms bunch length [mm]', title='Coupled ring envelope evolution')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'ellipsoid.{ext}', dpi=220)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
