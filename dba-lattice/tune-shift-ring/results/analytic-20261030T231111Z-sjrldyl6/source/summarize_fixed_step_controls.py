"""Summarize retained fixed-Boris controls; no OPALX execution or data deletion."""
import argparse
import contextlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_fixed_step_observations import audit_case


def aggregate_primary_modes(table):
    """Keep per-probe extremes: a cancelling mean is not convergence evidence."""
    required = {'amplitude_m', 'dt_s', 'probe', 'mode', 'analytic_tune',
                'phase_tune_absolute', 'spectral_tune_absolute',
                'spectral_coherent_fraction', 'estimator_difference'}
    if not required.issubset(table.columns):
        raise ValueError('Missing required primary-mode columns')
    rows = []
    for (amplitude, dt, mode), group in table.groupby(['amplitude_m', 'dt_s', 'mode']):
        by_probe = group.set_index('probe')
        opposite = [abs(by_probe.loc[f'{mode}{q}+', 'phase_tune_absolute'] -
                        by_probe.loc[f'{mode}{q}-', 'phase_tune_absolute']) for q in (0, 1)]
        error = group.phase_tune_absolute - group.analytic_tune
        rows.append(dict(amplitude_mm=1000*amplitude, dt_ps=dt*1e12, mode=mode,
                         mean_phase_error=error.mean(), max_abs_phase_error=abs(error).max(),
                         max_abs_spectral_error=abs(group.spectral_tune_absolute-group.analytic_tune).max(),
                         max_estimator_disagreement=group.estimator_difference.max(),
                         min_coherence=group.spectral_coherent_fraction.min(),
                         max_opposite_probe_difference=max(opposite)))
    return pd.DataFrame(rows)


def build_report(study, source):
    """Audit geometry/map fits independently and write a reproducible summary."""
    table = pd.read_csv(study/'tune-comparison.csv')
    summary = aggregate_primary_modes(table)
    summary.to_csv(study/'primary-mode-summary.csv', index=False)
    audits = []
    for directory in sorted(study.glob('amp*mm-dt*ps')):
        with contextlib.redirect_stdout(io.StringIO()):
            result = audit_case(directory, source, directory/'independent-audit.json')
        meta = json.loads((directory/'observation-audit.json').read_text())
        rows = dict(amplitude_mm=meta['amplitude_m']*1e3, dt_ps=meta['dt_s']*1e12,
                    observation_valid=meta['observation_valid'],
                    unordered_transitions=result['unordered_section_transitions'],
                    max_central_reference_m=result['max_particle_reference_position_m'],
                    max_reference_x_m=result['max_reference_abs_section_phase'][0],
                    fitted_map_relative_rms=result.get('fitted_map', {}).get('relative_rms_prediction_error'))
        audits.append(rows)
    pd.DataFrame(audits).to_csv(study/'independent-audit-summary.csv', index=False)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = {0.001: '#0072B2', 0.005: '#D55E00'}
    for ax, mode in zip(axes.flat, 'xyz'):
        for amp, group in table[table['mode']==mode].groupby('amplitude_m'):
            ax.scatter(group.dt_s*1e12, abs(group.phase_tune_absolute-group.analytic_tune),
                       color=colors[amp], marker='o' if amp==.001 else 'x',
                       label=f'{amp*1e3:g} mm', alpha=.8)
        ax.axhline(.001, color='0.35', ls='--', lw=1, label='0.001 target')
        ax.text(390, .08, '400 ps:\nsection check failed', ha='right', va='top', fontsize=8)
        ax.set(xlabel='Boris timestep (ps)', ylabel=f'Absolute primary {mode} tune error',
               xlim=(75,425), yscale='log', ylim=(1e-5,.15))
        ax.set_xticks([100,200,400]);ax.grid(axis='y', alpha=.15)
    axes.flat[0].legend(frameon=False, fontsize=8, loc='upper left')
    ax=axes.flat[3]
    rf=json.loads((study/'rf.json').read_text());period=rf['harmonic']/rf['frequency_hz']
    for directory in sorted(study.glob('amp*mm-dt*ps')):
        meta=json.loads((directory/'observation-audit.json').read_text())
        if meta['dt_s']==4e-10:continue
        history=np.load(directory/'reference-history.npz')
        separation=np.linalg.norm(history['central_local_r'],axis=1)
        ax.plot(history['t_s']/period, np.maximum(separation,1e-16),
                color=colors[meta['amplitude_m']],ls='-' if meta['dt_s']==1e-10 else '--',lw=1,
                label=f"{meta['amplitude_m']*1e3:g} mm, {meta['dt_s']*1e12:g} ps")
    ax.set(xlabel='Elapsed turns', ylabel='Centre–tracked reference separation (m)',
           yscale='log',xlim=(0,32),ylim=(1e-15,1))
    ax.legend(frameon=False,fontsize=8,loc='lower right');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Fixed Boris with point-sampled hard edges: no validated timestep',fontsize=13)
    for extension in ('png','pdf'):
        fig.savefig(study/f'fixed-step-audit.{extension}')
    plt.close(fig)
    return summary, pd.DataFrame(audits)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path);parser.add_argument('source',type=Path)
    args=parser.parse_args()
    summary,audits=build_report(args.study,args.source)
    print(summary.to_string(index=False));print(audits.to_string(index=False))
