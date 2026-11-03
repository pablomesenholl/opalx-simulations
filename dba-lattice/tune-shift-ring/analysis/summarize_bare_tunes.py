"""Plot and validate two completed bare-tune runs, ordered coarse then fine."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('runs', nargs=2, type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
records = []
for run in args.runs:
    result = json.loads((run/'tunes.json').read_text())
    for item in result['measurements']:
        records.append(dict(run=str(run.resolve()), dt_s=result['dt_s'], internal_dt_s=result['internal_dt_s'], **item))
frame = pd.DataFrame(records)
frame.to_csv(args.output/'tune_comparison.csv', index=False)
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 220, 'font.family': 'DejaVu Sans'})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
extrapolations = []
for axis, ax in zip(('x','y'), axes):
    subset = frame[frame.axis == axis]
    for particle, marker in ((1, 'o'), (2, '^'), (3, 's'), (4, 'v')):
        label = f"Device TRACK: {subset[subset.particle_id == particle].initial_position_m.iloc[0]*1e6:+g} µm"
        data = subset[subset.particle_id == particle].sort_values('internal_dt_s')
        x = data.internal_dt_s.to_numpy()**2
        q = data.tracking_tune.to_numpy()
        q0 = float((q[0]*x[1]-q[1]*x[0])/(x[1]-x[0]))
        exact = float(data.analytic_tune.iloc[0])
        extrapolations.append(dict(axis=axis, particle_id=particle, extrapolated_tune=q0,
                                  analytic_tune=exact, difference=q0-exact))
        ax.plot(x*1e18, (q-exact)*1e6, marker=marker, lw=1.4, label=label)
    data = subset[subset.particle_id == 1].sort_values('internal_dt_s')
    ax.plot(data.internal_dt_s**2*1e18, (data.cof_tune-data.analytic_tune)*1e6,
            marker='x', color='black', ls='--', lw=1., label='Host COF map')
    ax.axhline(0, color='0.5', lw=1.)
    ax.set(title=f'Horizontal tune Qx' if axis == 'x' else 'Vertical tune Qy',
           xlabel='Nominal Boris timestep squared [ns²]', ylabel='Tune − analytic tune [10⁻⁶]')
    ax.set_xlim(left=0)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=9)
fig.suptitle('Bare DBA ring · 2.5 MeV protons · 32 turns · one MPI rank')
for suffix in ('png', 'pdf'):
    fig.savefig(args.output/f'bare-tunes.{suffix}')
checks = dict(
    interpolation_orders_agree=bool(frame.interpolation_order_difference.abs().max() < 1e-6),
    tracking_matches_cof=bool((frame.tracking_minus_cof.abs() < 2e-6).all()),
    stable_half_record_tunes=bool(max(abs(v[0]-v[1]) for v in frame.half_tunes) < 1e-6),
    small_energy_variation=bool(frame.relative_energy_variation.max() < 1e-8),
    independent_frequency_agrees=bool(frame.frequency_estimator_difference.abs().max() < 1e-6),
    extrapolation_matches_analytic=bool(max(abs(v['difference']) for v in extrapolations) < 2e-6))
summary = dict(checks=checks, extrapolations=extrapolations,
               note='Tracking full tunes come from phase unwrapping at each particle’s own six cell crossings per turn, without analytic branch assignment. COF integer branches use cell optics. Two-step extrapolation assumes leading dt² error.')
(args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(frame[['dt_s','axis','amplitude_m','analytic_tune','cof_tune','tracking_tune','tracking_minus_cof']].to_string(index=False))
print(json.dumps(summary, indent=2))
if not all(checks.values()):
    raise RuntimeError('Bare tune validation failed; inspect numerical results')
