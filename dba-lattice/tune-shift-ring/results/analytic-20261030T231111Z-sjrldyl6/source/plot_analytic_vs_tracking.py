"""Plot the ellipsoid prediction; overlay measured OPALX tunes when provided.

Optional tracking CSV columns: charge_pC, qx, qy. Must include a measured
zero-charge baseline. Do not provide COF eigenphases as tracking measurements.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'opalx-kv-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--analytic', type=Path, required=True)
parser.add_argument('--tracking', type=Path)
args = parser.parse_args()
analytic = pd.read_csv(args.analytic/'frozen_length_scan.csv')
tracking = None
if args.tracking:
    tracking = pd.read_csv(args.tracking)
    if not {'charge_pC', 'qx', 'qy'}.issubset(tracking.columns):
        raise ValueError('Tracking CSV requires charge_pC,qx,qy')
    if not np.isfinite(tracking[['charge_pC', 'qx', 'qy']].to_numpy()).all():
        raise ValueError('Nonfinite tracking measurements')
    baseline = tracking[tracking.charge_pC == 0]
    if len(baseline) != 1:
        raise ValueError('Exactly one measured zero-charge tracking baseline required')
    for axis in ['x', 'y']:
        tracking[f'dq{axis}'] = tracking[f'q{axis}']-baseline[f'q{axis}'].iloc[0]

out = Path(tempfile.mkdtemp(prefix='analytic-vs-tracking-', dir=args.analytic.parent))
analytic.to_csv(out/'analytic.csv', index=False)
if tracking is not None:
    tracking.to_csv(out/'tracking.csv', index=False)
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharex=True, sharey=True)
for ax, axis, color, label in zip(axes, ['x', 'y'], ['#1764a0', '#bd5200'], ['Horizontal', 'Vertical']):
    ax.plot(analytic.charge_pC, analytic[f'dq{axis}'], color=color, lw=2.2, label='Analytic: fixed-length ellipsoid')
    ax.scatter(analytic.charge_pC.iloc[-1], analytic[f'dq{axis}'].iloc[-1], color=color, s=28)
    if tracking is not None:
        ax.scatter(tracking.charge_pC, tracking[f'dq{axis}'], facecolors='none', edgecolors='black',
                   marker='o', s=35, label='OPALX tracking')
    else:
        ax.text(.05, .12, 'OPALX tracking: pending\nNo validated tune measurements', transform=ax.transAxes,
                fontsize=10, bbox=dict(facecolor='#fff3dd', edgecolor='#b98633', boxstyle='round,pad=.5'))
    ax.set(title=label, xlabel='Bunch charge [pC]')
    ax.grid(alpha=.2)
    ax.legend(loc='upper right', fontsize=8)
axes[0].set_ylabel(r'Tune shift $\Delta Q = Q(Q_b)-Q(0)$')
fig.suptitle('Uniform ellipsoid · 2.5 MeV protons · full length held at 10 cm', fontsize=12)
fig.subplots_adjust(bottom=.23, top=.82, wspace=.12)
fig.text(.5, .05, 'Prediction at 41.13 pC: ΔQx = −0.2809, ΔQy = −0.2000\n'
         'A stationary-length assumption is required; these are not yet OPALX-validated shifts.',
         ha='center', fontsize=10)
for ext in ['png', 'pdf']:
    fig.savefig(out/f'analytic_vs_tracking.{ext}', dpi=220)
plt.close(fig)
(out/'provenance.json').write_text(json.dumps(dict(analytic_source=str(args.analytic.resolve()),
    tracking_source=None if args.tracking is None else str(args.tracking.resolve()),
    status='Tracking unavailable' if tracking is None else 'Tracking overlay supplied'), indent=2)+'\n')
print(out)
