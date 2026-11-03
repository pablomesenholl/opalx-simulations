"""Plot recorded one-rank DBA agreement runs; no simulation is launched."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runs', nargs=2, type=Path, required=True, help='Resolved 100 ps and 25 ps run folders')
parser.add_argument('--before', nargs=2, type=Path, required=True, help='Original 100 ps and 25 ps run folders')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 220})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
rows = []
for dt, color, before, after in zip((100, 25), ('#0072B2', '#D55E00'), args.before, args.runs):
    for ax, run, factor in zip(axes, (before, after), (1e3, 1e9)):
        frame = pd.read_csv(run / 'cof_initialorbit/trajectory.csv')
        error = np.linalg.norm(frame[['x_m', 'y_m', 'z_m']], axis=1)
        ax.plot(frame.s_m, factor * error, lw=1.6, color=color, label=f'{dt} ps')
        rows.append(dict(dt_ps=dt, run=str(run.resolve()), maximum_offset_m=float(error.max()),
                         final_offset_m=float(error[-1])))
    summary = json.loads((after / 'summary.json').read_text())[0]
    assert np.linalg.norm(summary['final_reference_xz_m']) < 1e-9
    assert error.max() < 1e-9
axes[0].set(title='Previous TRACK transport', ylabel='Particle–reference distance [mm]')
axes[1].set(title='Spatial, boundary-resolved Boris', ylabel='Particle–reference distance [nm]')
for ax in axes:
    ax.set_xlabel('Distance around ring [m]')
    ax.set_xlim(0, 25.77)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
fig.suptitle('DBA ring · 2.5 MeV proton · one particle · no space charge')
for suffix in ('png', 'pdf'):
    fig.savefig(args.output / f'track-cof-agreement.{suffix}')
(args.output / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))
