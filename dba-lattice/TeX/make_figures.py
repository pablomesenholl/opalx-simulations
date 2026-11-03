"""Reproduce publication figures from the baseline machine and saved KV scan.

Run with ~/.venv-h6/bin/python TeX/make_figures.py from the project root.
No new beam calculation or OPALX tracking is performed. Input hashes and
normalization/geometry checks are recorded alongside the vector figures.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib-figures'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.constants import c

sys.path.insert(0, str(ROOT/'model'))
from machine import advance_pose

OUTPUT = Path(__file__).resolve().parent/'figures'
# Plot-label classification only, not a solver-acceptance tolerance. Stable
# samples differ from unity by <1.5e-12; unstable growth begins above 5e-3.
ENVELOPE_UNITY_ATOL = 1e-8


def style():
    """Set readable, embedded-font vector output at document figure size."""
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10,
        'mathtext.fontset': 'dejavuserif', 'axes.labelsize': 11,
        'axes.titlesize': 11, 'axes.linewidth': .7, 'lines.linewidth': 1.8,
        'legend.fontsize': 9, 'legend.frameon': False,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
    })


def save(fig, name):
    """Save a vector master and a matching high-resolution preview."""
    for extension in ['pdf', 'png']:
        fig.savefig(OUTPUT/f'{name}.{extension}', bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


def ring_figure(machine):
    """Trace each element from its stored lab pose; verify endpoint closure."""
    colors = {'sector_bend': '#D99B18', 'drift': '#5B6770', 'quadrupole': '#1C78A5'}
    widths = {'sector_bend': 7.5, 'drift': 2.4, 'quadrupole': 11.5}
    fig, ax = plt.subplots(figsize=(6.5, 4.6), layout='constrained')
    first_poses, quad_midpoints = [], []
    for el in machine['elements']:
        start = np.array(el['entry_X_Z_theta'])
        points = np.array([advance_pose(start, distance, el['h_per_m'])
                           for distance in np.linspace(0, el['length_m'], 65)])
        np.testing.assert_allclose(points[-1], el['exit_X_Z_theta'], atol=2e-14, rtol=0)
        ax.plot(points[:, 0], points[:, 1], color=colors[el['type']],
                lw=widths[el['type']], solid_capstyle='butt', zorder=3)
        if el['type'] == 'sector_bend':
            # Small transverse strokes distinguish adjacent sector magnets.
            for pose in [points[0], points[-1]]:
                normal = np.array([np.cos(pose[2]), -np.sin(pose[2])])
                tick = pose[:2]+np.array([-.085, .085])[:, None]*normal
                ax.plot(tick[:, 0], tick[:, 1], color='#73500C', lw=.7, zorder=4)
        if el['name'].endswith('_B1'):
            first_poses.append(start)
        if el['type'] == 'quadrupole':
            quad_midpoints.append(points[len(points)//2, :2])
    center = np.mean([p[:2] for p in first_poses], axis=0)
    for index, midpoint in enumerate(quad_midpoints, 1):
        label = center+.76*(midpoint-center)
        ax.text(*label, f'C{index}', ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    ax.text(*center, '6 identical cells\n'+r'$B1-D1-Q-D2-B2$'+'\n'
            +rf'$C={machine["circumference_m"]:.5f}\,\mathrm{{m}}$',
            ha='center', va='center', linespacing=1.7, fontsize=10)
    ax.scatter([0], [0], s=18, color='black', zorder=5)
    ax.annotate(r'$s=0$', xy=(0, 0), xytext=(.48, -.43), fontsize=10,
                arrowprops=dict(arrowstyle='-', color='black', lw=.7))
    # Initial tangent is +Z, with the reference bending towards negative X.
    ax.add_patch(FancyArrowPatch((.46, .22), (.46, 1.02), arrowstyle='-|>',
                                mutation_scale=13, color='black', lw=1.0))
    ax.text(.68, .62, 'Beam', ha='left', va='center', fontsize=9)
    legend = [Line2D([], [], color=colors[k], lw=widths[k]*.7, solid_capstyle='butt', label=t)
              for k, t in [('sector_bend', 'Sector bends (12)'),
                           ('quadrupole', 'Quadrupoles (6)'), ('drift', 'Drifts (12)')]]
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, 1.15), ncol=3,
              handlelength=1.6, columnspacing=1.6)
    ax.set(aspect='equal', xlabel=r'Laboratory $X$ [m]', ylabel=r'Laboratory $Z$ [m]',
           xlim=(-9, 1.65), ylim=(-4.55, 4.55))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#D9DEE2', linewidth=.5, alpha=.6)
    save(fig, 'ring-lattice')


def tune_figure(scan, targets):
    """Plot full incoherent tune shifts against total coasting ring charge."""
    unstable = scan['max_envelope_multiplier'] > 1+ENVELOPE_UNITY_ATOL
    fig, ax = plt.subplots(figsize=(6.5, 3.65), layout='constrained')
    for column, color, label in [('dqx', '#1C78A5', r'$\Delta Q_x$'),
                                  ('dqy', '#C34D25', r'$\Delta Q_y$')]:
        ax.plot(scan.ring_charge_nC, scan[column], color=color, label=label, zorder=2)
        ax.scatter(scan.loc[unstable, 'ring_charge_nC'], scan.loc[unstable, column],
                   color='black', marker='x', s=25, linewidth=.9, zorder=4)
        ax.scatter(targets.ring_charge_nC, targets[column], color=color,
                   s=22, marker='o', zorder=3)
    main = targets.loc[np.isclose(targets.dqy, -.2, rtol=0, atol=1e-8)].iloc[0]
    ax.axvline(main.ring_charge_nC, color='#777777', lw=.8, ls=':', zorder=1)
    ax.annotate('Principal comparison\n'+rf'$Q_{{\rm ring}}={main.ring_charge_nC:.3f}\,\mathrm{{nC}}$',
                xy=(main.ring_charge_nC, main.dqy), xytext=(7.2, -.055),
                fontsize=9, ha='left', va='top',
                arrowprops=dict(arrowstyle='->', color='#555555', lw=.8))
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color='black', marker='x', ls='none', markersize=5))
    labels.append('Envelope-unstable sample')
    ax.legend(handles, labels, loc='lower left')
    ax.set(xlabel=r'Total coasting ring charge $Q_{\rm ring}$ [nC]',
           ylabel=r'Incoherent tune shift $\Delta Q$',
           xlim=(-.3, 21.2), ylim=(-.445, .018))
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, -.1, -.2, -.3, -.4])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#D9DEE2', linewidth=.6, alpha=.8)
    save(fig, 'kv-tune-shift-charge')
    return unstable


def main():
    """Load immutable inputs, validate normalization and record provenance."""
    style()
    OUTPUT.mkdir(exist_ok=True)
    index = ROOT/'REFERENCE_RUNS.json'
    baseline = ROOT/json.loads(index.read_text())['baseline']
    machine_path = ROOT/'model/machine.json'
    machine = json.loads(machine_path.read_text())
    summary = json.loads((baseline/'summary.json').read_text())
    scan = pd.read_csv(baseline/'current_scan.csv').sort_values('ring_charge_nC')
    targets = pd.read_csv(baseline/'targets.csv').sort_values('ring_charge_nC')
    expected = scan.current_mA*1e-3*summary['circumference_m']/(summary['beta_rel']*c)*1e9
    np.testing.assert_allclose(scan.ring_charge_nC, expected, atol=1e-13, rtol=1e-14)
    np.testing.assert_allclose(scan.dqx, scan.qx-summary['bare_tunes'][0], atol=2e-15)
    np.testing.assert_allclose(scan.dqy, scan.qy-summary['bare_tunes'][1], atol=2e-15)
    np.testing.assert_allclose(machine['circumference_m'], summary['circumference_m'], atol=1e-13)
    assert machine['position_closure_m'] < 1e-12
    assert len(machine['elements']) == 30
    assert {k: sum(e['type'] == k for e in machine['elements']) for k in
            ['sector_bend', 'quadrupole', 'drift']} == {'sector_bend': 12, 'quadrupole': 6, 'drift': 12}
    ring_figure(machine)
    unstable = tune_figure(scan, targets)
    files = [Path(__file__).resolve(), machine_path, index, baseline/'current_scan.csv',
             baseline/'targets.csv', baseline/'summary.json']
    provenance = dict(
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scan_points=len(scan), unstable_samples=int(unstable.sum()),
        envelope_unity_plot_atol=ENVELOPE_UNITY_ATOL,
        classification_note='Plot labels only: unity roundoff <1.5e-12, first unstable excess >5e-3; no solver tolerance changed.',
        last_stable_sample_charge_nC=float(scan.loc[~unstable, 'ring_charge_nC'].max()),
        first_unstable_sample_charge_nC=float(scan.loc[unstable, 'ring_charge_nC'].min()),
        max_charge_conversion_error_nC=float(np.max(np.abs(scan.ring_charge_nC-expected))),
        scope='Saved analytic KV reference; no new simulation; magnet widths schematic.',
        matplotlib=matplotlib.__version__, numpy=np.__version__, pandas=pd.__version__)
    (OUTPUT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
