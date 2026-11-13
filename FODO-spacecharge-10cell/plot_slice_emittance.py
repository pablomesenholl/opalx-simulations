#!/usr/bin/env python3
"""Longitudinal slice-emittance analysis for OPALX H5Part particle dumps.

Reads the real particle datasets x, y, z, px, py from every Step#N group.
The transverse momenta in OPALX H5Part are normalized as p/(mc) = beta*gamma;
therefore sqrt(det(Cov[x,px])) is the *normalized* RMS emittance in m rad.

For each dump, split particles into equal-population longitudinal (z) slices:
  1. Whole-bunch projected emittance.
  2. Charge-weighted mean of the individual slice emittances.
  3. Emittance from the average *within-slice covariance* (slice centroids
     subtracted) -- distinguishes centroid offsets from slice-optics mismatch.

These are not additive emittances. Differences are useful diagnostics:
  projected - within-slice: effect of different slice centroids;
  within-slice - mean-slice: effect of different within-slice covariance shapes.

Note: HDF5 dumps are relatively sparse in the current 10-cell run. They do
not time-resolve the very short emittance features at magnet boundaries.

Usage:
    python3 plot_slice_emittance.py FODO-spacecharge.h5
    python3 plot_slice_emittance.py . --slices 10
    python3 plot_slice_emittance.py data/FODO-spacecharge.h5 --outdir slice_plots

Outputs: PNG plots, a per-slice CSV, a per-dump summary CSV and a text summary.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")  # Save figures without a graphical desktop.
import matplotlib.pyplot as plt
import numpy as np

MM_MRAD = 1e6  # m rad -> mm mrad
PARTICLE_FIELDS = ("z", "x", "px", "y", "py")


def select_h5(source: Path) -> Path:
    """Accept an HDF5 path or a directory containing one particle HDF5 file."""
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"HDF5 file or directory does not exist: {source}")
    candidates = []
    for path in sorted(source.rglob("*.h5")):
        try:
            with h5py.File(path, "r") as f:
                if any(re.fullmatch(r"Step#\d+", key) and
                       set(PARTICLE_FIELDS).issubset(f[key].keys()) for key in f.keys()):
                    candidates.append(path)
        except OSError:
            pass
    if len(candidates) != 1:
        raise ValueError(
            f"Found {len(candidates)} compatible HDF5 files in {source}. "
            "Pass the exact .h5 filename instead.\n" + "\n".join(map(str, candidates))
        )
    return candidates[0]


def weighted_moments(a: np.ndarray, b: np.ndarray, w: np.ndarray):
    """Population covariance of (a,b), normalized positive macroparticle weights."""
    total = w.sum()
    if total <= 0:
        raise ValueError("Slice contains no positive particle weight")
    wn = w / total
    ma, mb = np.dot(wn, a), np.dot(wn, b)
    da, db = a - ma, b - mb
    va = np.dot(wn, da * da)
    vb = np.dot(wn, db * db)
    cab = np.dot(wn, da * db)
    cov = np.array([[va, cab], [cab, vb]], dtype=float)
    return np.array([ma, mb]), cov


def emittance(cov: np.ndarray) -> float:
    """Normalized RMS emittance in mm mrad for x/px or y/py covariance."""
    det = float(cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2)
    # Roundoff can produce a tiny negative determinant for degenerate data.
    return MM_MRAD * np.sqrt(max(0.0, det))


def correlation(cov: np.ndarray) -> float:
    scale = np.sqrt(max(0, cov[0, 0]) * max(0, cov[1, 1]))
    return float(np.clip(cov[0, 1] / scale, -1, 1)) if scale else np.nan


def analyze_step(group: h5py.Group, n_slices: int):
    if "SPOS" not in group.attrs:
        raise ValueError(f"{group.name} has no SPOS attribute; cannot plot vs distance")
    s = float(np.asarray(group.attrs["SPOS"]).reshape(-1)[0])
    data = {name: np.asarray(group[name], dtype=float).ravel()
            for name in PARTICLE_FIELDS}
    n = len(data["z"])
    if n < n_slices * 20:
        raise ValueError(f"{group.name}: only {n} particles for {n_slices} slices")
    if any(len(arr) != n or not np.all(np.isfinite(arr)) for arr in data.values()):
        raise ValueError(f"{group.name}: nonfinite or inconsistent particle arrays")

    # Use per-particle charge magnitudes for generality; OPALX's provided
    # FODO run has equal positive macroparticle charges (1e-13 C each).
    if "q" in group:
        charges = np.abs(np.asarray(group["q"], dtype=float).ravel())
        if len(charges) != n or not np.all(np.isfinite(charges)) or charges.sum() <= 0:
            raise ValueError(f"{group.name}: invalid macroparticle charges")
    else:
        charges = np.ones(n)

    # Equal-population slices are robust for a growing, nonuniform bunch.
    # Slice 1 has the lowest z, slice N the highest z at each snapshot.
    order = np.argsort(data["z"], kind="stable")
    partitions = np.array_split(order, n_slices)
    snapshot = {"s_m": s, "n_particles": n}
    slice_rows = []

    for plane, pos_name, mom_name in (("x", "x", "px"), ("y", "y", "py")):
        a, b, w = data[pos_name], data[mom_name], charges
        mean_all, cov_all = weighted_moments(a, b, w)
        projected = emittance(cov_all)
        within_cov = np.zeros((2, 2))
        mean_slice_emit = 0.0
        weighted_centroids = []
        slice_weights = []

        for j, ids in enumerate(partitions, start=1):
            mj, cj = weighted_moments(a[ids], b[ids], w[ids])
            weight_fraction = float(w[ids].sum() / w.sum())
            ej = emittance(cj)
            within_cov += weight_fraction * cj
            mean_slice_emit += weight_fraction * ej
            weighted_centroids.append(mj)
            slice_weights.append(weight_fraction)
            slice_rows.append({
                "s_m": s, "plane": plane, "slice": j,
                "n_particles": len(ids), "charge_fraction": weight_fraction,
                "z_mean_mm": float(np.average(data["z"][ids], weights=w[ids]) * 1e3),
                "z_min_mm": float(np.min(data["z"][ids]) * 1e3),
                "z_max_mm": float(np.max(data["z"][ids]) * 1e3),
                "emittance_mm_mrad": ej,
                "correlation": correlation(cj),
                "mean_position_mm": float(mj[0] * 1e3),
                "mean_momentum_betagamma": float(mj[1]),
            })

        # Exact covariance decomposition: Sigma_total = average(Sigma_slice)
        # + covariance_of_slice_centroids.  Numerical cross-check included.
        centroid_cov = np.zeros((2, 2))
        for mj, fj in zip(weighted_centroids, slice_weights):
            d = mj - mean_all
            centroid_cov += fj * np.outer(d, d)
        if not np.allclose(cov_all, within_cov + centroid_cov, rtol=5e-8, atol=1e-17):
            raise RuntimeError(f"Covariance decomposition failed for {group.name}, {plane}")

        within = emittance(within_cov)
        snapshot.update({
            f"projected_{plane}_mm_mrad": projected,
            f"within_cov_{plane}_mm_mrad": within,
            f"mean_slice_{plane}_mm_mrad": mean_slice_emit,
            f"centroid_excess_{plane}_mm_mrad": projected - within,
            f"optics_excess_{plane}_mm_mrad": within - mean_slice_emit,
        })

        # This version's OPALX H5Part stores its normalized emittances
        # in #varepsilon = [x, y, longitudinal] in m rad.
        if "#varepsilon" in group.attrs:
            stored = float(np.asarray(group.attrs["#varepsilon"]).ravel()[0 if plane == "x" else 1]) * MM_MRAD
            if not np.isclose(projected, stored, rtol=1e-6, atol=1e-7):
                print(f"Warning: H5 vs recomputed projected {plane} emittance differs "
                      f"at s={s:.3f} m: {stored:.6g} vs {projected:.6g} mm mrad")

    return snapshot, slice_rows


def dump_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decorate_axis(ax, max_s: float, period: float):
    if period > 0:
        for s in np.arange(period, max_s, period):
            ax.axvline(s, color="0.65", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.grid(alpha=0.24)


def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_plots(summary: list[dict], slices: list[dict], outdir: Path, n_slices: int, period: float):
    s = np.array([r["s_m"] for r in summary])
    palette = {"x": "tab:blue", "y": "tab:orange"}
    name = {"x": "Horizontal (x)", "y": "Vertical (y)"}
    all_max_s = float(s.max())
    n_steps = len(summary)

    # 1. The most important direct comparison.
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, plane in zip(axs, ("x", "y")):
        ax.plot(s, [r[f"projected_{plane}_mm_mrad"] for r in summary],
                color=palette[plane], lw=2, label="Whole-bunch projected")
        ax.plot(s, [r[f"within_cov_{plane}_mm_mrad"] for r in summary],
                color=palette[plane], lw=1.8, ls="--", label="Within-slice covariance")
        ax.plot(s, [r[f"mean_slice_{plane}_mm_mrad"] for r in summary],
                color="0.3", lw=1.8, ls=":", label=f"Weighted mean of {n_slices} slice emittances")
        ax.set(ylabel="Normalized emittance [mm mrad]", title=name[plane])
        decorate_axis(ax, all_max_s, period)
        ax.legend(fontsize=9)
    axs[-1].set_xlabel("Path length s [m]")
    fig.suptitle("FODO (10 cells): projected and longitudinal slice emittance")
    save_fig(fig, outdir / "01_projected_vs_slice_emittance.png")

    # 2. Longitudinal quantile-slice emittance heatmaps over all dumps.
    by_key = {(r["s_m"], r["plane"], r["slice"]): r for r in slices}
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, plane, cmap in zip(axs, ("x", "y"), ("Blues", "Oranges")):
        em = np.array([[by_key[(s_i, plane, j)]["emittance_mm_mrad"]
                        for s_i in s] for j in range(1, n_slices + 1)])
        # pcolormesh requires sample edges rather than sample centres.
        if n_steps > 1:
            xedges = np.r_[s[0] - (s[1]-s[0])/2, (s[:-1]+s[1:])/2,
                            s[-1]+(s[-1]-s[-2])/2]
        else:
            xedges = np.array([s[0]-0.05, s[0]+0.05])
        xedges[0] = max(0, xedges[0])
        mesh = ax.pcolormesh(xedges, np.arange(0.5, n_slices + 1.5), em,
                             cmap=cmap, shading="flat")
        fig.colorbar(mesh, ax=ax, label="Normalized slice emittance [mm mrad]")
        ax.set(ylabel="Slice index (increasing z)", title=name[plane],
               yticks=np.arange(1, n_slices+1, max(1, n_slices//10)))
        ax.set_ylim(0.5, n_slices + 0.5)
        for boundary in np.arange(period, all_max_s, period) if period > 0 else []:
            ax.axvline(boundary, color="0.5", ls=":", lw=0.6, alpha=0.55)
    axs[-1].set_xlabel("Path length s [m]")
    fig.suptitle("FODO (10 cells): normalized emittance by longitudinal slice")
    save_fig(fig, outdir / "02_slice_emittance_heatmaps.png")

    # 3. Decomposition of projected emittance relative to slices.
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, plane in zip(axs, ("x", "y")):
        ax.plot(s, [r[f"centroid_excess_{plane}_mm_mrad"] for r in summary],
                color=palette[plane], lw=2, label="Projected - within-slice covariance")
        ax.plot(s, [r[f"optics_excess_{plane}_mm_mrad"] for r in summary],
                color="0.35", ls="--", lw=1.8,
                label="Within-slice covariance - mean slice")
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set(ylabel="Emittance difference [mm mrad]", title=name[plane])
        decorate_axis(ax, all_max_s, period)
        ax.legend(fontsize=9)
    axs[-1].set_xlabel("Path length s [m]")
    fig.suptitle("FODO (10 cells): contributions to projected RMS emittance")
    save_fig(fig, outdir / "03_inter_slice_contributions.png")

    # 4. Compare the first and last *available* dumps as profiles in z.
    fig, axs = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)
    for ax, plane in zip(axs, ("x", "y")):
        for idx, style, label in [(0, "--", f"First available dump (s={s[0]:.2f} m)"),
                                   (n_steps - 1, "-", f"Last available dump (s={s[-1]:.2f} m)")]:
            y = [by_key[(s[idx], plane, j)]["emittance_mm_mrad"]
                 for j in range(1, n_slices + 1)]
            ax.plot(range(1, n_slices+1), y, color=palette[plane],
                    linestyle=style, marker="o", lw=1.6, label=label)
        ax.set(ylabel="Slice emittance [mm mrad]", title=name[plane])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axs[-1].set(xlabel="Longitudinal slice index (low z to high z)",
                xticks=range(1, n_slices+1))
    fig.suptitle("FODO (10 cells): first and last slice-emittance profiles")
    save_fig(fig, outdir / "04_first_last_slice_profiles.png")

    # 5. Correlation variation by slice identifies phase-space rotation.
    fig, axs = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)
    for ax, plane in zip(axs, ("x", "y")):
        for idx, style, label in [(0, "--", f"First available dump (s={s[0]:.2f} m)"),
                                   (n_steps - 1, "-", f"Last available dump (s={s[-1]:.2f} m)")]:
            y = [by_key[(s[idx], plane, j)]["correlation"]
                 for j in range(1, n_slices+1)]
            ax.plot(range(1, n_slices+1), y, color=palette[plane],
                    linestyle=style, marker="o", lw=1.6, label=label)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set(ylabel="Pearson correlation", title=name[plane], ylim=(-1.05, 1.05))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axs[-1].set(xlabel="Longitudinal slice index (low z to high z)",
                xticks=range(1, n_slices+1))
    fig.suptitle("FODO (10 cells): longitudinal slice phase-space correlations")
    save_fig(fig, outdir / "05_first_last_slice_correlations.png")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?", default=".",
                   help="OPALX particle .h5 file or directory containing it (default: current directory)")
    p.add_argument("--slices", type=int, default=10,
                   help="Equal-population longitudinal slices per dump (default: 10)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output folder (default: <h5 directory>/slice_analysis)")
    p.add_argument("--period-length", type=float, default=2.0,
                   help="Cell length in metres for plot guides (default: 2.0; 0 to disable)")
    args = p.parse_args()
    if args.slices < 2:
        p.error("--slices must be at least 2")
    if args.period_length < 0:
        p.error("--period-length cannot be negative")

    h5 = select_h5(Path(args.source).expanduser())
    outdir = args.outdir or (h5.parent / "slice_analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    summary, rows = [], []
    with h5py.File(h5, "r") as f:
        steps = [(int(m.group(1)), name) for name in f.keys()
                 if (m := re.fullmatch(r"Step#(\d+)", name))]
        steps.sort()
        if not steps:
            raise ValueError(f"No Step#N groups found in {h5}")
        for _, step_name in steps:
            group = f[step_name]
            missing = [n for n in PARTICLE_FIELDS if n not in group]
            if missing:
                raise ValueError(f"{step_name} missing particle arrays: {missing}")
            result, slice_data = analyze_step(group, args.slices)
            summary.append(result)
            rows.extend(slice_data)

    summary.sort(key=lambda r: r["s_m"])
    rows.sort(key=lambda r: (r["s_m"], r["plane"], r["slice"]))
    dump_csv(outdir / "snapshot_summary.csv", summary)
    dump_csv(outdir / "longitudinal_slices.csv", rows)
    make_plots(summary, rows, outdir, args.slices, args.period_length)

    first, last = summary[0], summary[-1]
    report = [
        f"File: {h5}",
        f"Dumps: {len(summary)}; particles: {first['n_particles']} -> {last['n_particles']}",
        f"Range: s={first['s_m']:.4f} to {last['s_m']:.4f} m",
        f"Longitudinal slices: {args.slices} equal-population bins, assigned independently at each dump",
        "Emittance units: normalized RMS [mm mrad] from (position, p/(mc))",
        "",
    ]
    for plane in ("x", "y"):
        for label, rec in (("First", first), ("Last", last)):
            report.append(
                f"{label} {plane}: projected={rec[f'projected_{plane}_mm_mrad']:.6f}, "
                f"within-cov={rec[f'within_cov_{plane}_mm_mrad']:.6f}, "
                f"mean-slice={rec[f'mean_slice_{plane}_mm_mrad']:.6f} mm mrad")
    report.extend([
        "",
        "Projected / within-cov / mean-slice are different 2D covariance measures, not additive components.",
        "Projected - within-cov isolates the impact of differences in slice centroids.",
        "Within-cov - mean-slice reflects differing slice covariance ellipses.",
        "Dump spacing may be too coarse to resolve magnet-boundary transitions directly.",
    ])
    (outdir / "README_results.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved 5 PNGs, 2 CSVs and README_results.txt in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
