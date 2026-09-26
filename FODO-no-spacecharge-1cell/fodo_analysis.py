#!/usr/bin/env python3
"""
Analyze an OPALX no-space-charge FODO run.

Usage:
    python fodo_analysis.py run.stat run.h5 --input FODO-no-spacecharge.in --output plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_opal_stat(path: Path) -> dict[str, np.ndarray]:
    """Read the ASCII SDDS .stat file written by OPALX."""
    lines = path.read_text(errors="replace").splitlines()

    columns = []
    data_header_end = None
    state = None

    for i, line in enumerate(lines):
        s = line.strip()

        if s == "&column":
            state = "column"
            continue
        if s == "&data":
            state = "data"
            continue
        if s == "&end":
            if state == "data":
                data_header_end = i + 1
            state = None
            continue
        if s.startswith("&"):
            state = "other"
            continue

        if state == "column":
            m = re.match(r"name=([^,]+),", s)
            if m:
                columns.append(m.group(1))

    if not columns or data_header_end is None:
        raise ValueError(f"Could not parse SDDS header in {path}")

    rows = []
    ncols = len(columns)

    for line in lines[data_header_end:]:
        fields = line.split()
        if len(fields) != ncols:
            continue
        try:
            rows.append([float(v) for v in fields])
        except ValueError:
            continue

    if not rows:
        raise ValueError(f"No numerical rows found in {path}")

    data = np.asarray(rows, dtype=float)
    return {name: data[:, i] for i, name in enumerate(columns)}


def parse_lattice_lengths(input_path: Path | None) -> tuple[float, float]:
    """Read literal LQ and LD definitions if an input file is supplied."""
    if input_path is None:
        return 0.5, 0.5

    text = input_path.read_text(errors="replace")

    def find_literal(name: str, default: float) -> float:
        pattern = rf"\bREAL\s+{re.escape(name)}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*;"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return float(m.group(1)) if m else default

    return find_literal("LQ", 0.5), find_literal("LD", 0.5)


def lattice_markers(ax, LQ: float, LD: float) -> None:
    bounds = [0.0, LQ, LQ + LD, 2.0 * LQ + LD, 2.0 * (LQ + LD)]
    labels = ["QF", "D1", "QD", "D2"]

    for x in bounds[1:-1]:
        ax.axvline(x, linewidth=0.8, alpha=0.5)

    for left, right, label in zip(bounds[:-1], bounds[1:], labels):
        ax.text(
            0.5 * (left + right),
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )


def save_figure(fig, outpath: Path) -> None:
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_envelopes(stat, outdir: Path, LQ: float, LD: float) -> None:
    fig, ax = plt.subplots()
    s = stat["s"]

    ax.plot(s, 1e3 * stat["rms_x"], label=r"$\sigma_x$")
    ax.plot(s, 1e3 * stat["rms_y"], label=r"$\sigma_y$")

    lattice_markers(ax, LQ, LD)
    ax.set_xlabel("Path length s [m]")
    ax.set_ylabel("RMS beam size [mm]")
    ax.set_title("FODO transverse RMS envelopes")
    ax.legend()
    ax.grid(alpha=0.25)

    save_figure(fig, outdir / "01_envelopes.png")


def plot_emittance(stat, outdir: Path, LQ: float, LD: float) -> None:
    fig, ax = plt.subplots()
    s = stat["s"]

    ax.plot(s, 1e6 * stat["emit_x"], label=r"$\varepsilon_{n,x}$")
    ax.plot(s, 1e6 * stat["emit_y"], label=r"$\varepsilon_{n,y}$")

    lattice_markers(ax, LQ, LD)
    ax.set_xlabel("Path length s [m]")
    ax.set_ylabel("Normalized RMS emittance [mm mrad]")
    ax.set_title("Normalized transverse RMS emittance")
    ax.legend()
    ax.grid(alpha=0.25)

    save_figure(fig, outdir / "02_emittance.png")


def plot_correlations(stat, outdir: Path, LQ: float, LD: float) -> None:
    fig, ax = plt.subplots()
    s = stat["s"]

    ax.plot(s, stat["xpx"], label=r"$r_{x,p_x}$")
    ax.plot(s, stat["ypy"], label=r"$r_{y,p_y}$")

    lattice_markers(ax, LQ, LD)
    ax.set_xlabel("Path length s [m]")
    ax.set_ylabel("Correlation coefficient")
    ax.set_title("Transverse phase-space correlations")
    ax.legend()
    ax.grid(alpha=0.25)

    save_figure(fig, outdir / "03_correlations.png")


def sorted_h5_steps(h5: h5py.File):
    steps = []
    for name in h5.keys():
        m = re.fullmatch(r"Step#(\d+)", name)
        if m:
            steps.append((int(m.group(1)), name))
    return [name for _, name in sorted(steps)]


def spos(group) -> float:
    value = np.asarray(group.attrs["SPOS"]).reshape(-1)
    return float(value[0])


def plot_phase_space(h5_path: Path, outdir: Path, plane: str) -> None:
    coord = plane
    mom = "p" + plane

    with h5py.File(h5_path, "r") as h5:
        names = sorted_h5_steps(h5)
        if not names:
            raise ValueError("No Step# groups found in HDF5 file")

        first = h5[names[0]]
        last = h5[names[-1]]

        x0 = np.asarray(first[coord]) * 1e3
        p0 = np.asarray(first[mom]) * 1e3
        x1 = np.asarray(last[coord]) * 1e3
        p1 = np.asarray(last[mom]) * 1e3

        s0 = spos(first)
        s1 = spos(last)

    max_points = 6000
    stride0 = max(1, len(x0) // max_points)
    stride1 = max(1, len(x1) // max_points)

    fig, ax = plt.subplots()
    ax.scatter(
        x0[::stride0],
        p0[::stride0],
        s=4,
        alpha=0.20,
        label=f"first H5 dump, s={s0:.3f} m",
    )
    ax.scatter(
        x1[::stride1],
        p1[::stride1],
        s=4,
        alpha=0.20,
        label=f"last H5 dump, s={s1:.3f} m",
    )

    ax.set_xlabel(f"{plane} [mm]")
    ax.set_ylabel(rf"$p_{plane}/(mc)$ [$10^{{-3}}$]")
    ax.set_title(f"{plane}-plane phase space")
    ax.legend(markerscale=3)
    ax.grid(alpha=0.25)

    save_figure(fig, outdir / f"04_phase_space_{plane}.png")


def print_summary(stat, h5_path: Path) -> None:
    i0, i1 = 0, -1

    sx0, sx1 = stat["rms_x"][i0], stat["rms_x"][i1]
    sy0, sy1 = stat["rms_y"][i0], stat["rms_y"][i1]
    ex0, ex1 = stat["emit_x"][i0], stat["emit_x"][i1]
    ey0, ey1 = stat["emit_y"][i0], stat["emit_y"][i1]
    e0, e1 = stat["energy"][i0], stat["energy"][i1]

    def pct(a, b):
        return 100.0 * (b - a) / a

    with h5py.File(h5_path, "r") as h5:
        names = sorted_h5_steps(h5)
        s_h5 = [spos(h5[name]) for name in names]

    print("\nRun summary")
    print("-----------")
    print(f".stat range: {stat['s'][0]:.6f} -> {stat['s'][-1]:.6f} m")
    print(f"H5 dumps:    {len(s_h5)} ({s_h5[0]:.6f} -> {s_h5[-1]:.6f} m)")
    print(f"Particles:   {int(stat['numParticles'][0])} -> {int(stat['numParticles'][-1])}")
    print()
    print(f"sigma_x: {1e3*sx0:.6f} -> {1e3*sx1:.6f} mm  ({pct(sx0,sx1):+.3f} %)")
    print(f"sigma_y: {1e3*sy0:.6f} -> {1e3*sy1:.6f} mm  ({pct(sy0,sy1):+.3f} %)")
    print(f"emit_nx: {1e6*ex0:.6f} -> {1e6*ex1:.6f} mm mrad  ({pct(ex0,ex1):+.5f} %)")
    print(f"emit_ny: {1e6*ey0:.6f} -> {1e6*ey1:.6f} mm mrad  ({pct(ey0,ey1):+.5f} %)")
    print(f"energy:  {e0:.9f} -> {e1:.9f} MeV  ({pct(e0,e1):+.3e} %)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stat", type=Path, help="OPALX .stat file")
    parser.add_argument("h5", type=Path, help="OPALX phase-space .h5 file")
    parser.add_argument("--input", type=Path, default=None, help="optional OPALX .in file")
    parser.add_argument("--output", type=Path, default=Path("plots"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    stat = read_opal_stat(args.stat)
    LQ, LD = parse_lattice_lengths(args.input)

    plot_envelopes(stat, args.output, LQ, LD)
    plot_emittance(stat, args.output, LQ, LD)
    plot_correlations(stat, args.output, LQ, LD)
    plot_phase_space(args.h5, args.output, "x")
    plot_phase_space(args.h5, args.output, "y")
    print_summary(stat, args.h5)

    print(f"\nPlots written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
