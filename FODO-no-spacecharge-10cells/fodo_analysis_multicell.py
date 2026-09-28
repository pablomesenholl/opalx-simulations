#!/usr/bin/env python3
"""
Analyze an OPALX no-space-charge FODO run.

Usage:
    python fodo_analysis.py run.stat run.h5 --input FODO-no-spacecharge.in --output plots
"""

from __future__ import annotations

import argparse
import io
import re
from contextlib import redirect_stdout
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


def parse_real_literal(text: str, name: str, default: float) -> float:
    """Read a literal REAL assignment such as REAL LQ=0.5; from an OPALX input."""
    pattern = rf"\bREAL\s+{re.escape(name)}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*;"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return float(m.group(1)) if m else default


def parse_fodo_parameters(input_path: Path | None) -> dict[str, float]:
    """Read the literal FODO parameters needed for the analytic model."""
    defaults = {
        "LQ": 0.5,
        "LD": 0.5,
        "k1": 1.5,
        "sigma_x": 1e-3,
        "sigma_y": 1e-3,
        "n_cells": 1.0,
    }

    if input_path is None:
        return defaults

    text = input_path.read_text(errors="replace")
    return {
        name: parse_real_literal(text, name, default)
        for name, default in defaults.items()
    }


def parse_lattice_lengths(input_path: Path | None) -> tuple[float, float]:
    """Backward-compatible helper used by the plotting functions."""
    p = parse_fodo_parameters(input_path)
    return p["LQ"], p["LD"]



def element_matrix(K: float, length: float) -> np.ndarray:
    """
    Exact 2x2 transfer matrix for a drift or hard-edge quadrupole.

    Coordinates are (x, x') or (y, y').
    K > 0: focusing
    K < 0: defocusing
    K = 0: drift
    """
    if abs(K) < 1e-15:
        return np.array([[1.0, length], [0.0, 1.0]])

    k = np.sqrt(abs(K))
    u = k * length

    if K > 0.0:
        c = np.cos(u)
        s = np.sin(u)
        return np.array([[c, s / k], [-k * s, c]])

    ch = np.cosh(u)
    sh = np.sinh(u)
    return np.array([[ch, sh / k], [k * sh, ch]])


def fodo_elements(params: dict[str, float], plane: str):
    """Return [(K, length), ...] for one FODO cell in x or y."""
    LQ = params["LQ"]
    LD = params["LD"]
    k1 = params["k1"]

    if plane == "x":
        return [(+k1, LQ), (0.0, LD), (-k1, LQ), (0.0, LD)]
    if plane == "y":
        return [(-k1, LQ), (0.0, LD), (+k1, LQ), (0.0, LD)]

    raise ValueError("plane must be 'x' or 'y'")


def transfer_matrix_to_s(
    s: float, params: dict[str, float], plane: str
) -> np.ndarray:
    """Exact transfer matrix from the cell entrance to longitudinal position s."""
    M = np.eye(2)
    remaining = float(s)

    for K, length in fodo_elements(params, plane):
        if remaining <= 0.0:
            break

        ds = min(remaining, length)
        M = element_matrix(K, ds) @ M
        remaining -= ds

    return M


def periodic_twiss(M: np.ndarray) -> tuple[float, float, float, float]:
    """
    Extract periodic (beta, alpha, gamma) and phase advance mu
    from a stable one-cell matrix.
    """
    cos_mu = 0.5 * np.trace(M)

    if abs(cos_mu) >= 1.0:
        raise ValueError(
            f"Cell is not linearly stable: |Tr(M)/2| = {abs(cos_mu):.6g}"
        )

    # Choose the branch that gives beta > 0 for this FODO cell.
    sin_abs = np.sqrt(1.0 - cos_mu**2)
    sin_mu = np.copysign(sin_abs, M[0, 1])

    beta = M[0, 1] / sin_mu
    alpha = (M[0, 0] - M[1, 1]) / (2.0 * sin_mu)
    gamma = -M[1, 0] / sin_mu
    mu = np.arctan2(sin_mu, cos_mu)

    if mu < 0.0:
        mu += 2.0 * np.pi

    return beta, alpha, gamma, mu


def analytic_matched_model(
    params: dict[str, float], points_per_cell: int = 500
) -> dict[str, np.ndarray | float]:
    """
    Build the exact linear matched-envelope prediction for repeated FODO cells.

    Because the initial covariance is the periodic matched covariance of one cell,
    the second moments repeat every Lcell. We therefore evaluate the exact
    thick-lens transport inside each cell and repeat it over the requested run.
    """
    Lcell = 2.0 * (params["LQ"] + params["LD"])
    n_cells = int(round(params.get("n_cells", 1.0)))

    if n_cells < 1:
        raise ValueError("n_cells must be at least 1")

    Ltotal = n_cells * Lcell
    npoints = n_cells * points_per_cell + 1
    s_grid = np.linspace(0.0, Ltotal, npoints)

    result: dict[str, np.ndarray | float] = {
        "s": s_grid,
        "Lcell": Lcell,
        "Ltotal": Ltotal,
        "n_cells": float(n_cells),
    }

    for plane in ("x", "y"):
        M_cell = transfer_matrix_to_s(Lcell, params, plane)
        beta0, alpha0, gamma0, mu = periodic_twiss(M_cell)

        sigma0 = params[f"sigma_{plane}"]
        emit = sigma0**2 / beta0

        Sigma0 = emit * np.array(
            [[beta0, -alpha0], [-alpha0, gamma0]]
        )

        sigmas = np.empty_like(s_grid)
        correlations = np.empty_like(s_grid)

        for i, s in enumerate(s_grid):
            # For a matched beam, Sigma is periodic from cell to cell.
            local_s = np.mod(float(s), Lcell)

            # Keep exact cell boundaries at local_s = 0 rather than values
            # extremely close to Lcell because of floating-point roundoff.
            if np.isclose(local_s, 0.0, atol=1e-12) or np.isclose(
                local_s, Lcell, atol=1e-12
            ):
                local_s = 0.0

            M = transfer_matrix_to_s(local_s, params, plane)
            Sigma = M @ Sigma0 @ M.T

            sigmas[i] = np.sqrt(Sigma[0, 0])
            correlations[i] = Sigma[0, 1] / np.sqrt(
                Sigma[0, 0] * Sigma[1, 1]
            )

        result[f"sigma_{plane}"] = sigmas
        result[f"corr_{plane}"] = correlations
        result[f"beta0_{plane}"] = beta0
        result[f"alpha0_{plane}"] = alpha0
        result[f"gamma0_{plane}"] = gamma0
        result[f"emit_{plane}"] = emit
        result[f"mu_{plane}"] = mu

    return result


def lattice_markers(
    ax, LQ: float, LD: float, n_cells: int = 1
) -> None:
    """Mark element and cell boundaries without overcrowding long runs."""
    Lcell = 2.0 * (LQ + LD)

    if n_cells <= 20:
        # Light element boundaries inside every cell.
        for i in range(n_cells):
            s0 = i * Lcell
            for x in (
                s0 + LQ,
                s0 + LQ + LD,
                s0 + 2.0 * LQ + LD,
            ):
                ax.axvline(x, linewidth=0.5, alpha=0.12)

        # Cell boundaries and cell numbers.
        for i in range(n_cells + 1):
            ax.axvline(i * Lcell, linewidth=0.8, alpha=0.30)

        for i in range(n_cells):
            ax.text(
                (i + 0.5) * Lcell,
                0.98,
                f"cell {i + 1}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
            )
    else:
        # For very long runs, only show a manageable subset of cell boundaries.
        stride = max(1, n_cells // 20)
        for i in range(0, n_cells + 1, stride):
            ax.axvline(i * Lcell, linewidth=0.7, alpha=0.25)
def save_figure(fig, outpath: Path) -> None:
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_envelopes(
    stat,
    outdir: Path,
    LQ: float,
    LD: float,
    theory: dict[str, np.ndarray | float] | None = None,
    n_cells: int = 1,
) -> None:
    fig, ax = plt.subplots()
    s = stat["s"]

    ax.plot(s, 1e3 * stat["rms_x"], label=r"OPALX $\sigma_x$")
    ax.plot(s, 1e3 * stat["rms_y"], label=r"OPALX $\sigma_y$")

    if theory is not None:
        ax.plot(
            theory["s"],
            1e3 * theory["sigma_x"],
            "--",
            linewidth=1.5,
            label=r"analytic $\sigma_x$",
        )
        ax.plot(
            theory["s"],
            1e3 * theory["sigma_y"],
            "--",
            linewidth=1.5,
            label=r"analytic $\sigma_y$",
        )

    lattice_markers(ax, LQ, LD, n_cells)
    ax.set_xlabel("Path length s [m]")
    ax.set_ylabel("RMS beam size [mm]")
    ax.set_title("FODO transverse RMS envelopes")
    ax.legend()
    ax.grid(alpha=0.25)

    save_figure(fig, outdir / "01_envelopes.png")


def plot_emittance(stat, outdir: Path, LQ: float, LD: float, n_cells: int = 1) -> None:
    fig, ax = plt.subplots()
    s = stat["s"]

    ax.plot(s, 1e6 * stat["emit_x"], label=r"$\varepsilon_{n,x}$")
    ax.plot(s, 1e6 * stat["emit_y"], label=r"$\varepsilon_{n,y}$")

    lattice_markers(ax, LQ, LD, n_cells)
    ax.set_xlabel("Path length s [m]")
    ax.set_ylabel("Normalized RMS emittance [mm mrad]")
    ax.set_title("Normalized transverse RMS emittance")
    ax.legend()
    ax.grid(alpha=0.25)

    save_figure(fig, outdir / "02_emittance.png")


def plot_correlations(
    stat,
    outdir: Path,
    LQ: float,
    LD: float,
    theory: dict[str, np.ndarray | float] | None = None,
    n_cells: int = 1,
) -> None:
    fig, ax = plt.subplots()
    s = stat["s"]

    ax.plot(s, stat["xpx"], label=r"OPALX $r_{x,p_x}$")
    ax.plot(s, stat["ypy"], label=r"OPALX $r_{y,p_y}$")

    if theory is not None:
        ax.plot(
            theory["s"],
            theory["corr_x"],
            "--",
            linewidth=1.5,
            label=r"analytic $r_x$",
        )
        ax.plot(
            theory["s"],
            theory["corr_y"],
            "--",
            linewidth=1.5,
            label=r"analytic $r_y$",
        )

    lattice_markers(ax, LQ, LD, n_cells)
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



def print_analytic_validation(
    stat,
    theory: dict[str, np.ndarray | float],
) -> None:
    """Print full-run closure and OPALX-vs-analytic envelope errors."""
    s = stat["s"]

    theory_sx = np.interp(s, theory["s"], theory["sigma_x"])
    theory_sy = np.interp(s, theory["s"], theory["sigma_y"])

    sx = stat["rms_x"]
    sy = stat["rms_y"]

    rms_err_x = np.sqrt(np.mean(((sx - theory_sx) / theory_sx) ** 2)) * 100.0
    rms_err_y = np.sqrt(np.mean(((sy - theory_sy) / theory_sy) ** 2)) * 100.0

    closure_sx = (sx[-1] - sx[0]) / sx[0] * 100.0
    closure_sy = (sy[-1] - sy[0]) / sy[0] * 100.0
    closure_rx = stat["xpx"][-1] - stat["xpx"][0]
    closure_ry = stat["ypy"][-1] - stat["ypy"][0]

    n_cells = int(round(theory["n_cells"]))

    print(f"\nAnalytic {n_cells}-cell validation")
    print("-" * (25 + len(str(n_cells))))
    print(
        f"phase advance x: {np.degrees(theory['mu_x']):.6f} deg/cell"
    )
    print(
        f"phase advance y: {np.degrees(theory['mu_y']):.6f} deg/cell"
    )
    print(
        f"total phase advance x: "
        f"{n_cells*np.degrees(theory['mu_x']):.6f} deg"
    )
    print(
        f"total phase advance y: "
        f"{n_cells*np.degrees(theory['mu_y']):.6f} deg"
    )
    print()
    print(
        f"OPALX-vs-analytic envelope RMS error x: {rms_err_x:.3f} %"
    )
    print(
        f"OPALX-vs-analytic envelope RMS error y: {rms_err_y:.3f} %"
    )
    print()
    print(f"full-run sigma_x closure: {closure_sx:+.4f} %")
    print(f"full-run sigma_y closure: {closure_sy:+.4f} %")
    print(f"full-run r_x closure:     {closure_rx:+.6f}")
    print(f"full-run r_y closure:     {closure_ry:+.6f}")
    print()
    print(
        "Note: the analytic curves use the exact requested RMS sizes from the "
        "input file, whereas OPALX tracks a finite random macroparticle sample. "
        "Small percent-level offsets are therefore normal."
    )


def print_cell_boundary_summary(
    stat,
    Lcell: float,
    n_cells: int,
) -> None:
    """
    Print OPALX statistics nearest to every nominal FODO-cell boundary.

    The .stat rows generally do not fall exactly on the requested boundary,
    so the nearest available statistics row is reported together with its
    actual s position.
    """
    s = stat["s"]

    print("\nCell-boundary statistics")
    print("------------------------")
    print(
        "cell  target_s   actual_s   sigma_x   sigma_y   "
        "emit_nx   emit_ny      r_x       r_y"
    )
    print(
        "      [m]        [m]        [mm]      [mm]      "
        "[mm mrad] [mm mrad]"
    )

    for cell in range(n_cells + 1):
        target = cell * Lcell
        idx = int(np.argmin(np.abs(s - target)))

        print(
            f"{cell:4d}  "
            f"{target:8.3f}  "
            f"{s[idx]:8.4f}  "
            f"{1e3*stat['rms_x'][idx]:8.4f}  "
            f"{1e3*stat['rms_y'][idx]:8.4f}  "
            f"{1e6*stat['emit_x'][idx]:8.4f}  "
            f"{1e6*stat['emit_y'][idx]:8.4f}  "
            f"{stat['xpx'][idx]:+8.4f}  "
            f"{stat['ypy'][idx]:+8.4f}"
        )

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
    params = parse_fodo_parameters(args.input)

    LQ = params["LQ"]
    LD = params["LD"]
    n_cells = int(round(params["n_cells"]))
    Lcell = 2.0 * (LQ + LD)

    theory = analytic_matched_model(params)

    plot_envelopes(
        stat, args.output, LQ, LD, theory, n_cells=n_cells
    )
    plot_emittance(
        stat, args.output, LQ, LD, n_cells=n_cells
    )
    plot_correlations(
        stat, args.output, LQ, LD, theory, n_cells=n_cells
    )
    plot_phase_space(args.h5, args.output, "x")
    plot_phase_space(args.h5, args.output, "y")

    # Capture the same numerical report that is printed to the terminal.
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        print_summary(stat, args.h5)
        print_analytic_validation(stat, theory)
        print_cell_boundary_summary(stat, Lcell, n_cells)

    report = buffer.getvalue()

    # Print it normally.
    print(report, end="")

    commentary = f"""
Interpretation
--------------
This run contains {n_cells} consecutive FODO cells.

For a linearly matched beam, sigma_x, sigma_y and the transverse
phase-space correlations should repeat at every FODO-cell boundary.
The beam sizes are expected to oscillate inside each cell.

The normalized transverse emittances should remain approximately
constant because this benchmark uses a linear lattice with space
charge disabled.

The analytic curves use exact hard-edge thick-lens transfer matrices
for the drifts and quadrupoles. The phase advance printed above is the
one-cell phase advance; the total phase advance is its accumulated
value over all {n_cells} cells.

The cell-boundary table uses the .stat row nearest to each nominal
boundary. These rows are generally slightly displaced from the exact
multiples of Lcell, so very small boundary-to-boundary differences are
expected even for an ideally matched run.

The first HDF5 dump is generally not at s=0. Therefore the phase-space
scatter plots compare the first available HDF5 snapshot with the last
snapshot; they are not an exact entrance-versus-exit closure test.
"""

    report_path = args.output / "validation_report.txt"
    report_path.write_text(
        report + "\n" + commentary.strip() + "\n"
    )

    print(f"\nValidation report written to: {report_path.resolve()}")
    print(f"Plots written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
