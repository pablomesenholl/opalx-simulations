#!/usr/bin/env python3
"""Plot OPALX 10-cell FODO observables, particle losses and phase-space dumps.

Requires numpy, matplotlib, h5py. The .stat file is an SDDS ASCII file;
particle snapshots are read from an OPALX/H5Part-style HDF5 file.

Usage:
    python3 analysis_script.py path/to/simulation_directory
    python3 analysis_script.py . --stat data/fodo.stat --h5 data/fodo.h5

The script reports exactly which files/dumps it selected and does not infer
statistics from unrelated HDF5 output files (monitors, checkpoints, losses).

It adds the survival fraction, macroparticle losses per 2 m FODO cell, and an
observables_summary.txt with before/after values. Without physical apertures,
actual particle loss would be unexpected in the supplied 10-period model.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_STAT = ("s", "rms_x", "rms_y", "emit_x", "emit_y")
REQUIRED_PARTICLE = ("x", "y", "px", "py", "pz")


def parse_stat(path: Path) -> dict[str, np.ndarray]:
    """Read an ASCII SDDS OPALX .stat file, using named &column headers.

    Handles multiline &column declarations, optional SDDS parameter values,
    and the common no_row_counts=1 convention. A row-count-only line is
    discarded for files that use row counts. Binary SDDS is not supported.
    """
    contents = path.read_text(encoding="utf-8", errors="replace")
    names = []
    for match in re.finditer(r"&column\b(.*?)&end\b", contents, re.I | re.S):
        m = re.search(r"\bname\s*=\s*[\"']?([\w.]+)", match.group(1), re.I)
        if m:
            names.append(m.group(1))
    if not names:
        raise ValueError(f"No SDDS &column names found in {path}")

    data_match = re.search(r"&data\b(.*?)&end\b", contents, re.I | re.S)
    if not data_match:
        raise ValueError(f"No SDDS &data block found in {path}")
    if re.search(r"\bmode\s*=\s*binary\b", data_match.group(1), re.I):
        raise ValueError("Binary SDDS not supported; use ASCII .stat output")
    no_row_counts = bool(re.search(r"\bno_row_counts\s*=\s*1\b", data_match.group(1)))

    # Statistics writers usually have no per-page parameters. Some SDDS files
    # do; in that case count and discard the parameter values before the rows.
    n_params = len(re.findall(r"&parameter\b", contents[:data_match.start()], re.I))
    rows = []
    parameter_lines_left = n_params
    for raw in contents[data_match.end():].splitlines():
        line = raw.strip()
        if not line or line.startswith(("!", "#", "%")):
            continue
        if parameter_lines_left:
            parameter_lines_left -= 1
            continue
        tokens = line.split()
        if not no_row_counts and len(tokens) == 1:
            # SDDS page row count, not a data row.
            continue
        if len(tokens) != len(names):
            raise ValueError(
                f".stat data has {len(tokens)} values but {len(names)} named columns: {line[:110]}"
            )
        try:
            rows.append([float(t.replace("D", "E").replace("d", "e")) for t in tokens])
        except ValueError as exc:
            raise ValueError(f"Non-numeric .stat data line: {line[:110]}") from exc
    if not rows:
        raise ValueError(f"No numerical statistics rows in {path}")
    arr = np.array(rows, dtype=float)
    return {name.lower(): arr[:, i] for i, name in enumerate(names)}


def get_column(stats: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key.lower() not in stats:
        raise KeyError(f"Missing .stat column '{key}'. Available: {', '.join(stats)}")
    return stats[key.lower()]


def get_particle_steps(path: Path) -> list[tuple[str, int | None]]:
    """Find step groups having x,y,px,py,pz; ignore checkpoint-style files."""
    found = []
    with h5py.File(path, "r") as f:
        def visit(name, node):
            if not isinstance(node, h5py.Group):
                return
            keys = {k.lower() for k, item in node.items() if isinstance(item, h5py.Dataset)}
            if set(REQUIRED_PARTICLE).issubset(keys):
                leaf = name.split("/")[-1]
                step = re.search(r"step[#_ -]?(\d+)$", leaf, re.I)
                if step:
                    found.append((name, int(step.group(1))))
        f.visititems(visit)
    return sorted(found, key=lambda entry: entry[1])


def step_description(path: Path, group_path: str, dump_index: int) -> str:
    """Prefer OPALX H5Part step attributes over the HDF5 dump index."""
    with h5py.File(path, "r") as f:
        group = f[group_path]
        attrs = {str(k).lower(): v for k, v in group.attrs.items()}
    parts = [f"dump {dump_index}"]
    if "spos" in attrs:
        parts.append(f"s={float(np.asarray(attrs['spos']).item()):.3f} m")
    return ", ".join(parts)


def load_snapshot(path: Path, group_path: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        group = f[group_path]
        keys = {k.lower(): k for k in group.keys()}
        data = {k: np.asarray(group[keys[k]], dtype=float).reshape(-1)
                for k in REQUIRED_PARTICLE}
    lengths = {a.size for a in data.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError(f"Invalid particle arrays in {path}:{group_path}")
    if any(not np.all(np.isfinite(a)) for a in data.values()):
        raise ValueError(f"Non-finite particle coordinates in {path}:{group_path}")
    return data


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    ax, bx = a - a.mean(), b - b.mean()
    denom = np.sqrt(np.mean(ax * ax) * np.mean(bx * bx))
    return float(np.mean(ax * bx) / denom) if denom > 0 else np.nan


def correlations(stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "xpx" in stats and "ypy" in stats:
        cx, cy = stats["xpx"], stats["ypy"]
    elif all(k in stats for k in ("s11", "s12", "s22", "s33", "s34", "s44")):
        with np.errstate(divide="ignore", invalid="ignore"):
            cx = stats["s12"] / np.sqrt(stats["s11"] * stats["s22"])
            cy = stats["s34"] / np.sqrt(stats["s33"] * stats["s44"])
    else:
        raise KeyError("Neither xpx/ypy nor S-matrix correlation columns are present")
    if np.nanmax(np.abs(np.r_[cx, cy])) > 1.0001:
        raise ValueError("Correlation entries exceed [-1,1]; check their meaning")
    return cx, cy


def choose_files(root: Path, stat_arg: str | None, h5_arg: str | None):
    def resolve(value):
        p = Path(value)
        return p if p.exists() else root / p

    if stat_arg:
        stat = resolve(stat_arg)
    else:
        stats = sorted(root.rglob("*.stat"))
        stats = [p for p in stats if "monitor" not in p.stem.lower()]
        if len(stats) != 1:
            raise ValueError(f"Found {len(stats)} candidate .stat files. Specify --stat. Candidates: {stats}")
        stat = stats[0]
    if not stat.is_file():
        raise FileNotFoundError(stat)

    if h5_arg:
        h5 = resolve(h5_arg)
        if not h5.is_file() or not get_particle_steps(h5):
            raise ValueError(f"No H5Part-style Step# particle dumps found in {h5}")
    else:
        candidates = []
        for p in sorted(root.rglob("*.h5")):
            if "checkpoint" in p.stem.lower():
                continue
            try:
                if get_particle_steps(p):
                    candidates.append(p)
            except OSError:
                continue
        same_stem = [p for p in candidates if p.stem == stat.stem]
        if len(same_stem) == 1:
            h5 = same_stem[0]
        elif len(candidates) == 1:
            h5 = candidates[0]
        else:
            raise ValueError(
                f"Found {len(candidates)} particle HDF5 candidates. "
                f"Specify --h5. Candidates: {candidates}"
            )
    return stat, h5


def make_evolution_plots(stats: dict[str, np.ndarray], outdir: Path):
    s = get_column(stats, "s")
    if np.any(np.diff(s) < -1e-9):
        print("Warning: path length s is not monotonic (restarts or multiple runs?)")

    # OPALX StatWriter labels emit_x / emit_y as normalized emittance in m.
    # 1 m rad = 1e6 mm mrad numerically.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(s, get_column(stats, "emit_x") * 1e6, label=r"$\varepsilon_{n,x}$")
    ax.plot(s, get_column(stats, "emit_y") * 1e6, label=r"$\varepsilon_{n,y}$")
    ax.set(xlabel="Path length s [m]", ylabel="Normalized RMS emittance [mm mrad]",
           title="FODO (10 cells): normalized emittance evolution")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(outdir / "01_emittance.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(s, get_column(stats, "rms_x") * 1e3, label=r"$\sigma_x$")
    ax.plot(s, get_column(stats, "rms_y") * 1e3, label=r"$\sigma_y$")
    ax.set(xlabel="Path length s [m]", ylabel=r"RMS envelope $\sigma$ [mm]",
           title="FODO (10 cells): RMS beam envelope evolution")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(outdir / "02_envelope.png", dpi=180); plt.close(fig)

    try:
        cx, cy = correlations(stats)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(s, cx, label=r"$\rho(x,p_x)$")
        ax.plot(s, cy, label=r"$\rho(y,p_y)$")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set(xlabel="Path length s [m]", ylabel="Pearson correlation coefficient",
               title="FODO (10 cells): transverse position-momentum correlations",
               ylim=(-1.05, 1.05))
        ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(outdir / "03_correlation.png", dpi=180); plt.close(fig)
    except (KeyError, ValueError) as exc:
        print(f"Warning: no .stat correlation evolution available: {exc}")
        return False
    return True


def make_h5_correlation_plot(h5: Path, steps, outdir: Path):
    vals = []
    for name, number in steps:
        d = load_snapshot(h5, name)
        vals.append([number, pearson(d["x"], d["px"]), pearson(d["y"], d["py"])])
    rows = np.array(vals, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(rows[:, 0], rows[:, 1], ".-", label=r"$\rho(x,p_x)$")
    ax.plot(rows[:, 0], rows[:, 2], ".-", label=r"$\rho(y,p_y)$")
    ax.set(xlabel="Particle dump index (not path length)",
           ylabel="Pearson correlation coefficient", ylim=(-1.05, 1.05),
           title="FODO (10 cells): transverse correlations (particle dumps only)")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(outdir / "03_correlation.png", dpi=180); plt.close(fig)


def make_phase_space_plot(h5: Path, steps, outdir: Path):
    first, last = steps[0], steps[-1]
    data = [load_snapshot(h5, first[0]), load_snapshot(h5, last[0])]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for col, (dump, bunch) in enumerate(zip((first, last), data)):
        if np.any(np.abs(bunch["pz"]) < 1e-12):
            raise ValueError("Cannot form particle slopes: pz is zero for at least one particle")
        # OPALX particle momenta are normalized mechanical momenta p/(mc).
        # Ratios px/pz and py/pz give paraxial x'=dx/dz, y'=dy/dz.
        for row, (coord, momentum) in enumerate((("x", "px"), ("y", "py"))):
            pos_mm = 1e3 * bunch[coord]
            slope_mrad = 1e3 * bunch[momentum] / bunch["pz"]
            ax = axs[row, col]
            ax.scatter(pos_mm, slope_mrad, s=1.5, alpha=0.25, rasterized=True,
                       color="tab:blue" if coord == "x" else "tab:orange")
            ax.set(xlabel=f"{coord} [mm]", ylabel=f"{coord}' [mrad]",
                   title=f"{coord.upper()} phase space — {'first' if col == 0 else 'last'}\n"
                         f"{step_description(h5, dump[0], dump[1])}")
            ax.title.set_fontsize(10)
            ax.grid(alpha=0.25)
    # Use identical axis limits for first/last dumps in each plane.
    for row, (coordinate, momentum) in enumerate((("x", "px"), ("y", "py"))):
        xx = np.concatenate([1e3 * d[coordinate] for d in data])
        yy = np.concatenate([1e3 * d[momentum] / d["pz"] for d in data])
        for values, setter in ((xx, "set_xlim"), (yy, "set_ylim")):
            low, high = float(np.min(values)), float(np.max(values))
            pad = 0.05 * (high - low) if high > low else 0.5
            for ax in axs[row, :]:
                getattr(ax, setter)(low - pad, high + pad)
    fig.savefig(outdir / "phase space plots.png", dpi=180)
    plt.close(fig)
    print(f"Phase-space dumps: first='{first[0]}', last='{last[0]}'" )
    if first == last:
        print("Warning: only one particle dump found: first and last are identical.")




def make_loss_plots(stats: dict[str, np.ndarray], outdir: Path,
                    cell_length: float, n_cells: int,
                    initial_particles: int | None) -> tuple[float, float]:
    """Plot survival and discrete particle losses using .stat numParticles.

    The observed maximum serves as the initial count unless --initial-particles
    supplies the known count, which detects losses before the first stat dump.
    Drops are attributed to the path length at the next saved statistics row;
    timing is therefore only resolved to STATDUMPFREQ.
    """
    spos = get_column(stats, "s")
    n = get_column(stats, "numparticles")
    if not np.all(np.isfinite(n)) or np.any(n < 0):
        raise ValueError("Invalid numParticles values in .stat")
    if np.any(np.diff(spos) < -1e-9):
        raise ValueError("Statistics path length decreases; separate restarts/runs first")
    if np.any(np.diff(n) > 0.5):
        print("WARNING: Particle counts increase at least once. Check for late emission,"
              " injection, restarts, or combined runs; loss attribution may be invalid.")

    n0 = float(initial_particles if initial_particles is not None else np.max(n))
    if n0 <= 0:
        raise ValueError("Initial particle count must be positive")
    if np.max(n) > n0 + 0.5:
        raise ValueError("Observed particle count exceeds --initial-particles")
    lost = np.maximum(0., n0 - n)
    survival_pct = 100. * n / n0
    lost_pct = 100. * lost / n0

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.step(spos, survival_pct, where="post", color="tab:blue",
            label="Surviving macroparticles")
    ax.set(xlabel="Path length s [m]", ylabel="Survival fraction [%]",
           title="FODO (10 cells): particle survival",
           xlim=(0, n_cells*cell_length))
    ax.set_ylim(max(0., float(np.min(survival_pct)) - 2.), 101.)
    for edge in np.arange(0, n_cells + 1) * cell_length:
        ax.axvline(edge, color="0.75", ls=":", lw=0.7)
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(outdir / "05_particle_survival.png", dpi=180)
    plt.close(fig)

    # Attribute each drop to the cell containing its first observed occurrence.
    # The initial record also receives any loss prior to the first stats dump.
    previous = np.concatenate(([n0], n[:-1]))
    drops = np.maximum(0.0, previous - n)
    # s exactly on a cell boundary is counted in the preceding cell.
    cell_idx = np.ceil(np.maximum(spos, 0.) / cell_length).astype(int) - 1
    cell_idx = np.clip(cell_idx, 0, n_cells - 1)
    loss_per_cell = np.bincount(cell_idx, weights=drops, minlength=n_cells)[:n_cells]
    indexes = np.arange(1, n_cells + 1)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(indexes, loss_per_cell, color="tab:orange")
    ax.set(xlabel="FODO cell", ylabel="Lost macroparticles",
           title="FODO (10 cells): particle losses per cell",
           xticks=indexes)
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(outdir / "06_particle_losses_per_cell.png", dpi=180)
    plt.close(fig)

    csv_lines = ["cell,s_start_m,s_end_m,lost_macroparticles"]
    for i, count in enumerate(loss_per_cell, start=1):
        csv_lines.append(f"{i},{(i-1)*cell_length:.6g},{i*cell_length:.6g},{count:.0f}")
    (outdir / "particle_losses_per_cell.csv").write_text("\n".join(csv_lines)+"\n")

    print(f"Initial macroparticles (reference): {n0:.0f}")
    print(f"Surviving at last statistics dump: {n[-1]:.0f}")
    print(f"Total lost: {lost[-1]:.0f} ({lost_pct[-1]:.4f}%)")
    print(f"Last statistics position: s={spos[-1]:.4f} m "
          f"(target {n_cells*cell_length:g} m)")
    if spos[-1] < n_cells * cell_length - 0.1:
        print("WARNING: Run appears to stop before the end of the target lattice")
    return n0, float(n[-1])


def write_observable_summary(stats: dict[str, np.ndarray], outdir: Path,
                             initial_particles: float, final_particles: float):
    s = get_column(stats, "s")
    text = ["OPALX 10-cell FODO observables summary", "=" * 42,
            "Values use the first and last available .stat rows, not necessarily s=0 and s=20 m.",
            f"Path length: {s[0]:.6g} -> {s[-1]:.6g} m",
            f"Reference macroparticles: {initial_particles:.0f}",
            f"Final macroparticles: {final_particles:.0f}",
            f"Lost: {initial_particles - final_particles:.0f} "
            f"({100*(initial_particles-final_particles)/initial_particles:.6g}%)", ""]
    for key, label, factor, unit in [
        ("emit_x", "Normalized rms emittance x", 1e6, "mm mrad"),
        ("emit_y", "Normalized rms emittance y", 1e6, "mm mrad"),
        ("rms_x", "RMS envelope x", 1e3, "mm"),
        ("rms_y", "RMS envelope y", 1e3, "mm"),
    ]:
        a = get_column(stats, key) * factor
        change = 100 * (a[-1] / a[0] - 1) if a[0] != 0 else float("nan")
        text.append(f"{label}: {a[0]:.6g} -> {a[-1]:.6g} {unit} "
                    f"({change:+.3f}%)")
    try:
        cx, cy = correlations(stats)
        text += [f"rho(x,px): {cx[0]:+.6g} -> {cx[-1]:+.6g}",
                 f"rho(y,py): {cy[0]:+.6g} -> {cy[-1]:+.6g}"]
    except (KeyError, ValueError):
        text.append("Correlations not available in .stat")
    text += ["", "Scientific interpretation:",
             "- Counts measure macroparticle deletion, not physical beam intensity losses unless",
             "  the deletion mechanism represents a specified physical aperture or interaction.",
             "- If losses occur, the displayed emittance/envelopes describe survivors only.",
             "- The current generator defaults to no aperture and disables BOUNDPDESTROY.",
             "- Verify the tracking log for unexpected deletion or premature termination."]
    (outdir / "observables_summary.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text[:11]))

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", default=".", type=Path,
                        help="Directory containing OPALX output files, possibly in data/")
    parser.add_argument("--stat", help="Explicit .stat file (path or relative to directory)")
    parser.add_argument("--h5", help="Explicit phase-space .h5 file")
    parser.add_argument("--outdir", type=Path, help="Plot output directory (default: DIRECTORY/plots)")
    parser.add_argument("--initial-particles", type=int, default=None,
                        help="Known initial macroparticle count (e.g. 10000). "
                             "Default: maximum observed numParticles in .stat")
    parser.add_argument("--cell-length", type=float, default=2.0,
                        help="FODO cell length in metres (default: 2.0)")
    parser.add_argument("--cells", type=int, default=10,
                        help="Number of FODO cells (default: 10)")
    args = parser.parse_args()
    if args.cells < 1 or args.cell_length <= 0 or (args.initial_particles is not None and args.initial_particles < 1):
        parser.error("--cells, --cell-length and --initial-particles must be positive")
    root = args.directory.resolve()
    outdir = (args.outdir or root / "plots").resolve()
    stat, h5 = choose_files(root, args.stat, args.h5)
    print(f"Statistics: {stat}")
    print(f"Phase space: {h5}")
    stats = parse_stat(stat)
    for col in REQUIRED_STAT:
        get_column(stats, col)
    steps = get_particle_steps(h5)
    if not steps:
        raise ValueError(f"No particle dumps found in {h5}")
    outdir.mkdir(parents=True, exist_ok=True)
    if not make_evolution_plots(stats, outdir):
        make_h5_correlation_plot(h5, steps, outdir)
    make_phase_space_plot(h5, steps, outdir)
    initial, final = make_loss_plots(stats, outdir, args.cell_length,
                                     args.cells, args.initial_particles)
    write_observable_summary(stats, outdir, initial, final)
    print(f"Saved 6 figures and observables summary in: {outdir}")


if __name__ == "__main__":
    main()
