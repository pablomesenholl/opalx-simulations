"""Prepare a full-population, one-cell space-charge feasibility case.

The case uses the ordinary collective boundary controller with no experimental
subset environment variable. It is intentionally a preparation-only runner:
the generated input, particle file, binary and source snapshot are immutable so
the subsequent timeout is an interpretable feasibility measurement.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from large_ellipsoid_benchmark import generate_ellipsoid, sha256, write_json, write_particles
from rf_ring_study import ROOT, cyclic_d1_lattice, tracking_deck

REPO = ROOT.parents[2]

COUNT = 131072
CHARGE = 160e-12
DT = 62.5e-12
MESH = 32
SEED = 20260911


def prepare(binary, output=None):
    """Freeze a one-cell case and return its directory.

    The DBA has six identical cells. The reference starts at the middle of
    C1_D1, so one cell is circumference/6. DT is rounded to the nearest whole
    number of nominal steps; the manifest records the resulting endpoint and
    spatial mismatch explicitly.
    """
    binary = Path(binary).resolve()
    study = Path(output) if output else Path(tempfile.mkdtemp(
        prefix="full-population-one-cell-", dir=ROOT / "results"))
    study.mkdir(parents=True, exist_ok=True)
    if (study / "manifest.json").exists():
        raise ValueError(f"Study already exists: {study}")

    calibration = json.loads((ROOT / "results/rf-ring-s3a01yyz/bare-cof.json").read_text())
    sections = json.loads((ROOT / "results/rf-ring-s3a01yyz/sections.json").read_text())
    model = np.load(ROOT / "results/ellipsoid-match-stable-long-d1center/match-160pc.npz")
    covariance = model["covariance"]
    reference_p = np.asarray(calibration["momentum_mc"], dtype=float)
    p0 = float(np.linalg.norm(reference_p))
    gamma = float(np.sqrt(1.0 + p0 * p0))
    speed = 299792458.0 * p0 / gamma
    circumference = 25.766370614359175
    cell_length = circumference / 6.0
    exact_cell_time = cell_length / speed
    steps = int(round(exact_cell_time / DT))
    duration = steps * DT
    travelled = speed * duration

    base, cell_sections = cyclic_d1_lattice((ROOT / "bare-gaussian-cof.in").read_text())
    if not np.allclose(cell_sections[0]["centre_m"], calibration["position_m"], atol=2e-14):
        raise ValueError("COF section does not match C1_D1 centre")
    # tracking_deck appends the distribution/beam/run declarations. Remove the
    # old bare input's distribution and COF declarations first.
    lattice = base.split("DIST: DISTRIBUTION", 1)[0]
    lattice = re.sub(r"PSDUMPFREQ=\d+", "PSDUMPFREQ=1000000000", lattice)
    lattice = re.sub(r"STATDUMPFREQ=\d+", "STATDUMPFREQ=1000000000", lattice)
    lattice = re.sub(r"STEPINFOFQ=\d+", "STEPINFOFQ=100", lattice)
    deck = tracking_deck(lattice, COUNT, DT, solver="OPEN", charge=CHARGE,
                         mesh=MESH, steps=steps)

    (study / "source").mkdir()
    changed = subprocess.check_output(
        ["git", "ls-files", "--modified", "--others", "--exclude-standard", "--",
         "src", "unit_tests"], cwd=REPO, text=True).splitlines()
    for relative in dict.fromkeys(changed):
        source = REPO / relative
        if source.is_file():
            target = study / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (study / "source" / "working-tree.diff").write_text(
        subprocess.check_output(["git", "diff"], cwd=REPO, text=True))
    (study / "source" / "analysis").mkdir()
    for source in (ROOT / "analysis").glob("*.py"):
        shutil.copy2(source, study / "source" / "analysis" / source.name)

    phase = generate_ellipsoid(covariance, COUNT, seed=SEED)
    particles = np.asarray(phase[:, [0, 2, 4]])
    theta = float(sections[0]["theta_rad"])
    rotation = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(theta), 0.0, np.cos(theta)]])
    positions = particles @ rotation.T + np.asarray(calibration["position_m"])
    momenta = p0 * phase[:, [1, 3, 5]] @ rotation.T + reference_p
    absolute = np.column_stack((positions, momenta))
    particle_file = study / "particles.txt"
    write_particles(particle_file, absolute)
    (study / "case.in").write_text(deck)
    executable = study / "opalx"
    shutil.copy2(binary, executable)
    np.savez(study / "initial.npz", phase=phase, covariance=covariance,
             ids=np.arange(COUNT, dtype=np.int64))

    manifest = dict(
        status="prepared", binary=str(executable), binary_sha256=sha256(executable),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                         text=True).strip(),
        count=COUNT, charge_C=CHARGE, mesh=MESH, dt_s=DT, mpi_ranks=1,
        omp_threads=8, seed=SEED, circumference_m=circumference,
        cell_length_m=cell_length, exact_cell_time_s=exact_cell_time,
        steps=steps, duration_s=duration, travelled_length_m=travelled,
        endpoint_length_error_m=travelled-cell_length,
        start_section="C1_D1 centre", end_section="C2_D1 centre (one cell)",
        boundary_control="full population (environment unset; stride=1)",
        solver="OPEN", deck_sha256=sha256(study / "case.in"),
        particles_sha256=sha256(particle_file), initial_npz_sha256=sha256(study / "initial.npz"),
        source_sha256={str(path.relative_to(study)): sha256(path)
                       for path in (study / "source").rglob("*") if path.is_file()},
        note="Preparation only. Run with OMP_NUM_THREADS=8 and no OPALX_TEST_* environment variable; use a task-owned timeout.",
    )
    write_json(study / "manifest.json", manifest)
    print(study)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=REPO / "omp-build/src/opalx")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prepare(args.binary, args.output)
