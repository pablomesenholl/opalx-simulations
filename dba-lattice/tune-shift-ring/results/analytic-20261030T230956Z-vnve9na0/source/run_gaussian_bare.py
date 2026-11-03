"""Run the single-input COF/GAUSS bare-tune benchmark in a fresh directory.

Lengths and timestep arguments use SI units. COF and TRACK deliberately use the
same requested timestep. Examples (use ~/.venv-h6/bin/python):
  run_gaussian_bare.py
  run_gaussian_bare.py --particles 1 --turns 1 --sigma-xy 1e-12 --sigma-z 1e-12
  run_gaussian_bare.py --dt 3.125e-11 --prepare-only
  run_gaussian_bare.py --self-test
"""

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
ELEMENTARY_CHARGE_C = Decimal("1.602176634e-19")


def replace_once(text, pattern, replacement):
    """Require one unambiguous match for a targeted input edit."""
    result, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one match, found {count}: {pattern}")
    return result


def edit_command(text, command_pattern, attributes):
    """Edit only named attributes within exactly one complete OPAL command."""
    def edit(match):
        command = match.group()
        for name, value in attributes.items():
            command = replace_once(command, rf"\b{name}\s*=\s*[^,;]+", f"{name}={value}")
        return command
    return replace_once(text, command_pattern, edit)


def render_deck(base, args):
    """Preserve the lattice and launch model while applying explicit run controls."""
    fs = re.search(r"^FS:\s*FIELDSOLVER\b[^;]*;", base, re.MULTILINE)
    if fs is None or not re.search(r"\bTYPE\s*=\s*NONE\b", fs.group()):
        raise ValueError("The bare-tune base must use FIELDSOLVER TYPE=NONE")
    text = edit_command(base, r"^DIST:\s*DISTRIBUTION\b[^;]*;", {
        "SIGMAX": f"{args.sigma_xy:.17g}", "SIGMAY": f"{args.sigma_xy:.17g}",
        "SIGMAZ": f"{args.sigma_z:.17g}", "NPARTDIST": args.particles,
    })
    text = edit_command(text, r"^BEAM1:\s*BEAM\b[^;]*;", {
        "NALLOC": args.particles, "BCHARGE": str(ELEMENTARY_CHARGE_C * args.particles),
    })
    for command in (r"^ic:\s*COF\b[^;]*;", r"^TRACK\b[^;]*;"):
        text = edit_command(text, command, {"DT": f"{args.dt:.17g}"})
    text = edit_command(text, r"^RUN\b[^;]*;", {"TURNS": args.turns})
    # A later OPTION overrides an earlier value without changing dump cadence.
    text = replace_once(text, r"^TITLE,", "OPTION, STEPINFOFQ=10000, REPARTFREQ=0;\nTITLE,")
    text = replace_once(text, r"^// sigma_x=.*$",
                        f"// sigma_x=sigma_y={args.sigma_xy:.17g} m; three-sigma cutoff; cold transverse launch.")
    text = replace_once(text, r"^// sigma_z=.*$",
                        f"// sigma_z={args.sigma_z:.17g} m; FSNONE bare-tune launch.")
    return text


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_output(repo, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=repo, text=True)


def source_paths(repo):
    """Include every changed/untracked source or test, plus core numerical helpers."""
    paths = set(git_output(repo, "diff", "--name-only", "-z", "HEAD", "--",
                           "src", "unit_tests").split("\0"))
    paths.update(git_output(repo, "ls-files", "--others", "--exclude-standard", "-z",
                            "--", "src", "unit_tests").split("\0"))
    for pattern in ("src/Algorithms/Boris*", "src/Algorithms/Quaternion*",
                    "src/Algorithms/DeviceExternalField*", "src/PartBunch/Binning/AdaptBins.*"):
        paths.update(str(path.relative_to(repo)) for path in repo.glob(pattern) if path.is_file())
    paths.discard("")
    return sorted(paths)


def snapshot_sources(repo, run):
    """Capture changed file bytes, including additions absent from git diff."""
    hashes, deleted = {}, []
    for relative in source_paths(repo):
        source = repo / relative
        if not source.is_file():
            deleted.append(relative)
            continue
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = sha256(target)
    (run / "source.diff").write_text(git_output(repo, "diff", "--binary", "HEAD"))
    (run / "git-status.txt").write_text(git_output(repo, "status", "--short"))
    return hashes, deleted


def write_metadata(run, metadata):
    (run / "provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def prepare(args):
    """Create a fresh input and record the actual requested settings and source."""
    repo = Path(git_output(ROOT, "rev-parse", "--show-toplevel").strip())
    binary = repo / "omp-build/src/opalx"
    base = ROOT / "bare-gaussian-cof.in"
    text = render_deck(base.read_text(), args)
    (ROOT / "results").mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"gaussian-bare-{args.dt * 1e12:g}ps-", dir=ROOT / "results"))
    (run / "case.in").write_text(text)
    for source, name in ((base, "base.in"), (ROOT / "config.json", "config.json"),
                         (Path(__file__), "run_gaussian_bare.py")):
        shutil.copy2(source, run / name)
    source_hashes, deleted = snapshot_sources(repo, run)
    metadata = dict(
        dt_s=args.dt, cof_dt_s=args.dt, requested_turns=args.turns,
        particle_count=args.particles, allocation_count=args.particles,
        physical_proton_count=args.particles,
        bunch_charge_C=float(ELEMENTARY_CHARGE_C * args.particles),
        distribution="GAUSS", sigma_x_m=args.sigma_xy, sigma_y_m=args.sigma_xy,
        sigma_z_m=args.sigma_z, cutoff_sigma=3, momentum_spreads=[0, 0, 0],
        minimum_analysis_amplitude_m=2e-6, initialorbit="ic", seed=20260910,
        nominal_period_s=1.179718718127957e-6, mpi_ranks=1, omp_threads=1,
        backend="Kokkos OpenMP", fields_solver="NONE", step_info_frequency=10000,
        repartition_frequency=0,
        binary=str(binary), binary_sha256=sha256(binary),
        revision=git_output(repo, "rev-parse", "HEAD").strip(),
        source_sha256=source_hashes, deleted_source_files=deleted,
        base_input_sha256=sha256(base), input_sha256=sha256(run / "case.in"),
        script_sha256=sha256(run / "run_gaussian_bare.py"),
        arguments=vars(args), invocation=[sys.executable, *sys.argv],
        command=[str(binary), "--info", "1", "case.in"],
        completed=False, status="prepared",
    )
    write_metadata(run, metadata)
    return run, metadata


def run_case(run, metadata):
    """Run one rank with one OpenMP thread and retain failure provenance."""
    if sha256(metadata["binary"]) != metadata["binary_sha256"]:
        raise RuntimeError("The binary changed after preparation")
    start = time.monotonic()
    metadata["status"] = "running"
    write_metadata(run, metadata)
    try:
        with (run / "run.log").open("w") as log:
            result = subprocess.run(metadata["command"], cwd=run,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_PROC_BIND": "false"},
                stdout=log, stderr=subprocess.STDOUT, check=False)
        metadata.update(returncode=result.returncode, completed=result.returncode == 0,
                        status="completed" if result.returncode == 0 else "failed")
        if metadata["completed"]:
            log_text = (run / "run.log").read_text()
            trials = re.findall(r"Boris boundary trials=(\d+), rejected=(\d+)", log_text)
            enabled = "Ring boundary control: synchronized ordinary Boris/PIC substeps." in log_text
            metadata.update(boundary_control_enabled=enabled,
                            boundary_trials=int(trials[-1][0]) if trials else 0,
                            boundary_rejected=int(trials[-1][1]) if trials else 0)
            metadata["controller_validation_passed"] = enabled and metadata["boundary_trials"] > 0
    except BaseException:
        metadata["status"] = "interrupted_or_failed"
        raise
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - start
        write_metadata(run, metadata)
    if not metadata["completed"]:
        raise RuntimeError(f"OPALX failed; see {run / 'run.log'}")
    if not metadata["controller_validation_passed"]:
        raise RuntimeError(f"Run did not exercise ordinary Boris/PIC boundary control; see {run / 'run.log'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--particles", type=int, default=32, help="Number of physical protons (default: 32)")
    parser.add_argument("--turns", type=int, default=32, help="Requested turns (default: 32)")
    parser.add_argument("--dt", type=float, default=6.25e-11, help="COF and TRACK timestep in seconds (default: 62.5 ps)")
    parser.add_argument("--sigma-xy", type=float, default=50e-6 / 3, help="RMS x/y size in metres (default: 50/3 um)")
    parser.add_argument("--sigma-z", type=float, default=1e-12, help="RMS z size in metres (default: 1 pm)")
    parser.add_argument("--prepare-only", action="store_true", help="Write a fresh input and provenance without running OPALX")
    parser.add_argument("--self-test", action="store_true", help="Test input rendering and provenance helpers without OPALX")
    args = parser.parse_args(argv)
    if args.particles < 1 or args.turns < 1:
        parser.error("--particles and --turns must be positive integers")
    if not math.isfinite(args.dt) or args.dt <= 0:
        parser.error("--dt must be finite and positive")
    if any(not math.isfinite(value) or value <= 0 for value in (args.sigma_xy, args.sigma_z)):
        parser.error("--sigma-xy and --sigma-z must be finite and positive; use 1e-12 for a numerical point launch")
    return args


class ScriptTests(unittest.TestCase):
    def test_defaults_and_targeted_single_particle_edits(self):
        base = (ROOT / "bare-gaussian-cof.in").read_text()
        for args in (parse_args([]), parse_args(["--particles", "1", "--turns", "1",
                                               "--dt", "3.125e-11", "--sigma-xy", "1e-12", "--sigma-z", "1e-12"])):
            text = render_deck(base, args)
            self.assertEqual(re.findall(r"\bDT=([^,;]+)", text), [f"{args.dt:.17g}"] * 2)
            self.assertEqual(re.findall(r"\bMAXSTEPS=([^,;]+)", text), ["200000", "200000"])
            self.assertIn(f"NPARTDIST={args.particles}", text)
            self.assertIn(f"NALLOC={args.particles}", text)
            charge = re.search(r"\bBCHARGE=([^,;]+)", text).group(1)
            self.assertEqual(Decimal(charge), ELEMENTARY_CHARGE_C * args.particles)
            self.assertIn(f"TURNS={args.turns}", text)
            self.assertIn("STEPINFOFQ=10000", text)
            self.assertIn("REPARTFREQ=0", text)
            self.assertIn("PSDUMPFREQ=197, STATDUMPFREQ=197", text)
            self.assertIn("INITIALORBIT=ic", text)
            self.assertIn("SIGMAPX=0, SIGMAPY=0, SIGMAPZ=0", text)
            self.assertIn(f"SIGMAX={args.sigma_xy:.17g}", text)
            self.assertIn(f"SIGMAY={args.sigma_xy:.17g}", text)
            self.assertIn(f"SIGMAZ={args.sigma_z:.17g}", text)
            lattice_lines = lambda deck: [line for line in deck.splitlines() if re.match(r"^(?:C\d_|LATTICE:)", line)]
            self.assertEqual(lattice_lines(text), lattice_lines(base))

    def test_ambiguous_deck_fails(self):
        with self.assertRaises(ValueError):
            edit_command("TRACK, DT=1;\nTRACK, DT=2;", r"^TRACK\b[^;]*;", {"DT": 3})

    def test_source_snapshot_has_modified_and_untracked_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            quaternion = repo / "src/Algorithms/Quaternion.cpp"
            quaternion.parent.mkdir(parents=True)
            quaternion.write_text("old\n")
            deleted = repo / "src/removed.cpp"
            deleted.write_text("old\n")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "commit", "-qm", "initial"], cwd=repo, check=True)
            quaternion.write_text("changed\n")
            deleted.unlink()
            for name in ("src/Algorithms/BorisMidpoint.h", "src/PartBunch/Binning/AdaptBins.tpp",
                         "src/Algorithms/OtherNewKernel.h", "unit_tests/TestKernel.cpp"):
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name + "\n")
            run = Path(directory) / "run"
            run.mkdir()
            hashes, removed = snapshot_sources(repo, run)
            self.assertEqual(len(hashes), 5)
            self.assertEqual(removed, ["src/removed.cpp"])
            for name, digest in hashes.items():
                self.assertEqual((run / "source" / name).read_bytes(), (repo / name).read_bytes())
                self.assertEqual(digest, sha256(repo / name))
            self.assertIn("changed", (run / "source.diff").read_text())


def main():
    args = parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScriptTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    run, metadata = prepare(args)
    print(run, flush=True)
    if not args.prepare_only:
        run_case(run, metadata)
    print(json.dumps(metadata), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
