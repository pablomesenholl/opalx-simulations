"""Prepare/run a short, reproducible ordinary-Boris/PIC coupling check.

This is a cold Gaussian launch test, not a matched beam or a tune-shift model.
The five cases end at 0.625 ns: bare/q1/q2 use 62.5 ps and ten nominal steps;
bare_fine/q1_fine use 31.25 ps and twenty. COF settings stay identical across cases.

Examples (use ~/.venv-h6/bin/python):
  check_boris_pic.py --prepare-only
  check_boris_pic.py --binning one --prepare-only
  check_boris_pic.py --study PATH --run all --analyze
  check_boris_pic.py --study PATH --analyze
  check_boris_pic.py --self-test

The timestep comparison subtracts the bare result separately at each resolution
to isolate the self-field response. Two resolutions measure sensitivity; they
do not establish an asymptotic convergence order. Default 5% charge-scaling and
10% timestep gates are smoke-test criteria, not precision tune tolerances.
Use --binning one for one fixed GAMMAZ bin and relativistic self E/B composition;
the default none uses the existing unbinned electrostatic gather. Both modes
disable load balancing and bin file/table diagnostics.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import unittest

import h5py
import numpy as np
import pandas as pd


CASES = {
    "bare": {"solver": "NONE", "charge_C": 1e-18, "dt_s": 6.25e-11, "steps": 10},
    "q1": {"solver": "OPEN", "charge_C": 1e-18, "dt_s": 6.25e-11, "steps": 10},
    "q2": {"solver": "OPEN", "charge_C": 2e-18, "dt_s": 6.25e-11, "steps": 10},
    "bare_fine": {"solver": "NONE", "charge_C": 1e-18, "dt_s": 3.125e-11, "steps": 20},
    "q1_fine": {"solver": "OPEN", "charge_C": 1e-18, "dt_s": 3.125e-11, "steps": 20},
}
COMPONENTS = ["x", "y", "z", "px", "py", "pz", "Ex", "Ey", "Ez", "Bx", "By", "Bz"]
SOURCE_FILES = [
    "src/Algorithms/BorisMidpoint.h", "src/Algorithms/BorisStepControl.h",
    "src/Algorithms/DeviceExternalField.h", "src/Algorithms/DeviceExternalField.cpp",
    "src/Algorithms/ParallelTracker.cpp", "src/Algorithms/ParallelTracker.h",
    "src/Algorithms/TrackReferenceStep.cpp", "src/Algorithms/TrackReferenceStep.h",
    "src/AbsBeamline/SBend.cpp", "src/AbsBeamline/RBend.cpp",
    "src/AbsBeamline/Multipole.cpp", "src/Track/TrackRun.cpp",
    "src/PartBunch/Binning/AdaptBins.h", "src/PartBunch/Binning/AdaptBins.tpp",
]


def replace_once(text, pattern, replacement):
    """Reject ambiguous deck edits instead of silently changing unrelated commands."""
    result, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one match, found {count}: {pattern}")
    return result


def render_deck(base, case, binning="none"):
    """Change the generated bunch and TRACK controls while preserving COF verbatim."""
    original_cof = re.search(r"^ic:\s*COF\b[^;]*;", base, re.MULTILINE).group()
    text = replace_once(base, r"^// sigma_z=.*$", "// Finite 1 mm RMS bunch; cold, unmatched launch test.")
    if binning not in ("none", "one"):
        raise ValueError(f"Unsupported binning mode: {binning}")
    text = replace_once(text, r"^TITLE,", "OPTION, EBDUMP=TRUE, PSDUMPFREQ=1, STATDUMPFREQ=1, REPARTFREQ=0;\nTITLE,")
    text = replace_once(text, r"^DIST:\s*DISTRIBUTION\b[^;]*;",
        "DIST: DISTRIBUTION, TYPE=GAUSS, SIGMAX=50e-6/3, SIGMAY=50e-6/3, "
        "CUTOFFX=3, CUTOFFY=3, SIGMAZ=1e-3, SIGMAPX=0, SIGMAPY=0, SIGMAPZ=0, NPARTDIST=32;")
    bin_definition = ('ONE: BINNING, PARAMETER="GAMMAZ", MAXBINS=1, ADAPTIVEBINNING=FALSE, '
                      'DUMPBINSFILE="NONE", TABLEPRINTFREQ=0;\n') if binning == "one" else ""
    bin_name = "ONE" if binning == "one" else "NONE"
    text = replace_once(text, r"^FS:\s*FIELDSOLVER\b[^;]*;", bin_definition +
        f"FS: FIELDSOLVER, TYPE={case['solver']}, BINS={bin_name}, NX=16, NY=16, NZ=16, "
        "PARFFTX=TRUE, PARFFTY=TRUE, PARFFTZ=TRUE, BCFFTX=OPEN, BCFFTY=OPEN, BCFFTZ=OPEN;")
    beam = re.search(r"^BEAM1:\s*BEAM\b[^;]*;", text, re.MULTILINE).group()
    beam = replace_once(beam, r"\bBCHARGE\s*=\s*[^,;]+", f"BCHARGE={case['charge_C']:.17g}")
    text = replace_once(text, r"^BEAM1:\s*BEAM\b[^;]*;", lambda _: beam)
    track = re.search(r"^TRACK\b[^;]*;", text, re.MULTILINE).group()
    track = replace_once(track, r"\bDT\s*=\s*[^,;]+", f"DT={case['dt_s']:.17g}")
    track = replace_once(track, r"\bMAXSTEPS\s*=\s*[^,;]+", f"MAXSTEPS={case['steps']}")
    text = replace_once(text, r"^TRACK\b[^;]*;", lambda _: track)
    text = replace_once(text, r"^RUN\b[^;]*;", 'RUN, METHOD="PARALLEL", FIELDSOLVER=FS;')
    if re.search(r"^ic:\s*COF\b[^;]*;", text, re.MULTILINE).group() != original_cof:
        raise ValueError("COF command changed unexpectedly")
    return text


def sha256(path):
    """Hash the actual artifact bytes without loading a large executable at once."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Write strict JSON so a nonfinite result cannot masquerade as valid data."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(base_path, binary, binning="none"):
    """Create a fresh study with five independent inputs and source provenance."""
    root = base_path.parent
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root, text=True).strip())
    (root / "results").mkdir(exist_ok=True)
    study = Path(tempfile.mkdtemp(prefix="boris-pic-", dir=root / "results"))
    shutil.copy2(__file__, study / "check_boris_pic.py")
    shutil.copy2(base_path, study / "base.in")
    (study / "source.diff").write_text(subprocess.check_output(["git", "diff"], cwd=repo, text=True))
    source_hashes = {}
    for relative in SOURCE_FILES:
        source = repo / relative
        if source.exists():
            target = study / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_hashes[relative] = sha256(source)
    metadata = {
        "repository": str(repo), "base_input": str(base_path), "binary": str(binary),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_sha256": source_hashes, "script_sha256": sha256(__file__),
        "seed": 20260910, "particle_count": 32, "sigma_xy_m": 50e-6 / 3,
        "sigma_z_m": 1e-3, "momentum_spreads": [0, 0, 0],
        "mesh": [16, 16, 16], "binning": binning, "repartition_frequency": 0,
        "bin_file_diagnostics": False, "bin_table_frequency": 0, "mpi_ranks": 1, "omp_threads": 1,
        "expected_end_time_s": 6.25e-10, "uses_turns": False,
        "scope": "Cold Gaussian short-step coupling check; no matched-beam or analytic tune claim.",
    }
    write_json(study / "study.json", metadata)
    for name, case in CASES.items():
        directory = study / name
        directory.mkdir()
        (directory / "case.in").write_text(render_deck(base_path.read_text(), case, binning))
        write_json(directory / "provenance.json", {
            **case, "case": name, "binning": binning, "status": "prepared", "completed": False,
            "input_sha256": sha256(directory / "case.in"),
        })
    return study


def run_case(study, name, binary):
    """Run exactly one MPI rank, preserving the prepared input and separate log."""
    directory = study / name
    metadata = json.loads((directory / "provenance.json").read_text())
    if metadata["status"] != "prepared":
        raise ValueError(f"{name} has status {metadata['status']}; prepare a fresh study to rerun")
    if sha256(directory / "case.in") != metadata["input_sha256"]:
        raise ValueError(f"{name}: prepared input was edited")
    metadata.update(status="running", binary=str(binary), binary_sha256=sha256(binary),
                    mpi_ranks=1, omp_threads=1, command=[str(binary), "--info", "1", "case.in"])
    write_json(directory / "provenance.json", metadata)
    started = time.monotonic()
    try:
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(metadata["command"], cwd=directory,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_PROC_BIND": "false"},
                stdout=log, stderr=subprocess.STDOUT, check=False)
        metadata.update(returncode=result.returncode, completed=result.returncode == 0,
                        status="completed" if result.returncode == 0 else "failed")
    except BaseException:
        metadata["status"] = "interrupted_or_failed"
        raise
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - started
        write_json(directory / "provenance.json", metadata)
    if not metadata["completed"]:
        raise RuntimeError(f"{name} failed; see {directory / 'run.log'}")
    return extract_final(directory)


def laboratory_vectors(values, reference_p):
    """Rotate this planar DBA's reference-frame vectors into fixed laboratory axes."""
    ez = np.asarray(reference_p, dtype=float)
    ez = ez / np.linalg.norm(ez)
    if abs(ez[1]) > 1e-12:
        raise ValueError("Expected a planar DBA reference frame")
    rotation = np.column_stack(([ez[2], 0., -ez[0]], [0., 1., 0.], ez))
    return np.asarray(values) @ rotation.T


def extract_final(directory):
    """Export ID-sorted final particles and field/finite-value diagnostics."""
    metadata = json.loads((directory / "provenance.json").read_text())
    if not metadata.get("completed"):
        raise ValueError(f"{directory.name} is not completed")
    with h5py.File(directory / "case.h5", "r") as output:
        keys = sorted((k for k in output if re.fullmatch(r"Step#\d+", k)), key=lambda k: int(k[5:]))
        if not keys:
            raise ValueError("No HDF5 particle snapshots")
        group = output[keys[-1]]
        missing = set(["id", *COMPONENTS]) - set(group)
        if missing:
            raise ValueError(f"Missing HDF5 datasets (EBDUMP is required): {sorted(missing)}")
        frame = pd.DataFrame({key: np.asarray(group[key]) for key in ["id", *COMPONENTS]})
        frame = frame.sort_values("id").reset_index(drop=True)
        ref_r, ref_p = np.asarray(group.attrs["RefPartR"]), np.asarray(group.attrs["RefPartP"])
        end_time = float(np.asarray(group.attrs["TIME"]).ravel()[0])
        global_step = int(np.asarray(group.attrs["GlobalTrackStep"]).ravel()[0])
    for source, prefix in [(["x", "y", "z"], "lab_r"), (["px", "py", "pz"], "lab_p"),
                           (["Ex", "Ey", "Ez"], "lab_E"), (["Bx", "By", "Bz"], "lab_B")]:
        vectors = laboratory_vectors(frame[source].to_numpy(), ref_p)
        if prefix == "lab_r":
            vectors += ref_r
        for index, axis in enumerate("xyz"):
            frame[f"{prefix}{axis}"] = vectors[:, index]
    frame.to_csv(directory / "final-particles.csv", index=False)
    finite = bool(np.isfinite(frame.to_numpy()).all())
    summary = {
        "time_s": end_time, "global_step": global_step, "snapshots": len(keys),
        "particle_count": len(frame), "unique_ids": bool(frame.id.is_unique), "finite": finite,
        "position_std_m": frame[["x", "y", "z"]].std(ddof=0).to_list(),
        "momentum_rms": np.sqrt(np.mean(frame[["px", "py", "pz"]].to_numpy() ** 2, axis=0)).tolist(),
        "electric_rms_V_per_m": np.sqrt(np.mean(frame[["Ex", "Ey", "Ez"]].to_numpy() ** 2, axis=0)).tolist(),
        "magnetic_rms_T": np.sqrt(np.mean(frame[["Bx", "By", "Bz"]].to_numpy() ** 2, axis=0)).tolist(),
        "lab_magnetic_rms_T": np.sqrt(np.mean(frame[["lab_Bx", "lab_By", "lab_Bz"]].to_numpy() ** 2, axis=0)).tolist(),
        "h5_sha256": sha256(directory / "case.h5"),
    }
    if not finite:
        raise ValueError(f"{directory.name}: nonfinite particle or field values")
    write_json(directory / "final.json", summary)
    return frame, summary


def response_metrics(delta1, delta2):
    """Measure charge-doubling proportionality using all three momentum components."""
    norm = np.linalg.norm(delta1)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("No finite, nonzero self-field momentum response")
    return {
        "response_rms": float(np.sqrt(np.mean(delta1 ** 2))),
        "response_rms_by_axis": np.sqrt(np.mean(delta1 ** 2, axis=0)).tolist(),
        "charge_doubling_projection": float(np.vdot(delta1, delta2) / np.vdot(delta1, delta1)),
        "charge_doubling_relative_residual": float(np.linalg.norm(delta2 - 2 * delta1) / (2 * norm)),
    }


def analyze(study, scaling_tolerance=0.05, timestep_tolerance=0.1):
    """Compare completed cases; partial studies are reported without a success claim."""
    frames, summaries, hashes = {}, {}, set()
    study_metadata = json.loads((study / "study.json").read_text())
    binning = study_metadata.get("binning", "none").lower()
    for name in CASES:
        metadata = json.loads((study / name / "provenance.json").read_text())
        if metadata.get("completed"):
            frames[name], summaries[name] = extract_final(study / name)
            hashes.add(metadata["binary_sha256"])
    report = {"complete": len(frames) == len(CASES), "binning": binning, "cases": summaries, "checks": {},
              "scaling_tolerance": scaling_tolerance, "timestep_tolerance": timestep_tolerance,
              "limitation": "Bare-subtracted responses at two resolutions test timestep sensitivity, not asymptotic convergence order."}
    checks = report["checks"]
    checks["same_binary"] = len(hashes) <= 1
    for name, summary in summaries.items():
        checks[f"{name}_32_unique_particles"] = summary["particle_count"] == 32 and summary["unique_ids"]
        checks[f"{name}_end_time"] = bool(np.isclose(summary["time_s"], 6.25e-10, rtol=1e-10, atol=1e-22))
        checks[f"{name}_finite_nonzero_xyz_spread"] = summary["finite"] and all(v > 0 for v in summary["position_std_m"])
        checks[f"{name}_nonzero_momentum"] = any(v > 0 for v in summary["momentum_rms"])
        if not name.startswith("bare"):
            checks[f"{name}_nonzero_self_electric_field"] = all(v > 0 for v in summary["electric_rms_V_per_m"])
            if binning == "one":
                # During this short launch only vertical analytic dipole fields
                # are present; lab Bx/Bz can therefore reveal self-field composition.
                checks[f"{name}_nonzero_self_magnetic_field"] = any(summary["lab_magnetic_rms_T"][i] > 0 for i in (0, 2))
        else:
            checks[f"{name}_zero_electric_field"] = all(v == 0 for v in summary["electric_rms_V_per_m"])
            if binning == "one":
                checks[f"{name}_zero_transverse_magnetic_field"] = all(summary["lab_magnetic_rms_T"][i] == 0 for i in (0, 2))
    if {"bare", "q1", "q2"} <= frames.keys():
        for name, frame in frames.items():
            if not np.array_equal(frame.id, frames["bare"].id):
                raise ValueError(f"Particle IDs differ in {name}")
        columns = ["lab_px", "lab_py", "lab_pz"]
        baseline = frames["bare"][columns].to_numpy()
        delta1 = frames["q1"][columns].to_numpy() - baseline
        delta2 = frames["q2"][columns].to_numpy() - baseline
        report["response"] = response_metrics(delta1, delta2)
        checks["charge_doubling"] = report["response"]["charge_doubling_relative_residual"] < scaling_tolerance
        response = pd.DataFrame({"id": frames["bare"].id})
        for index, axis in enumerate("xyz"):
            response[f"delta_p1_{axis}"] = delta1[:, index]
            response[f"delta_p2_{axis}"] = delta2[:, index]
        if {"bare_fine", "q1_fine"} <= frames.keys():
            delta_fine = frames["q1_fine"][columns].to_numpy() - frames["bare_fine"][columns].to_numpy()
            sensitivity = delta_fine - delta1
            relative = float(np.linalg.norm(sensitivity) / np.linalg.norm(delta1))
            report["self_field_timestep_relative_sensitivity"] = relative
            report["fine_response_rms_by_axis"] = np.sqrt(np.mean(delta_fine ** 2, axis=0)).tolist()
            checks["self_field_timestep_sensitivity"] = relative < timestep_tolerance
            for index, axis in enumerate("xyz"):
                response[f"delta_p1_fine_{axis}"] = delta_fine[:, index]
                response[f"fine_minus_coarse_delta_p_{axis}"] = sensitivity[:, index]
        response.to_csv(study / "momentum-response.csv", index=False)
    report["passed"] = report["complete"] and all(checks.values())
    write_json(study / "summary.json", report)
    return report


class ScriptTests(unittest.TestCase):
    """Pure helper checks; never start OPALX or modify study data."""

    def test_targeted_deck_edits_preserve_cof(self):
        base = ('// sigma_z=old\nTITLE, STRING="test";\nDIST: DISTRIBUTION, TYPE=GAUSS;\n'
                'FS: FIELDSOLVER, TYPE=NONE;\nBEAM1: BEAM, BCHARGE=1;\n'
                'ic: COF, DT=6.25e-11, MAXSTEPS=200000;\n'
                'TRACK, DT=6.25e-11, MAXSTEPS=200000, INITIALORBIT=ic;\nRUN, TURNS=32;\n')
        for case in CASES.values():
            for binning in ("none", "one"):
                deck = render_deck(base, case, binning)
                self.assertIn('ic: COF, DT=6.25e-11, MAXSTEPS=200000;', deck)
                self.assertNotIn('TURNS=', deck)
                self.assertIn('SIGMAZ=1e-3', deck)
                self.assertIn('REPARTFREQ=0', deck)
                self.assertIn(f"MAXSTEPS={case['steps']}, INITIALORBIT=ic", deck)
                if binning == "one":
                    self.assertIn('BINS=ONE', deck)
                    self.assertIn('MAXBINS=1, ADAPTIVEBINNING=FALSE', deck)
                    self.assertIn('DUMPBINSFILE="NONE", TABLEPRINTFREQ=0', deck)
                else:
                    self.assertIn('BINS=NONE', deck)
                    self.assertNotIn('ONE: BINNING', deck)
        with self.assertRaises(ValueError):
            replace_once("two two", "two", "one")

    def test_rotation_and_charge_scaling(self):
        np.testing.assert_allclose(laboratory_vectors([[1, 2, 3]], [1, 0, 0]), [[3, 2, -1]])
        delta = np.arange(1, 10, dtype=float).reshape(3, 3) * 1e-10
        metrics = response_metrics(delta, 2 * delta)
        self.assertAlmostEqual(metrics["charge_doubling_projection"], 2.)
        self.assertEqual(metrics["charge_doubling_relative_residual"], 0.)
        with self.assertRaises(ValueError):
            response_metrics(np.zeros((2, 3)), np.zeros((2, 3)))

    def test_h5_analysis_matches_ids_and_subtracts_each_resolution(self):
        with tempfile.TemporaryDirectory(prefix="boris-pic-test-") as temporary:
            study = Path(temporary)
            write_json(study / "study.json", {"binning": "one"})
            particle_ids = np.arange(32, dtype=np.int64)[::-1]
            offsets = np.column_stack((particle_ids - 15., particle_ids - 12., particle_ids - 10.))
            coarse_response = offsets * 1e-8
            for name, case in CASES.items():
                directory = study / name
                directory.mkdir()
                write_json(directory / "provenance.json", {**case, "completed": True, "binary_sha256": "fixture"})
                p = np.tile([0., 0., 1.], (32, 1))
                if "fine" in name:
                    p += offsets * 2e-5  # Large bare discretization change must cancel.
                if name == "q1":
                    p += coarse_response
                elif name == "q2":
                    p += 2 * coarse_response
                elif name == "q1_fine":
                    p += 0.99 * coarse_response
                with h5py.File(directory / "case.h5", "w") as output:
                    group = output.create_group("Step#0")
                    group.attrs["TIME"] = [6.25e-10]
                    group.attrs["GlobalTrackStep"] = [case["steps"] - 1]
                    group.attrs["RefPartR"] = [0., 0., 0.]
                    group.attrs["RefPartP"] = [0., 0., 1.]
                    group["id"] = particle_ids
                    for index, axis in enumerate("xyz"):
                        group[axis] = offsets[:, index] * 1e-5
                        group["p" + axis] = p[:, index]
                        group["E" + axis] = np.zeros(32) if name.startswith("bare") else offsets[:, index]
                        group["B" + axis] = np.zeros(32) if name.startswith("bare") and axis != "y" else np.ones(32)
            report = analyze(study)
            self.assertTrue(report["passed"])
            self.assertAlmostEqual(report["self_field_timestep_relative_sensitivity"], 0.01, places=8)
            result = pd.read_csv(study / "momentum-response.csv")
            np.testing.assert_array_equal(result.id, np.arange(32))
            with h5py.File(study / "q1" / "case.h5", "r+") as output:
                output["Step#0/Bx"][:] = 0.
                output["Step#0/Bz"][:] = 0.
            report = analyze(study)
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["q1_nonzero_self_magnetic_field"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1] / "bare-gaussian-cof.in")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--binning", choices=("none", "one"), default="none",
                        help="Mode for newly prepared studies; existing studies retain their saved mode.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", nargs="+", choices=list(CASES) + ["all"])
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--scaling-tolerance", type=float, default=0.05)
    parser.add_argument("--timestep-tolerance", type=float, default=0.1)
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScriptTests))
        return 0 if result.wasSuccessful() else 1
    if args.prepare_only and (args.run or args.analyze):
        parser.error("--prepare-only cannot be combined with --run or --analyze")
    if args.study:
        study = args.study.resolve()
        metadata = json.loads((study / "study.json").read_text())
        binary = (args.binary or Path(metadata["binary"])).resolve()
    else:
        base = args.base.resolve()
        repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=base.parent, text=True).strip())
        binary = (args.binary or repo / "omp-build/src/opalx").resolve()
        study = prepare(base, binary, args.binning)
    print(study, flush=True)
    if args.run:
        selected = list(CASES) if "all" in args.run else list(dict.fromkeys(args.run))
        for name in selected:
            print(f"Running {name}: one rank, one OpenMP thread", flush=True)
            run_case(study, name, binary)
    if args.analyze:
        report = analyze(study, args.scaling_tolerance, args.timestep_tolerance)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
