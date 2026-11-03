#!/usr/bin/env python3
"""Run nominal and displaced-seed COF cases and compare with exact optics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(name: str, text: str, output: Path, executable: Path, analytic: np.ndarray) -> dict:
    case = output / name
    case.mkdir()
    (case / "cof.in").write_text(text)
    command = [str(executable), "--info", "1", "cof.in"]
    started = time.monotonic()
    with (case / "run.log").open("w") as stream:
        completed = subprocess.run(
            command,
            cwd=case,
            env=os.environ | {"OMP_NUM_THREADS": "1", "OMP_PROC_BIND": "false", "OMP_DYNAMIC": "false"},
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"{name} failed; inspect {case / 'run.log'}")
    result = json.loads((case / "closed-orbit.json").read_text())
    match = re.search(r"p0_mc=([^ ]+)", (case / "run.log").read_text())
    if match is None:
        raise ValueError("could not recover normalized reference momentum")
    momentum = float(match[1])
    transform = np.diag([1.0, momentum, 1.0, momentum])
    measured = np.linalg.inv(transform) @ np.asarray(result["matrix"]) @ transform
    difference = measured - analytic[:4, :4]
    np.savetxt(case / "tracked-slope-map.txt", measured, fmt="%.16e")
    np.savetxt(case / "minus-analytic.txt", difference, fmt="%+.16e")
    residual = np.asarray(result["residual"])
    passed = (
        result["stability"] == "STABLE"
        and float(abs(difference).max()) < 1e-6
        and float(abs(residual).max()) < 1e-7
    )
    record = {
        "case": name,
        "status": "PASS" if passed else "FAIL",
        "elapsed_s": time.monotonic() - started,
        "stability": result["stability"],
        "coordinates": result["coordinates"],
        "residual": result["residual"],
        "max_matrix_error": float(abs(difference).max()),
        "relative_energy_drift": result["relative_energy_drift"],
        "command": command,
        "input_sha256": digest(case / "cof.in"),
    }
    (case / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    template = (ROOT / "cof.in").read_text()
    cases = {
        "nominal": template,
        "displaced": template.replace(
            "X=0, PX=0, Y=0, PY=0,",
            "X=0.0001, PX=0.001, Y=0.0001, PY=0.001,",
        ),
    }
    if cases["displaced"] == template:
        raise ValueError("could not construct displaced-seed input")
    analytic = np.loadtxt(args.analytic / "ring-slope-map.txt")
    records = [run_case(name, text, args.output, args.executable.resolve(), analytic)
               for name, text in cases.items()]
    pd.DataFrame(records).to_csv(args.output / "summary.csv", index=False)
    summary = {"all_passed": all(row["status"] == "PASS" for row in records), "cases": records}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(records)[["case", "status", "stability", "max_matrix_error", "elapsed_s"]].to_string(index=False))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
