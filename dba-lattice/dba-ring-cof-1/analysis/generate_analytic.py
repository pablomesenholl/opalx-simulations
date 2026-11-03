#!/usr/bin/env python3
"""Generate the exact hard-edge linear optics for the six-cell DBA ring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P = 490.23677597553325
GAMMA2 = 1.0 + P**2


def drift(length: float) -> np.ndarray:
    matrix = np.eye(6)
    matrix[0, 1] = matrix[2, 3] = length
    matrix[4, 5] = length / GAMMA2
    return matrix


def plane(k: float, length: float) -> np.ndarray:
    if k == 0.0:
        return np.array([[1.0, length], [0.0, 1.0]])
    root = math.sqrt(abs(k))
    phase = root * length
    if k > 0:
        return np.array([[math.cos(phase), math.sin(phase) / root],
                         [-root * math.sin(phase), math.cos(phase)]])
    return np.array([[math.cosh(phase), math.sinh(phase) / root],
                     [root * math.sinh(phase), math.cosh(phase)]])


def quadrupole(k1: float, length: float) -> np.ndarray:
    matrix = np.eye(6)
    matrix[:2, :2] = plane(-k1, length)
    matrix[2:4, 2:4] = plane(k1, length)
    matrix[4, 5] = length / GAMMA2
    return matrix


def sector_bend(radius: float, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    length = radius * angle
    matrix = np.eye(6)
    matrix[0, 0] = matrix[1, 1] = c
    matrix[0, 1] = radius * s
    matrix[1, 0] = -s / radius
    matrix[0, 5] = radius * (1.0 - c)
    matrix[1, 5] = s
    matrix[2, 3] = length
    matrix[4, 0] = -s
    matrix[4, 1] = -radius * (1.0 - c)
    matrix[4, 5] = radius * (s - angle) + length / GAMMA2
    return matrix


def product(*matrices: np.ndarray) -> np.ndarray:
    result = np.eye(6)
    for matrix in matrices:
        result = matrix @ result
    return result


def cell(k1: float) -> np.ndarray:
    bend = sector_bend(2.0, math.pi / 6.0)
    return product(bend, drift(1.0), quadrupole(k1, 0.2), drift(1.0), bend)


def describe(k1: float) -> dict[str, float | bool]:
    matrix = cell(k1)
    half_trace = np.array([np.trace(matrix[:2, :2]), np.trace(matrix[2:4, 2:4])]) / 2.0
    stable = bool(np.all(abs(half_trace) < 1.0))
    row: dict[str, float | bool] = {
        "k1_m2": float(k1),
        "half_trace_x": float(half_trace[0]),
        "half_trace_y": float(half_trace[1]),
        "stable": stable,
    }
    if stable:
        phase = (6.0 * np.arccos(half_trace) / (2.0 * math.pi)) % 1.0
        principal = np.minimum(phase, 1.0 - phase)
        row.update(qx_principal=float(principal[0]), qy_principal=float(principal[1]))
    return row


def generate(output: Path) -> dict:
    scan = pd.DataFrame([describe(k1) for k1 in np.linspace(-15.0, 15.0, 6001)])
    candidates = scan[
        scan.stable & (scan.half_trace_x.abs() < 0.95) & (scan.half_trace_y.abs() < 0.95)
    ].copy()
    candidates["score"] = (candidates.qx_principal - 0.23) ** 2 + (candidates.qy_principal - 0.31) ** 2
    selected = candidates.sort_values(["score", "k1_m2"]).iloc[0]
    cell_map = cell(float(selected.k1_m2))
    ring = np.linalg.matrix_power(cell_map, 6)
    transform = np.diag([1.0, P, 1.0, P])
    mechanical = transform @ ring[:4, :4] @ np.linalg.inv(transform)
    dispersion = np.linalg.solve(np.eye(2) - ring[:2, :2], ring[:2, 5])
    output.mkdir(parents=True, exist_ok=True)
    scan.to_csv(output / "one-family-scan.csv", index=False, float_format="%.16e")
    np.savetxt(output / "cell-slope-map.txt", cell_map, fmt="%.16e")
    np.savetxt(output / "ring-slope-map.txt", ring, fmt="%.16e")
    np.savetxt(output / "ring-mechanical-map.txt", mechanical, fmt="%.16e")
    summary = {
        "model": "exact hard-edge linear optics; no OPALX tracking",
        "selected_k1_m2": float(selected.k1_m2),
        "qx_principal": float(selected.qx_principal),
        "qy_principal": float(selected.qy_principal),
        "circumference_m": 6.0 * (2.0 * math.pi / 3.0 + 2.2),
        "periodic_dispersion_m": float(dispersion[0]),
        "periodic_dispersion_slope": float(dispersion[1]),
        "stable_samples": int(scan.stable.sum()),
        "determinant_error": float(abs(np.linalg.det(ring) - 1.0)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "analytic")
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
