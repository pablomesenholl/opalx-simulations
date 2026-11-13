#!/usr/bin/env python3
"""Generate an OPALX FODO lattice include file.

The output is intended to be imported from a main OPALX input using:
    CALL, FILE="FODO-spacecharge.in";

The main input must define LQ, LD, k1, and Lcell *before* CALL.
Apertures are optional; the default is no physical aperture.

Example:
    python3 generate_lattice.py --periods 10
    python3 generate_lattice.py --periods 10 --aperture-radius-mm 10
    python3 generate_lattice.py --periods 10 --output fodo_lattice.in
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_lattice(periods: int, aperture_radius_mm: float | None) -> str:
    if periods < 1:
        raise ValueError("--periods must be >= 1")
    if aperture_radius_mm is not None and aperture_radius_mm <= 0:
        raise ValueError("--aperture-radius-mm must be positive")

    # OPALX CIRCLE(...) takes the FULL diameter in metres, not the radius.
    aperture = ""
    if aperture_radius_mm is not None:
        diameter_m = 2.0 * aperture_radius_mm * 1e-3
        aperture = f', APERTURE="CIRCLE({diameter_m:.12g})"'

    text = [
        "// Generated OPALX FODO lattice; do not edit by hand.",
        "// Required definitions in caller: LQ, LD, k1, Lcell.",
        "// Each period is QF -> D1 -> QD -> D2 (length Lcell).",
        "// This file must NOT contain QUIT or ENDTRACK.",
        f"// Physical aperture: {'none' if aperture_radius_mm is None else f'circle, radius {aperture_radius_mm:g} mm'}.",
        "",
        f"REAL n_periods={periods};",
        "",
    ]

    sequence: list[str] = []
    for index in range(periods):
        num = f"{index + 1:02d}"
        qf, d1, qd, d2 = (f"{part}_{num}" for part in ("QF", "D1", "QD", "D2"))
        base = f"({index}*Lcell)"
        text.extend([
            f"// Period {index + 1}",
            f"{qf}: QUADRUPOLE, L=LQ, ELEMEDGE={base}, K1=k1{aperture};",
            f"{d1}: DRIFT, L=LD, ELEMEDGE={base}+LQ{aperture};",
            f"{qd}: QUADRUPOLE, L=LQ, ELEMEDGE={base}+LQ+LD, K1=-k1{aperture};",
            f"{d2}: DRIFT, L=LD, ELEMEDGE={base}+2*LQ+LD{aperture};",
            "",
        ])
        sequence.extend((qf, d1, qd, d2))

    # Give every physical occurrence its own name and placement.
    # Avoid repeating a named element with a fixed ELEMEDGE.
    text.append("FODO_LATTICE: LINE=(")
    for j in range(0, len(sequence), 4):
        group = ", ".join(sequence[j : j + 4])
        trailing_comma = "," if j + 4 < len(sequence) else ""
        text.append(f"    {group}{trailing_comma}")
    text.extend([ ");", ""])
    return "\n".join(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, default=10,
                        help="number of FODO periods (default: 10)")
    parser.add_argument("--output", type=Path, default=Path("fodo_lattice.in"),
                        help="generated OPALX include file (default: fodo_lattice.in)")
    parser.add_argument("--aperture-radius-mm", type=float, default=None,
                        help="optional circular aperture radius in mm (default: no aperture)")
    args = parser.parse_args()

    try:
        content = build_lattice(args.periods, args.aperture_radius_mm)
    except ValueError as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}: {args.periods} periods, {4*args.periods} elements, "
          f"{'no aperture' if args.aperture_radius_mm is None else f'aperture radius {args.aperture_radius_mm:g} mm'}")


if __name__ == "__main__":
    main()
