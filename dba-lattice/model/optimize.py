"""Local two-knob linear-optics design example; user-specified full tune targets.

Vary the physical quadrupole coefficient and both equal drift lengths. The
bend angle stays fixed; six-fold symmetry preserves reference-orbit closure.
This local optimizer is a deterministic scoring/reference tool for an AI, not
an AI design claim. It stays on the baseline cell's tune branch.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import root
from machine import generate


def tune_design(spec, target):
    """Match two full tunes, reject nonphysical/unstable/unclosed candidates."""
    quadrupoles = [e for e in spec['elements'] if e['type'] == 'quadrupole']
    drifts = [e for e in spec['elements'] if e['type'] == 'drift']
    if len(quadrupoles) != 1 or len(drifts) != 2 or drifts[0]['length_m'] != drifts[1]['length_m']:
        raise ValueError('Expected one quadrupole family and two equal drifts')
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.all(np.isfinite(target)) or np.any(target <= 0):
        raise ValueError('Two positive finite full tune targets required')
    def candidate(knobs):
        result = copy.deepcopy(spec)
        strength, length = knobs
        if strength <= 0 or length <= 0:
            raise ValueError('Positive strength and drift length required')
        for el in result['elements']:
            if el['type'] == 'quadrupole':
                el['kx_m2'], el['ky_m2'] = -strength, strength
            elif el['type'] == 'drift':
                el['length_m'] = length
        return result
    def residual(knobs):
        return np.array(generate(candidate(knobs))['full_tunes'])-target
    start = [quadrupoles[0]['ky_m2'], drifts[0]['length_m']]
    solution = root(residual, start, tol=1e-10)
    if not solution.success or np.max(np.abs(residual(solution.x))) > 1e-8:
        raise RuntimeError('Local tune match failed; choose closer targets or a different search')
    result = candidate(solution.x)
    machine = generate(result)
    if machine['position_closure_m'] > 1e-10 or machine['symplectic_error'] > 1e-10:
        raise RuntimeError('Geometry or symplecticity validation failed')
    result['name'] = spec['name']+'-tune-matched'
    result['design'] = dict(target_full_tunes=target.tolist(), algorithm='local deterministic root',
                            knobs=['quadrupole ky [m^-2]', 'equal drift lengths [m]'],
                            warning='No fixed circumference constraint; KV rematching required')
    return result


def main():
    """Export a fresh optimized cell, expanded machine and compatible KV config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=Path(__file__).with_name('cell.json'))
    parser.add_argument('--qx', type=float, required=True)
    parser.add_argument('--qy', type=float, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    spec = tune_design(json.loads(args.spec.read_text()), [args.qx, args.qy])
    kv_path = Path(__file__).resolve().parents[1]/'tune-shift-ring/config.json'
    kv = json.loads(kv_path.read_text())
    kv.update(kinetic_energy_MeV=spec['kinetic_energy_MeV'], cells=spec['cells'],
              elements=[{k: e[k] for k in ['name', 'length_m', 'kx_m2', 'ky_m2']}
                        for e in spec['elements']])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in [('cell.json', spec), ('machine.json', generate(spec)), ('kv-config.json', kv)]:
        (args.output_dir/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output_dir), full_tunes=generate(spec)['full_tunes'],
                         circumference_m=generate(spec)['circumference_m']), indent=2))


if __name__ == '__main__':
    main()
