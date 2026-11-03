"""Compare the OPALX mechanical one-turn map to analytic slope-coordinate optics."""
import argparse
import json
from pathlib import Path
import numpy as np
from kv_model import beam_parameters, plane_map, matched


def compare(config, result):
    """Validate closure, energy and map; assign tune branches from cell optics."""
    beta, gamma, _, _ = beam_parameters(config['kinetic_energy_MeV'])
    transform = np.diag([1., beta*gamma, 1., beta*gamma])
    slope_map = np.linalg.inv(transform) @ np.array(result['matrix']) @ transform
    expected = np.zeros((4, 4))
    for axis, start in [('x', 0), ('y', 2)]:
        cell = np.eye(2)
        for el in config['elements']:
            cell = plane_map(el[f'k{axis}_m2'], el['length_m']) @ cell
        expected[start:start+2, start:start+2] = np.linalg.matrix_power(cell, config['cells'])
    _, q, _ = matched(config['elements'], 4*config['rms_emittance_x_m_rad'],
                      4*config['rms_emittance_y_m_rad'], 0.)
    full = config['cells']*q
    modes = result['fractional_modes']
    opal_full = []
    for i in range(2):
        if modes[i]['tune'] is None:
            raise ValueError('Missing OPALX stable mode')
        candidates = [modes[i]['tune'], modes[i]['complement']]
        opal_full.append(min((t+round(full[i]-t) for t in candidates), key=lambda t: abs(t-full[i])))
    delta = np.array(opal_full)-full
    error = float(np.max(np.abs(slope_map-expected)/(1+np.abs(expected))))
    report = dict(analytic_full_tunes=full.tolist(), opalx_branch_assigned_tunes=opal_full,
                  tune_difference=delta.tolist(), scaled_map_error=error,
                  closed_orbit=result['coordinates'], residual=result['residual'],
                  stability=result['stability'], energy_MeV=result['energy_MeV'],
                  note='Integer/complement branches assigned using analytic cell phase; COF alone gives fractional eigenphases')
    if (not result['converged'] or result['stability'] != 'STABLE' or error > 1e-6
            or max(abs(delta)) > 1e-7 or max(abs(np.array(result['residual']))) > 1e-8
            or abs(result['energy_MeV']-config['kinetic_energy_MeV']) > 1e-6):
        raise ValueError(f'Bare validation failed: {report}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.result_dir/'config.json'
    if not config_path.exists():
        config_path = root/'config.json'
    report = compare(json.loads(config_path.read_text()),
                     json.loads((args.result_dir/'bare-cof.json').read_text()))
    (args.result_dir/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
