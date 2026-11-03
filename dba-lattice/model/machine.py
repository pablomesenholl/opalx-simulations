"""Generate a ring from a quadratic canonical cell Hamiltonian; SI units.

Independent variable s [m], state (x,px/p0,y,py/p0,zeta,eta).
See TeX/main.tex for the energy/time canonical pair and physical scope.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import c, elementary_charge, proton_mass
from scipy.linalg import expm, block_diag

J = block_diag(*([np.array([[0., 1.], [-1., 0.]])] * 3))


def hessian(element, gamma):
    """Symmetric G for H=u.T G u/2; canonical generator is J G."""
    if gamma <= 1 or element['length_m'] <= 0:
        raise ValueError('Positive kinetic energy and element length required')
    g = np.diag([element['kx_m2'], 1., element['ky_m2'], 1., 0., 1/gamma**2])
    g[0, 5] = g[5, 0] = -element['h_per_m']
    return g


def element_map(element, gamma):
    """Exact map of the constant quadratic Hamiltonian across one element."""
    return expm(J @ hessian(element, gamma) * element['length_m'])


def advance_pose(pose, length, curvature):
    """Return lab (X,Z,theta); tangent=(sin theta,cos theta), bends turn -X."""
    x, z, theta = pose
    end = theta-curvature*length
    if abs(curvature) < 1e-15:
        return np.array([x+length*np.sin(theta), z+length*np.cos(theta), theta])
    return np.array([x+(np.cos(end)-np.cos(theta))/curvature,
                     z+(np.sin(theta)-np.sin(end))/curvature, end])


def generate(spec):
    """Expand ordered cells, exact maps, placements, dispersion and diagnostics."""
    if spec['particle'] != 'proton' or spec['cells'] < 1 or not isinstance(spec['cells'], int):
        raise ValueError('This fixture requires protons and a positive integer cell count')
    gamma = 1+spec['kinetic_energy_MeV']*1e6*elementary_charge/(proton_mass*c*c)
    beta = np.sqrt(1-gamma**-2)
    cell = np.eye(6)
    for el in spec['elements']:
        cell = element_map(el, gamma) @ cell
    ring = np.linalg.matrix_power(cell, spec['cells'])
    phases = []
    for start in [0, 2]:
        m = cell[start:start+2, start:start+2]
        cosine = np.trace(m)/2
        if abs(cosine) >= 1:
            raise ValueError('Transversely unstable cell')
        phase = np.arccos(cosine)
        if m[0, 1] < 0:
            phase = 2*np.pi-phase
        phases.append(float(phase))
    pose, distance, elements = np.zeros(3), 0., []
    for number in range(1, spec['cells']+1):
        for el in spec['elements']:
            end = advance_pose(pose, el['length_m'], el['h_per_m'])
            elements.append(dict(el, name=f'C{number}_{el["name"]}', s_start_m=distance,
                                 entry_X_Z_theta=pose.tolist(), exit_X_Z_theta=end.tolist(),
                                 hamiltonian_hessian=hessian(el, gamma).tolist(),
                                 map_6d=element_map(el, gamma).tolist()))
            pose = end
            distance += el['length_m']
    dispersion = np.linalg.solve(np.eye(2)-ring[:2, :2], ring[:2, 5])
    return dict(name=spec['name'], particle=spec['particle'],
                kinetic_energy_MeV=spec['kinetic_energy_MeV'],
                canonical_coordinates=spec['canonical_coordinates'],
                longitudinal_convention=spec['longitudinal_convention'],
                hamiltonian=spec['hamiltonian'], model_order=spec['model_order'],
                gamma0=gamma, beta0=beta, circumference_m=distance,
                cell_map_6d=cell.tolist(), ring_map_6d=ring.tolist(),
                full_tunes=(spec['cells']*np.array(phases)/(2*np.pi)).tolist(),
                tune_branch='unwrapped as repeated stable cell phase in (0,2*pi)',
                periodic_dispersion_m_and_slope=dispersion.tolist(),
                symplectic_error=float(np.max(np.abs(ring.T@J@ring-J))),
                position_closure_m=float(np.linalg.norm(pose[:2])),
                total_bend_rad=float(-pose[2]), elements=elements)


def main():
    """Write a fresh machine JSON; refuse to overwrite an existing artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=Path(__file__).with_name('cell.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = generate(json.loads(args.spec.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ['elements', 'cell_map_6d', 'ring_map_6d']}, indent=2))


if __name__ == '__main__':
    main()
