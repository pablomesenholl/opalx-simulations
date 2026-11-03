"""Reproducible KV shell sampling; slopes are paraxial mechanical slopes.

Section samples are not a finite 3D bunch. Ring samples carry longitudinal
position s and a local transverse matched section at each s.
"""
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from kv_model import envelope_rhs


def quiet_sphere(count, seed):
    """Uniform S3 samples with exact first/second moments in each block of 64."""
    if count <= 0 or count % 64:
        raise ValueError('Particle count must be a positive multiple of 64')
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(count//64, 4))
    base /= np.linalg.norm(base, axis=1)[:, None]
    signs = np.array(list(itertools.product([-1., 1.], repeat=4)))
    return np.stack([np.roll(base, j, axis=1)[:, None, :]*signs for j in range(4)], axis=1).reshape(count, 4)


def transverse_coordinates(unit, envelopes, ex, ey):
    """Map unit S3 to (x,x',y,y') using KV edge radii and edge emittances."""
    env = np.broadcast_to(envelopes, (len(unit), 4))
    a, ap, b, bp = env.T
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError('Positive radii required')
    return np.column_stack((a*unit[:, 0], ap*unit[:, 0]+ex/a*unit[:, 1],
                            b*unit[:, 2], bp*unit[:, 2]+ey/b*unit[:, 3]))


def section_fromfile(transverse, bg):
    """Zero-length observation section, x px y py z pz; p is in m_p*c units.

    Preserve total momentum exactly, giving zero momentum-magnitude spread.
    This section must not be loaded as a charged finite bunch for 3D PIC.
    """
    x, xp, y, yp = transverse.T
    pz = bg/np.sqrt(1+xp*xp+yp*yp)
    return np.column_stack((x, xp*pz, y, yp*pz, np.zeros(len(x)), pz))


def envelopes_at(s, initial, elements, ex, ey, K):
    """Evaluate the matched envelope at arbitrary positions in a periodic cell."""
    length = sum(el['length_m'] for el in elements)
    local = np.remainder(np.asarray(s), length)
    result = np.empty((len(local), 4))
    state, offset = np.r_[initial, 0., 0.], 0.
    for index, el in enumerate(elements):
        end = offset+el['length_m']
        sol = solve_ivp(envelope_rhs, (0., el['length_m']), state, method='DOP853',
                        args=(el['kx_m2'], el['ky_m2'], ex, ey, K),
                        rtol=2e-11, atol=2e-15, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        selected = (local >= offset) & (local < end)
        if index == len(elements)-1:
            selected = local >= offset
        result[selected] = sol.sol(local[selected]-offset)[:4].T
        state, offset = sol.y[:, -1], end
    return result


def shell_invariant(transverse, envelopes, ex, ey):
    """KV quadratic invariant; every ideal particle has value one."""
    a, ap, b, bp = np.broadcast_to(envelopes, (len(transverse), 4)).T
    x, xp, y, yp = transverse.T
    return (x/a)**2+((a*xp-ap*x)/ex)**2+(y/b)**2+((b*yp-bp*y)/ey)**2


def moments(transverse):
    """Population covariance and geometric rms emittances, without ddof correction."""
    covariance = np.cov(transverse, rowvar=False, bias=True)
    return dict(centroid=transverse.mean(axis=0).tolist(), covariance=covariance.tolist(),
                rms_emittance_x_m_rad=float(np.sqrt(np.linalg.det(covariance[:2, :2]))),
                rms_emittance_y_m_rad=float(np.sqrt(np.linalg.det(covariance[2:, 2:]))))
