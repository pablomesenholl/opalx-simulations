"""Uniform ellipsoid internal fields and rms closure in a linear curved lattice.

Coordinates (x,px/p0,y,py/p0,z,delta), z positive toward bunch head.
Uniform spatial density is an assumed closure, not an exact finite-emittance
Vlasov equilibrium. Fields are exact inside the assumed isolated ellipsoid.
"""
import numpy as np
from scipy.constants import c, elementary_charge, epsilon_0, proton_mass
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.special import elliprd
from scipy.linalg import expm
from kv_model import beam_parameters, periodic_twiss


def depolarization(axes):
    """Ellipsoid factors A_i, sum=1, using Carlson's symmetric R_D integral."""
    axes = np.asarray(axes, dtype=float)
    if axes.shape != (3,) or np.any(axes <= 0):
        raise ValueError('Three positive semiaxes required')
    a = axes/max(axes)
    return np.array([np.prod(a)/3*elliprd(a[(i+1)%3]**2, a[(i+2)%3]**2, a[i]**2) for i in range(3)])


def field_tensor(position_covariance, charge_C, beta, gamma):
    """Symmetric force coefficients S: (px/p0,py/p0,delta)' = S (x,y,z).

    Position covariance is lab-frame m^2. Rest-frame ellipsoid semiaxes are
    sqrt(5 eigenvalues(D Sigma_r D)), D=diag(1,1,gamma). Relativistic magnetic
    cancellation is included for direct space charge; no conducting boundaries.
    """
    if charge_C < 0:
        raise ValueError('Proton bunch charge must be nonnegative')
    stretch = np.diag([1., 1., gamma])
    values, vectors = np.linalg.eigh(stretch @ position_covariance @ stretch)
    if min(values) <= 0:
        raise ValueError('Positive definite spatial covariance required')
    axes = np.sqrt(5*values)
    rho_rest = charge_C/(4*np.pi*np.prod(axes)/3)
    rest = (vectors*depolarization(axes)) @ vectors.T
    coefficient = elementary_charge*rho_rest/(epsilon_0*proton_mass*c*c*beta*beta*gamma*gamma)
    return coefficient*stretch @ rest @ stretch


def axis_coefficients(a, b, length, charge, beta, gamma):
    """Aligned ellipsoid coefficients for semiaxes a,b,length in lab metres."""
    return np.diag(field_tensor(np.diag(np.array([a, b, length])**2/5), charge, beta, gamma))


def transverse_pass(initial, elements, ex, ey, length, charge, beta, gamma, samples=0, rtol=2e-10):
    """Fixed-length, zero-dispersion transverse rms closure; ex,ey=5 eps_rms."""
    def rhs(s, u, kx, ky):
        a, ap, b, bp = u[:4]
        sx, sy, _ = axis_coefficients(a, b, length, charge, beta, gamma)
        return [ap, -(kx-sx)*a+ex*ex/a**3, bp, -(ky-sy)*b+ey*ey/b**3, ex/a**2, ey/b**2]
    state, offset = np.r_[initial[:4], 0., 0.], 0.
    positions, states = [0.], [state.copy()]
    for el in elements:
        sol = solve_ivp(rhs, (0, el['length_m']), state, args=(el['kx_m2'], el['ky_m2']),
                        method='DOP853', rtol=rtol, atol=rtol*1e-4, dense_output=bool(samples))
        if not sol.success:
            raise RuntimeError(sol.message)
        if samples:
            ss = np.linspace(0, el['length_m'], samples+1)[1:]
            positions.extend(offset+ss)
            states.extend(sol.sol(ss).T)
        state, offset = sol.y[:, -1], offset+el['length_m']
    return state, np.array(positions), np.array(states)


def match_transverse(elements, epsx, epsy, length, charge, beta, gamma, rtol=2e-10):
    """Periodic transverse ellipsoid rms closure with externally fixed length."""
    ex, ey = 5*epsx, 5*epsy
    bx, ax = periodic_twiss(elements, 'x')
    by, ay = periodic_twiss(elements, 'y')
    scale = np.array([np.sqrt(ex*bx), np.sqrt(ey*by)])
    def decode(z):
        return np.array([scale[0]*np.exp(z[0]), scale[0]*z[1], scale[1]*np.exp(z[2]), scale[1]*z[3]])
    def residual(z):
        start = decode(z)
        end = transverse_pass(start, elements, ex, ey, length, charge, beta, gamma, rtol=rtol)[0]
        return (end[:4]-start)/np.repeat(scale, 2)
    def jac(z):
        return np.column_stack([(residual(z+h)-residual(z-h))/2e-5 for h in np.eye(4)*1e-5])
    z0 = np.array([0., -ax/bx, 0., -ay/by])
    z0[abs(z0)<1e-12] = 0.
    fit = root(residual, z0, jac=jac, tol=1e-9)
    error = max(abs(residual(fit.x)))
    if error > 2e-8:
        raise RuntimeError(f'Ellipsoid matching failed: {error}')
    start = decode(fit.x)
    end = transverse_pass(start, elements, ex, ey, length, charge, beta, gamma, rtol=rtol)[0]
    return start, end[4:]/(2*np.pi), error


def transport_generator(element, gamma):
    """Hamiltonian paraxial ring generator with curvature/dispersion coupling.

    Bend convention h=+1/rho matches the analytic sector map; z'=delta/gamma^2-h*x.
    This oriented local frame must be translated consistently to OPALX placement.
    """
    h = np.sqrt(element['kx_m2']) if element['name'].startswith('B') else 0.
    matrix = np.zeros((6, 6))
    matrix[0, 1] = matrix[2, 3] = 1.
    matrix[1, 0], matrix[3, 2] = -element['kx_m2'], -element['ky_m2']
    matrix[1, 5], matrix[4, 0], matrix[4, 5] = h, -h, gamma**-2
    return matrix


def initial_covariance(envelope, epsx, epsy, length, sigma_delta):
    """Initially untilted uniform-ellipsoid rms closure; no imposed z-delta chirp."""
    result = np.zeros((6, 6))
    for i, eps, (a, ap) in [(0, epsx, envelope[:2]), (2, epsy, envelope[2:])]:
        result[i:i+2, i:i+2] = [[a*a/5, a*ap/5], [a*ap/5, (ap*ap+(5*eps/a)**2)/5]]
    result[4, 4], result[5, 5] = length*length/5, sigma_delta*sigma_delta
    return result


def rf_kick(slope_per_m):
    """Thin zero-crossing buncher: delta_out=delta_in+slope*z; no mean energy gain."""
    matrix = np.eye(6)
    matrix[5, 4] = slope_per_m
    return matrix


def ring_slip(elements, cells, gamma):
    """Momentum compaction and eta=alpha_p-1/gamma^2, including closed dispersion."""
    cell = np.eye(6)
    for el in elements:
        cell = expm(transport_generator(el, gamma)*el['length_m']) @ cell
    ring = np.linalg.matrix_power(cell, cells)
    dispersion = np.linalg.solve(np.eye(2)-ring[:2, :2], ring[:2, 5])
    circumference = cells*sum(el['length_m'] for el in elements)
    r56_closed = ring[4, 5]+ring[4, :2]@dispersion
    eta = -r56_closed/circumference
    return dict(eta=float(eta), alpha_p=float(eta+gamma**-2), closed_R56_m=float(r56_closed), dispersion=dispersion.tolist())


def evolve_covariance(initial, elements, cells, turns, charge, beta, gamma,
                      rf_slope=0., samples=20, maximum_axis=.2, rtol=2e-11):
    """Coupled 6D rms closure with a thin cavity at each cell exit.

    Reconstruct an equivalent uniform ellipsoid from spatial covariance after
    each ODE evaluation, retaining x-z tilt and dispersion. Stop at the chosen
    small-bunch validity bound. This is not self-consistent particle tracking.
    """
    pos = [0, 2, 4]
    mom = [1, 3, 5]
    scale = np.sqrt(np.diag(initial))
    outer = np.outer(scale, scale)
    state = (initial/outer).ravel()
    rows = [(0., initial.copy())]
    offset, reason = 0., 'completed requested path'
    def rhs(s, y, external):
        cov = y.reshape(6, 6)*outer
        cov = (cov+cov.T)/2
        matrix = external.copy()
        matrix[np.ix_(mom, pos)] += field_tensor(cov[np.ix_(pos, pos)], charge, beta, gamma)
        return ((matrix@cov+cov@matrix.T)/outer).ravel()
    def limit(s, y, external):
        cov = y.reshape(6, 6)*outer
        return maximum_axis-np.sqrt(5*max(np.linalg.eigvalsh(cov[np.ix_(pos, pos)])))
    limit.terminal, limit.direction = True, -1
    for _ in range(cells*turns):
        for el in elements:
            matrix = transport_generator(el, gamma)
            sol = solve_ivp(rhs, (0, el['length_m']), state, args=(matrix,), method='DOP853',
                            rtol=rtol, atol=rtol*.01, dense_output=True, events=limit)
            if not sol.success:
                raise RuntimeError(sol.message)
            grid = np.linspace(0, sol.t[-1], samples+1)[1:]
            rows.extend((offset+s, sol.sol(s).reshape(6, 6)*outer) for s in grid)
            state, offset = sol.y[:, -1], offset+sol.t[-1]
            if sol.status == 1:
                return rows, 'stopped: spatial semiaxis reached local-model validity bound'
        cov = state.reshape(6, 6)*outer
        kick = rf_kick(rf_slope)
        cov = kick@cov@kick.T
        state = (cov/outer).ravel()
        rows.append((offset, cov.copy()))
    return rows, reason
