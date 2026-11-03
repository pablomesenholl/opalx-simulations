"""Periodic, uncoupled coasting KV model; SI units, geometric edge emittances.

a,b are twice the rms sizes; edge emittances are four times rms emittances.
K=2 I/(I_A beta_rel**3 gamma**3). No dispersion, images or longitudinal force.
Analytic self-fields are integrated numerically, separately across each hard edge.
"""
import numpy as np
from scipy.constants import c, elementary_charge, epsilon_0, proton_mass
from scipy.integrate import solve_ivp
from scipy.optimize import root


def beam_parameters(energy_MeV):
    """Return proton beta, gamma, rigidity [T m] and Alfven current [A]."""
    if energy_MeV <= 0:
        raise ValueError('Positive kinetic energy required')
    gamma = 1 + energy_MeV * 1e6 * elementary_charge / (proton_mass * c*c)
    beta = np.sqrt(1 - gamma**-2)
    return beta, gamma, beta*gamma*proton_mass*c/elementary_charge, 4*np.pi*epsilon_0*proton_mass*c**3/elementary_charge


def plane_map(k, length):
    """Exact hard-edge map for u''+k*u=0."""
    if k == 0:
        return np.array([[1., length], [0., 1.]])
    w = np.sqrt(abs(k))
    f, g = (np.cos(w*length), np.sin(w*length)/w) if k > 0 else (np.cosh(w*length), np.sinh(w*length)/w)
    return np.array([[f, g], [-k*g, f]])


def periodic_twiss(elements, axis):
    """Periodic beta [m], alpha for a stable cell, including phase sign."""
    matrix = np.eye(2)
    for el in elements:
        matrix = plane_map(el[f'k{axis}_m2'], el['length_m']) @ matrix
    cosine = np.trace(matrix)/2
    if abs(cosine) >= 1:
        raise ValueError('Unstable zero-current cell')
    sine = np.copysign(np.sqrt(1-cosine*cosine), matrix[0, 1])
    return matrix[0, 1]/sine, (matrix[0, 0]-matrix[1, 1])/(2*sine)


def envelope_rhs(s, state, kx, ky, ex, ey, perveance):
    """KV edge-envelope equations and positive accumulated particle phases."""
    a, ap, b, bp = state[:4]
    if min(a, b) <= 0:
        raise ValueError('Nonpositive envelope')
    force = 2*perveance/(a+b)
    return [ap, -kx*a + ex*ex/a**3 + force,
            bp, -ky*b + ey*ey/b**3 + force, ex/a**2, ey/b**2]


def propagate(initial, elements, ex, ey, perveance, rtol=2e-10, samples=0):
    """Advance one cell, resolving every boundary; optionally sample each element."""
    state = np.r_[initial[:4], 0., 0.]
    ss, yy = [0.], [state.copy()]
    offset = 0.
    for el in elements:
        length = el['length_m']
        sol = solve_ivp(envelope_rhs, (0, length), state, method='DOP853',
                        args=(el['kx_m2'], el['ky_m2'], ex, ey, perveance),
                        rtol=rtol, atol=rtol*1e-4, dense_output=bool(samples))
        if not sol.success:
            raise RuntimeError(sol.message)
        if samples:
            grid = np.linspace(0, length, samples+1)[1:]
            ss.extend(offset+grid)
            yy.extend(sol.sol(grid).T)
        state = sol.y[:, -1]
        offset += length
    return state, np.array(ss), np.array(yy)


def matched(elements, ex, ey, perveance, guess=None, rtol=2e-10):
    """Shoot periodic envelopes in scaled log-radius/slope coordinates."""
    if ex <= 0 or ey <= 0 or perveance < 0:
        raise ValueError('Positive emittances and nonnegative perveance required')
    bx, ax = periodic_twiss(elements, 'x')
    by, ay = periodic_twiss(elements, 'y')
    scale = np.array([np.sqrt(ex*bx), np.sqrt(ey*by)])
    def decode(z):
        return np.array([scale[0]*np.exp(z[0]), scale[0]*z[1], scale[1]*np.exp(z[2]), scale[1]*z[3]])
    def residual(z):
        initial = decode(z)
        end, _, _ = propagate(initial, elements, ex, ey, perveance, rtol)
        return (end[:4]-initial)/np.repeat(scale, 2)
    if guess is None:
        z0 = [0., -ax/bx, 0., -ay/by]
    else:
        z0 = [np.log(guess[0]/scale[0]), guess[1]/scale[0], np.log(guess[2]/scale[1]), guess[3]/scale[1]]
    def jacobian(z):
        # Fixed dimensionless increments resolve derivatives even when symmetric
        # lattice slopes are at roundoff; relative steps there would vanish.
        steps = np.eye(4)*1e-5
        return np.column_stack([(residual(z+h)-residual(z-h))/2e-5 for h in steps])
    # MINPACK's initial trust radius scales with ||z0||. For a symmetric
    # zero-current guess, roundoff-size slopes otherwise pin it near zero.
    z0 = np.asarray(z0)
    z0[np.abs(z0) < 1e-12] = 0.
    solution = root(residual, z0, jac=jacobian, tol=1e-9)
    error = np.max(np.abs(residual(solution.x)))
    if error > 2e-8:
        raise RuntimeError(f'Periodic matching failed: {error}, {solution.message}')
    initial = decode(solution.x)
    end, _, _ = propagate(initial, elements, ex, ey, perveance, rtol)
    return initial, end[4:]/(2*np.pi), error


def constant_solution(k, edge_emittance, perveance):
    """Exact round matched radius and depressed wavenumber in uniform focusing."""
    if k <= 0 or edge_emittance <= 0 or perveance < 0:
        raise ValueError('Invalid constant-focusing parameters')
    radius2 = (perveance + np.sqrt(perveance**2 + 4*k*edge_emittance**2))/(2*k)
    return np.sqrt(radius2), edge_emittance/radius2


def envelope_multipliers(initial, elements, ex, ey, perveance):
    """Cell envelope Floquet multipliers from the linearized coupled equations."""
    state = np.r_[initial, np.eye(4).ravel()]
    def rhs(s, y, kx, ky):
        a, ap, b, bp = y[:4]
        coupling = 2*perveance/(a+b)**2
        jac = np.array([[0, 1, 0, 0],
                        [-kx-3*ex**2/a**4-coupling, 0, -coupling, 0],
                        [0, 0, 0, 1],
                        [-coupling, 0, -ky-3*ey**2/b**4-coupling, 0]])
        return np.r_[envelope_rhs(s, y[:4], kx, ky, ex, ey, perveance)[:4],
                     (jac @ y[4:].reshape(4, 4)).ravel()]
    for el in elements:
        sol = solve_ivp(rhs, (0, el['length_m']), state, args=(el['kx_m2'], el['ky_m2']),
                        method='DOP853', rtol=2e-10, atol=2e-13)
        if not sol.success:
            raise RuntimeError(sol.message)
        state = sol.y[:, -1]
    return np.linalg.eigvals(state[4:].reshape(4, 4))
