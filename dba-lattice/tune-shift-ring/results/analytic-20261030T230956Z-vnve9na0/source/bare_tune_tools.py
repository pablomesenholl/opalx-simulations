"""Bare tune extraction from densely sampled particle position and slope."""
import numpy as np
from scipy.interpolate import CubicSpline, BarycentricInterpolator
from scipy.optimize import brentq, minimize_scalar
from kv_model import plane_map


def cell_optics(config, axis):
    """Exact hard-edge cell matrix, full ring tune, and section Twiss parameters."""
    matrix = np.eye(2)
    for element in config['elements']:
        matrix = plane_map(element[f'k{axis}_m2'], element['length_m']) @ matrix
    cosine = np.trace(matrix)/2
    if abs(cosine) >= 1:
        raise ValueError('Unstable cell')
    sine = np.copysign(np.sqrt(1-cosine*cosine), matrix[0, 1])
    phase = np.arctan2(sine, cosine) % (2*np.pi)
    return dict(matrix=matrix, tune=config['cells']*phase/(2*np.pi),
                beta=matrix[0, 1]/sine, alpha=(matrix[0, 0]-matrix[1, 1])/(2*sine))


def section_returns(time, reference_z, reference_pz, period):
    """Forward z=0 crossings; normalized time prevents SI root-tolerance errors."""
    time = np.asarray(time)/period
    z = np.asarray(reference_z)
    pz = np.asarray(reference_pz)
    spline = CubicSpline(time, z)
    brackets = np.flatnonzero((z[:-1] <= 0) & (z[1:] > 0) & (pz[1:] > 0))
    returns = np.array([brentq(spline, time[i], time[i+1], xtol=1e-13) for i in brackets])
    if len(returns) < 16 or np.max(np.abs(np.diff(returns)-1)) > 0.01:
        raise ValueError('Missing or irregular reference turns')
    return period*returns


def phase_tune(time, position, slope, returns, beta, alpha):
    """Unwrap dense phase before strobing, retaining the integer tune.

    At each reference return the same Twiss normalization applies. The phase
    between returns need not increase uniformly. No predicted tune branch is
    used in extracting the slope. Half-record estimates diagnose phase drift.
    """
    position, slope = np.asarray(position), np.asarray(slope)
    normalized = position + 1j*(alpha*position + beta*slope)
    if np.min(np.abs(normalized)) < 1e-14:
        raise ValueError('Insufficient oscillation amplitude')
    phase = -np.unwrap(np.angle(normalized))/(2*np.pi)
    if np.max(np.abs(np.diff(phase))) > 0.4:
        raise ValueError('Phase sampling too sparse')
    scale = np.mean(np.diff(returns))
    sampled = CubicSpline(np.asarray(time)/scale, phase)(np.asarray(returns)/scale)
    turn = np.arange(1, len(sampled)+1)
    fit = np.polyfit(turn, sampled, 1)
    middle = len(turn)//2
    halves = [np.polyfit(turn[:middle], sampled[:middle], 1)[0],
              np.polyfit(turn[middle:], sampled[middle:], 1)[0]]
    return dict(tune=float(fit[0]), half_tunes=[float(v) for v in halves],
                phase_residual_rms_turns=float(np.std(sampled-np.polyval(fit, turn))),
                phase=sampled, turns=turn)


def section_frequency(position):
    """Independent real-signal sinusoid fit; returns the aliased tune in [0, .5]."""
    signal = np.asarray(position, dtype=float)
    turn = np.arange(len(signal))
    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft((signal-signal.mean())*window))
    peak = np.argmax(spectrum[1:])+1
    guess = peak/len(signal)
    def residual(frequency):
        angle = 2*np.pi*frequency*turn
        model = np.column_stack([np.cos(angle), np.sin(angle), np.ones(len(turn))])
        coefficients = np.linalg.lstsq(model*window[:, None], signal*window, rcond=None)[0]
        return np.sum(((model@coefficients-signal)*window)**2)
    fit = minimize_scalar(residual, bounds=(max(1e-8, guess-1/len(signal)),
                          min(.5-1e-8, guess+1/len(signal))), method='bounded', options={'xatol':1e-13})
    if not fit.success:
        raise ValueError('Section-frequency fit failed')
    return float(fit.x)


def laboratory_coordinates(reference_r, reference_p, local_r, local_p):
    """Recover laboratory coordinates for a planar reference with vertical y axis."""
    direction = reference_p/np.linalg.norm(reference_p, axis=1)[:, None]
    if np.max(np.abs(direction[:,1])) > 1e-12:
        raise ValueError('Reference is not planar')
    horizontal = np.column_stack([direction[:,2], np.zeros(len(direction)), -direction[:,0]])
    vertical = np.array([0.,1.,0.])
    return (reference_r+horizontal*local_r[:,0,None]+vertical*local_r[:,1,None]+direction*local_r[:,2,None],
            horizontal*local_p[:,0,None]+vertical*local_p[:,1,None]+direction*local_p[:,2,None])


def matched_phase_difference(first, second, period, cells):
    """Compare interpolation orders on the same physical crossings only.

    Endpoint stencils of different orders can yield different event sets. An
    extra terminal event must not be mistaken for a bulk interpolation error.
    """
    a, b = np.asarray(first['time']), np.asarray(second['time'])
    nearest = np.argmin(abs(a[:, None] - b[None, :]), axis=1)
    keep = ((abs(a - b[nearest]) < period / (4*cells))
            & (np.asarray(first['cells']) == np.asarray(second['cells'])[nearest]))
    if keep.sum() < 32 or len(np.unique(nearest[keep])) != keep.sum():
        raise ValueError('Insufficient distinct matching section crossings')
    turns = np.asarray(first['turns'])[keep]
    difference = np.asarray(second['phase'])[nearest[keep]] - np.asarray(first['phase'])[keep]
    return float(np.polyfit(turns, difference, 1)[0]), int(keep.sum())


def cell_crossing_tune(time, lab_r, lab_p, poses, period, beta, alpha, axis, interpolation_order=5, bend_radius=None):
    """Full tune from each particle's own crossings of equivalent cell entrances.

    Six samples per revolution avoid the integer/complement ambiguity here.
    Sampling each particle at its own plane crossing avoids mixing arrival-time
    offsets with transverse momentum in the moving reference frame.
    """
    time = np.asarray(time)/period
    events = []
    for cell, (origin, theta) in enumerate(poses):
        ex = np.array([np.cos(theta), 0., -np.sin(theta)])
        ez = np.array([np.sin(theta), 0., np.cos(theta)])
        r = lab_r-origin
        x, z, pz = r@ex, r@ez, lab_p@ez
        coordinate = x if axis == 'x' else r[:,1]
        transverse_p = lab_p@ex if axis == 'x' else lab_p[:,1]
        interpolate_pz = pz.copy()
        if bend_radius is not None:
            # Interpolate small deviations from the known adjacent circular bends.
            # At z=0 these equal Cartesian x, px, pz. This avoids subtracting
            # metre-scale interpolants to obtain micrometre oscillations.
            radius = np.hypot(x+bend_radius,z)
            px = lab_p@ex
            interpolate_pz = (-px*z+pz*(x+bend_radius))/radius
            if axis == 'x':
                coordinate = radius-bend_radius
                transverse_p = (px*(x+bend_radius)+pz*z)/radius
        crossing = np.flatnonzero((z[:-1] <= 0) & (z[1:] > 0) & (pz[1:] > 0)
                                 & (abs(x[:-1]) < .1) & (abs(x[1:]) < .1))
        for i in crossing:
            # Local polynomial: global splines transmit distant hard-edge kinks
            # and momentum ratios have poles where this plane is crossed sideways.
            start = i-interpolation_order//2
            if start < 0 or start+interpolation_order >= len(time):
                continue
            stencil = slice(start, start+interpolation_order+1)
            scale = time[i+1]-time[i]
            nodes = (time[stencil]-time[i])/scale
            zs, xs, ps, pzs = [BarycentricInterpolator(nodes, values[stencil])
                               for values in (z, coordinate, transverse_p, interpolate_pz)]
            u = brentq(zs,0.,1.,xtol=1e-13)
            events.append((time[i]+u*scale,cell,float(xs(u)),float(ps(u)/pzs(u))))
    events = np.array(sorted(events))
    cells = len(poses)
    if len(events) < 32 or np.any(np.diff(events[:,1].astype(int)) % cells != 1):
        raise ValueError('Missing or incorrectly ordered cell crossings')
    coordinate, slope = events[:,2], events[:,3]
    phase = -np.unwrap(np.angle(coordinate+1j*(alpha*coordinate+beta*slope)))/(2*np.pi)
    if max(abs(np.diff(phase))) > .4:
        raise ValueError('Cell sampling too sparse for unambiguous phase')
    turns = np.arange(len(events))/cells
    fit = np.polyfit(turns,phase,1)
    middle = len(turns)//2
    independent = cells*section_frequency(coordinate)
    return dict(tune=float(fit[0]), independent_tune=independent,
                half_tunes=[float(np.polyfit(turns[:middle],phase[:middle],1)[0]),
                            float(np.polyfit(turns[middle:],phase[middle:],1)[0])],
                phase_residual_rms_turns=float(np.std(phase-np.polyval(fit,turns))),
                phase=phase, turns=turns, time=events[:,0]*period,
                cells=events[:,1].astype(int), coordinate=coordinate, slope=slope)
