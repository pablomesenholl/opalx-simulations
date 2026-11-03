"""Continuum sensitivity of the isolated ellipsoid force to source-direction lag.

This follows the one-bin quasi-static stretch/composition model, with fixed
laboratory spatial covariance and source gamma. It is not an OPALX run, PIC
discretization test, midpoint predictor, or tune-error estimate.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.constants import c, epsilon_0

from ellipsoid_model import depolarization


def cross_matrix(vector):
    """Matrix implementing vector cross a column vector."""
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def continuum_gradients(covariance, charge, beta, source_direction, test_beta):
    """Return E, B, E+v_test cross B gradients in fixed laboratory axes.

    Covariance is in m^2, charge in C; beta and test_beta are velocities/c.
    Gradients have units V/m^2, T/m and V/m^2 respectively. The source direction
    is a unit vector. The uniform ellipsoid semiaxes are sqrt(5 eig(covariance)).
    Its interior field is exact for the assumed quasi-static continuum model.
    """
    covariance = np.asarray(covariance, dtype=float)
    direction = np.asarray(source_direction, dtype=float)
    test_beta = np.asarray(test_beta, dtype=float)
    if (covariance.shape != (3, 3) or direction.shape != (3,)
            or test_beta.shape != (3,) or not np.isfinite(covariance).all()
            or not np.isfinite(direction).all() or not np.isfinite(test_beta).all()
            or not np.isfinite(charge) or charge < 0 or not np.isfinite(beta)
            or not 0 <= beta < 1 or np.linalg.norm(test_beta) >= 1
            or not np.isclose(np.linalg.norm(direction), 1., rtol=0, atol=1e-13)
            or not np.allclose(covariance, covariance.T, rtol=1e-13, atol=0)):
        raise ValueError('Finite positive covariance, physical velocities and unit direction required')
    gamma = 1 / np.sqrt(1 - beta**2)
    stretch = np.eye(3) + (gamma - 1) * np.outer(direction, direction)
    eigenvalues, vectors = np.linalg.eigh(stretch @ covariance @ stretch)
    if eigenvalues.min() <= 0:
        raise ValueError('Spatial covariance must be positive definite')
    axes = np.sqrt(5 * eigenvalues)
    density = charge / (4 * np.pi * np.prod(axes) / 3)
    rest_gradient = density / epsilon_0 * (vectors * depolarization(axes)) @ vectors.T
    rest_at_lab = rest_gradient @ stretch
    electric_conversion = gamma * np.eye(3) - (gamma - 1) * np.outer(direction, direction)
    electric = electric_conversion @ rest_at_lab
    magnetic = gamma * beta / c * cross_matrix(direction) @ rest_at_lab
    force = electric + c * cross_matrix(test_beta) @ magnetic
    return electric, magnetic, force


def analyze_covariances(covariances, p0, intervals, charge=160e-12, radius=2.):
    """Worst force-direction ratio over fixed shapes and either bend sign.

    For invertible K0, ||(Klag-K0) K0^-1||_2 bounds
    ||F_lag(r)-F_0(r)|| / ||F_0(r)|| for every nonzero r in the ellipsoid.
    Using bend curvature for all supplied shapes is an envelope of this local
    effect, not the actual integrated error along the straight and RF sections.
    """
    covariances = np.asarray(covariances, dtype=float)
    intervals = np.asarray(intervals, dtype=float)
    if (covariances.ndim != 3 or covariances.shape[1:] != (3, 3)
            or len(covariances) == 0 or intervals.ndim != 1
            or not np.isfinite(intervals).all() or np.any(intervals < 0)
            or not np.isfinite(p0) or p0 <= 0 or not np.isfinite(radius)
            or radius <= 0 or not np.isfinite(charge) or charge <= 0):
        raise ValueError('Physical shapes, intervals, momentum, radius and charge required')
    beta = p0 / np.sqrt(1 + p0**2)
    true_direction = np.array([0., 0., 1.])
    records = []
    for index, covariance in enumerate(covariances):
        _, _, centred = continuum_gradients(covariance, charge, beta, true_direction,
                                             beta * true_direction)
        for interval in intervals:
            angle = c * beta * interval / (2 * radius)
            for sign in [-1, 1]:
                lagged_direction = np.array([sign * np.sin(angle), 0., np.cos(angle)])
                _, _, lagged = continuum_gradients(covariance, charge, beta, lagged_direction,
                                                    beta * true_direction)
                difference = lagged - centred
                ratio = np.linalg.solve(centred.T, difference.T).T
                records.append(dict(sample=index, dt_s=interval, direction_sign=sign,
                    lag_angle_rad=angle, relative_force_bound=np.linalg.norm(ratio, 2),
                    relative_gradient_norm=np.linalg.norm(difference, 2) / np.linalg.norm(centred, 2),
                    centred_gradient_condition=np.linalg.cond(centred)))
    return pd.DataFrame(records)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    model_path = root / 'results/ellipsoid-match-stable-long-d1center/match-160pc.npz'
    cof_path = root / 'results/rf-ring-s3a01yyz/bare-cof.json'
    model = np.load(model_path)
    cof = json.loads(cof_path.read_text())
    p0 = np.linalg.norm(cof['momentum_mc'])
    covariances = model['covariances'][:, [0, 2, 4]][:, :, [0, 2, 4]]
    table = analyze_covariances(covariances, p0, [62.5e-12, 125e-12, 400e-12, 1e-9])
    output = Path(tempfile.mkdtemp(prefix='ellipsoid-boost-sensitivity-', dir=root / 'results'))
    for name in ['ellipsoid_boost_sensitivity.py', 'test_ellipsoid_boost_sensitivity.py',
                 'ellipsoid_model.py', 'kv_model.py']:
        shutil.copy2(Path(__file__).with_name(name), output / name)
    table['s_m'] = model['s'][table['sample'].to_numpy()]
    table.to_csv(output / 'all-shapes.csv', index=False)
    summary = table.groupby('dt_s', as_index=False).agg(
        lag_angle_rad=('lag_angle_rad', 'max'),
        maximum_relative_force_bound=('relative_force_bound', 'max'),
        maximum_relative_gradient_norm=('relative_gradient_norm', 'max'))
    summary.to_csv(output / 'summary.csv', index=False)
    metadata = dict(scope=__doc__, model=str(model_path),
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        cof=str(cof_path), cof_sha256=hashlib.sha256(cof_path.read_bytes()).hexdigest(),
        p0_mc=float(p0), beta=float(p0 / np.sqrt(1 + p0**2)), radius_m=2.,
        charge_C=160e-12, covariance_count=len(covariances),
        samples=summary.to_dict(orient='records'))
    (output / 'provenance.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(output)
    print(summary.to_string(index=False))
