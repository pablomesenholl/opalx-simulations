# Source-direction lag in the continuum ellipsoid model

The source-direction lag has a small coefficient for this low-beta ring. Across
all 98 spatial covariances of the 160 pC matched-moment cell, the largest local
relative force bound is 1.66e-6 at 62.5 ps and 1.06e-5 at 400 ps. This supports
retaining the current quasistatic source approximation for the initial proposed
implementation tests. It does not establish the PIC error or a tune tolerance.

Results and frozen sources:
[ellipsoid-boost-sensitivity-r0jlowih](results/ellipsoid-boost-sensitivity-r0jlowih).
Seven pure continuum tests pass. No OPALX execution or production change occurred.
The separate [composition-only audit](SOURCE_BOOST_AUDIT.md) holds the rest-frame
field fixed; this calculation also includes the changed rest-frame ellipsoid.

## Model and bound

Hold the laboratory spatial covariance Sigma, bunch charge Q and source beta
fixed. For source unit direction u, define

    gamma = 1/sqrt(1-beta^2),
    S = I + (gamma-1) u u^T,
    Sigma' = S Sigma S^T.

The assumed uniform rest ellipsoid has semiaxes sqrt(5 eig(Sigma')) and volume
V'=gamma V. Its density is Q/V'. Its exact interior electric-field gradient G'
comes from the usual ellipsoid depolarization factors. In fixed laboratory axes,

    E(r) = [gamma I-(gamma-1)u u^T] G' S r,
    B(r) = (gamma beta/c) [u]_cross G' S r,
    F(r)/q = E(r) + c [w]_cross B(r) = K r,

where w=v_test/c and [a]_cross b=a cross b. E-gradient and force-per-charge
gradient units are V/m^2; B-gradient units are T/m. This follows the current
one-bin equal-time stretch/field-composition model; it is not a calculation of
retarded fields from an accelerating extended bunch.

The centred direction is +z and w=beta zhat. The alternative source direction
is rotated by either sign of

    delta_theta = beta c H / (2 rho),   rho=2 m.

H is the common interval in seconds. We use p0 from the validated RF study's
bare-cof.json, giving beta=0.072854113175..., and the spatial blocks of the
160 pC analytic cell covariances. Applying the bend's curvature to every shape,
including shapes occurring in straights, makes this a local envelope of the
direction-lag effect. It is not the actual error integrated around the lattice.

For centred gradient K0 and lagged gradient Klag, the reported bound is

    ||(Klag-K0) K0^-1||_2.

It bounds ||F_lag(r)-F_0(r)||/||F_0(r)|| for every nonzero displacement within
the assumed ellipsoid. Charge cancels from this relative bound. We maximize
over both direction signs and the 98 supplied shapes.

| H | Direction lag [rad] | Maximum relative force bound | Relative gradient norm |
|---:|---:|---:|---:|
| 62.5 ps | 0.0003412674 | 1.656393e-6 | 5.306916e-7 |
| 125 ps | 0.0006825348 | 3.312786e-6 | 1.061383e-6 |
| 400 ps | 0.0021841114 | 1.060093e-5 | 3.396422e-6 |
| 1 ns | 0.0054602784 | 2.650251e-5 | 8.490945e-6 |

The error is approximately linear in H. Its small coefficient does not make
this source-direction approximation formally second order. A force bound of
this kind cannot be converted directly into a coupled-ring tune bound.

## Verification and limits

The pure tests check the direct Lorentz-force sign and units, the existing
aligned ellipsoid oracle, the stationary sphere/Gauss-law result, arbitrary
rotations, charge/size scaling, zero lag, and coverage of sampled force
directions by the spectral-norm bound. An independent source review confirmed
the stretch, density normalization, matrix ordering and force-bound definition.

The current source uses unchanged P_n after the first position drift:
ParallelTracker::makeSpaceChargeFrameTransforms aligns the solve to its mean;
ParticleBinTraversal computes mean momentum and gamma; CartesianPIC3DAlgorithm
stretches mesh z by gamma; ParticleMeshFieldTransfer divides density by gamma;
RelativisticFieldComposer adds the direct electric and magnetic fields.

This test excludes changing source gamma/energy, actual covariance evolution
within H, particle velocity spread, mesh/interpolation noise, nonlinear density
deformation, grazing boundaries, RF timing, and full tracking. The ellipsoid's
spatial covariance is the local Cartesian moment closure, not a verified
stationary kinetic distribution. Full H/N/mesh and individual-tune convergence
remain mandatory before completing point 5.

Reproduce from the study analysis directory using the OPALX Python environment:

```sh
/Users/adelmann/.venv-h6/bin/python -m unittest test_ellipsoid_boost_sensitivity.py
/Users/adelmann/.venv-h6/bin/python ellipsoid_boost_sensitivity.py
```
