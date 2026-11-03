# Source-direction lag in the field composer

For this ring's nominal co-moving test velocity, the **composition-only**
worst-case relative force errors are 0.912 ppm at 62.5 ps and 5.836 ppm at
400 ps. These are attained spectral-norm bounds, not conservative sums of
separate electric and magnetic errors. They do not bound the total space-charge
error, PIC noise, or an accumulated tune shift.

The calculation holds gamma, source speed, test velocity and the rest electric
vector E' fixed in one common Cartesian frame. Only the source direction used
in the electric/magnetic conversion changes. No OPALX source or simulation was
changed; this is exact algebra and a small independent Python calculation.

## Map and exact norms

`src/SpaceCharge/CartesianPIC3D/RelativisticFieldComposer.cpp:28–52` implements
the direct-source conversion, with magneticSign=+1,

    E = gamma E' − (gamma−1)(n·E')n,
    B = gamma beta_source (n×E') / c.

For test velocity v_test=c w, the triple-product identity gives

    E + v_test×B = L(n,w) E',
    L(n,w) = gamma(1−beta_source w·n) I
             − (gamma−1) n n^T + gamma beta_source n w^T.

E and v_test×B are in V/m. L is dimensionless; multiplying by test charge gives
force, without changing the relative bound. This general map permits different
source and test velocities. The table below specializes to a cold nominal
reference: beta_source=beta, gamma=(1−beta²)^−1/2, true source direction u=z and
w=beta u. Then

    L0 = diag(1/gamma, 1/gamma, 1).

Let the lagged direction be n=(sin theta,0,cos theta), b=gamma−1 and
eta=gamma beta². Writing s=sin theta, k=cos theta, the exact difference is

    D = L(n,beta u)−L0
      = [[eta(1−k)−b s²,  0, s(eta−b k)],
         [0,          eta(1−k),      0],
         [−b s k,           0,   b s²]].

Both signs of theta have the same norms. The tight absolute operator and
force-relative bounds are respectively

    ||delta F/q|| / ||E'|| <= ||D||_2,
    ||delta F|| / ||F0|| <= ||D L0^−1||_2.

They are attained by the corresponding right singular vectors. The latter is
not ||D||/||L0||. E' can point in any direction; L0 is invertible, so the
denominator is nonzero whenever E' is nonzero.

The y component is decoupled. Each norm is therefore the larger of its y entry
and a two-by-two singular norm in the xz plane. For a block A, its largest
singular value squared is

    (T + sqrt(T²−4 det(A)²))/2,   T=sum(A_ij²).

The code evaluates 1−cos(theta) as 2 sin²(theta/2), avoiding subtraction loss.
It independently checks this invariant formula against SVD and constructs a
field direction attaining the force-relative bound.

## Ring values

Use beta=0.0728541, gamma=1.002664471348018, c=299792458 m/s, and bend radius
rho=2 m. The full-body half-step source-direction lag is theta=beta c H/(2rho).
The supplied beta is the requested rounded nominal value; the associated
continuum study uses the slightly more precise COF value.

| H (ps) | Lag (mrad) | Exact ||D||₂ | Exact relative force bound (ppm) |
|---:|---:|---:|---:|
| 62.5 | 0.3412673 | 9.093069e−7 | 0.911725 |
| 125 | 0.6825347 | 1.818672e−6 | 1.823479 |
| 400 | 2.1841110 | 5.821771e−6 | 5.836250 |
| 1000 | 5.4602774 | 1.457353e−5 | 14.604137 |

For small theta,

    ||D||_2 = (gamma−1)|theta| + higher orders,
    ||D L0^−1||_2 = gamma(gamma−1)|theta| + higher orders.

The leading error couples transverse and longitudinal components. Restricting
both E' and the observed force to the transverse plane suppresses the first-order
term: the relative norm starts as gamma² beta² theta²/2. Its exact values are
3.1073e−10, 1.2429e−9, 1.2727e−8 and 7.9546e−8 for the same four steps. This
restricted result is **not** a bound for a finite ellipsoid with longitudinal
field or for its full three-dimensional force.

The electric conversion alone has ||delta E/E'||_op=(gamma−1)|sin theta|; the
magnetic conversion has c||delta B/E'||_op=2 gamma beta |sin(theta/2)|. The total
Lorentz map above includes their direction-dependent cancellation exactly.

## Scope and independent checks

These bounds exclude the change of E' caused by rotating/stretching the source
coordinates, rest-density normalization, the Poisson solution, covariance or
midpoint-trajectory errors, changes in source gamma, and all mesh/gather/noise
effects. They also assume the nominal test velocity. For an individual particle
with different w, use the general L and compute
||(L_lag−L_true)L_true^−1|| directly; the table is not a bound for arbitrary
velocity spread. OPALX's bin-averaged gamma and mean momentum need not obey the
cold reference relation exactly.

Root's separate `analysis/ellipsoid_boost_sensitivity.py` includes the continuum
stretch and field change for fixed lab covariance using K=L G' S. I inspected
its dimensions, signs and frame operations and found no inconsistency with the
one-bin quasi-static normalization. Its results in
`results/ellipsoid-boost-sensitivity-r0jlowih/` are a broader, still conditional
continuum calculation and must not be conflated with this composition-only
table. Neither study establishes the total numerical SC error or tune accuracy.

Reproducible script, assumptions, CSV and source snapshots:
`results/source-boost-audit/`. Run without OPALX:

```sh
cd sandbox/Regression-Tests/tune-shift-ring/analysis
~/.venv-h6/bin/python -m unittest test_source_boost_audit.py
~/.venv-h6/bin/python source_boost_audit.py ../results/source-boost-audit
```

Five regression checks pass: direct cross-product composition for independent
velocities, the closed-form matrix and zero-lag limit, SVD agreement and
attainment, small-angle orders, and reproducible output. One initially
over-strict floating-point identity assertion was corrected from 2e−16 to two
binary64 epsilons after observing a 2.22e−16 difference between equivalent
beta/gamma expressions. This is a documented algebra-test roundoff allowance;
no physics tolerance, algorithm or numerical result was relaxed.
