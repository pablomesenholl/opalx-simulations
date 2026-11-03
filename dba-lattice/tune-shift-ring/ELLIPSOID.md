# Isolated uniform ellipsoid: analytic field and rms model

The coasting baseline and its predicted tune shifts are specified in the
[LaTeX note](../TeX/main.tex), `sec:kv` and `sec:fft2d5`. This report records the
separate isolated-bunch extension and its historical validation evidence.

## What was established with KV

The adapted proton lattice's OPALX COF map agrees with the analytic bare tunes
in the LaTeX note within 7e-10. Its closed-orbit residual is below
1e-13 m. The generated coasting KV sample matches the prescribed transverse
second moments and shell invariant to roundoff.

**The coasting space-charge shifts have not been validated by OPALX tracking.** A new
one-rank proton PARALLEL baseline with FS=NONE and DT=1e-10 s completes, but ends
at x=40.7769 mm, z=−256.6681 mm relative to its moving reference. The largest
saved |x| is 83.0559 mm. The final saved path is 25.7725122 m (the step/dump
position overshoots the circumference). Thus this is a near-one-turn diagnostic,
not an exact return-plane residual. The particle should follow its reference
throughout when launched on it; these offsets rule out using this run as a
precision tune benchmark.

The diagnostic is in `results/proton-track-baseline-vac8lq4z/`, including input,
binary provenance, log, HDF5 and `baseline-check.json`. Two earlier setup attempts
aborted: FROMFILE disallows PC on BEAM; the successful input obtains momentum
from the particle file. No production tracker changes were made.

## Reproduce the ellipsoid calculation

Run `./run_ellipsoid.sh` from this experiment, or use its absolute path.
It uses `~/.venv-h6`, runs all unit tests and writes a new `results/ellipsoid-*/`
directory with input/source snapshots, provenance, CSVs, summary and PNG/PDF.
The final verified run is `results/ellipsoid-fhbk437d/`.

`config.json` supplies the baseline lattice and transverse rms emittances
specified in the LaTeX note. `ellipsoid-config.json` supplies the full length 0.10 m
(semiaxis 0.05 m), initial rms delta=1e-4, target vertical shift −0.2 and RF
harmonic 20. All computations are serial; no OPALX space-charge run is performed.

## Exact internal field, approximate evolution

A uniformly filled **spatial** ellipsoid with semiaxes a,b,l has rms sizes
a/sqrt(5), b/sqrt(5), l/sqrt(5). Its projection is not the transverse KV uniform
ellipse: the transverse projected density tapers toward the edge. Do not reuse
the KV edge-to-rms factor of two. Uniform spatial density alone does not specify
an equilibrium six-dimensional phase-space distribution at finite emittance.

In the bunch rest frame, define semiaxes a_i and density
rho'=Q/(4 pi a_1 a_2 a_3/3). The internal field is

```text
E'_i = rho' A_i r'_i / epsilon_0
A_i = (a_1 a_2 a_3 / 2) integral_0^infinity
      du / [(a_i^2+u) sqrt((a_1^2+u)(a_2^2+u)(a_3^2+u))]
sum_i A_i = 1
```

The code evaluates these coefficients using Carlson R_D, with axis scaling for
numerical conditioning. Independent quadrature, the sphere/Gauss-law limit and
the long-ellipsoid/coasting-cylinder limit validate the formula. See
[NIST DLMF: ellipsoid depolarization factors](https://dlmf.nist.gov/19.33).

For lab spatial covariance C_r, the rest covariance is D C_r D with
D=diag(1,1,gamma). Diagonalize it to find squared semiaxes 5 times its eigenvalues.
Rotate the factors back into the original frame to obtain tensor A. In the
paraxial variables (x,px/p0,y,py/p0,z,delta), the symmetric self-force matrix is

```text
S = [e rho'/(epsilon_0 m_p c^2 beta^2 gamma^2)] D A D
(px/p0, py/p0, delta)'_SC = S (x,y,z)
```

This includes direct transverse electric/magnetic cancellation and longitudinal
force. It assumes an isolated bunch, locally straight geometry and fixed reference
energy. It excludes images, radiation, nonlinear magnetic fields and RF transverse
forces. The internal field is exact for the assumed ellipsoid, but repeatedly
reconstructing a uniform ellipsoid from moments is an **rms closure approximation**,
not an exact finite-emittance Vlasov solution or particle simulation.

## Conditional tune-shift benchmark

The first model holds l fixed and neglects momentum dispersion. It solves the
periodic transverse equations with effective emittances E_x=5 epsilon_x,
E_y=5 epsilon_y:

```text
a'' + (kx-Sxx) a - E_x^2/a^3 = 0
b'' + (ky-Syy) b - E_y^2/b^3 = 0
Qx = cells/(2 pi) integral_cell E_x/a^2 ds
Qy = cells/(2 pi) integral_cell E_y/b^2 ds
```

Solving for Delta Qy=−0.2 gives:

| Quantity | Value |
|---|---:|
| Bunch charge | 41.13101073 pC |
| Protons | 2.56719577e8 |
| Full length | 100 mm |
| Rms length | 22.36067977 mm |
| Initial transverse semiaxes a,b | 13.44224413 / 5.54556635 mm |
| Peak current | 13.47520619 mA |
| One-bunch average circulating current | 0.03486510 mA |
| Conditional Qx,Qy | 0.489215965 / 1.489536397 |
| Conditional Delta Qx,Delta Qy | −0.280891604 / −0.200000000 |

Peak line charge is 3Q/(4l); it is not Q/circumference. These tunes apply only to
the constrained transverse model. Once bunch length, tilt and energy spread evolve,
there is no basis for assigning these same stationary tunes to the actual bunch.

## Coupled longitudinal evolution and trial cavities

The second model propagates the full covariance via C'=F C+C F^T. It retains
curvature terms x''=−kx x+h delta and z'=delta/gamma^2−h x, with h=1/rho in a
bend, plus the self-force tensor. Spatial x-z tilt is retained. At zero charge,
this reproduces the exact linear 6D transfer map. Dispersion cannot be omitted
once longitudinal momentum spread exists; see
[Venturini and Reiser, PRL 81, 96 (1998)](https://doi.org/10.1103/PhysRevLett.81.96).

Initial conditions use the constrained transverse match, no z-delta chirp and
sigma_delta=1e-4. They are **not a fully matched 6D ring equilibrium**. Even the
zero-charge control therefore changes shape. The calculated momentum compaction
is alpha_p=2.19631752 and eta=alpha_p−gamma^−2=+1.20162524. In the adopted z-head
convention, the closed-dispersion R56=−30.96152129 m per turn. This strong
longitudinal/transverse coupling is why a free-bunch straight-drift expansion
estimate does not predict its behaviour in this ring.

An exploratory thin buncher at each of six cell exits applies delta += g z,
with zero synchronous energy gain. The magnitude |g| is the integrated
longitudinal self-force coefficient of the fixed-length cell, and both signs
are tested. At harmonic 20 this corresponds to **11.564 kV per cavity** at
**16.9532 MHz**. Cavities are idealized and independently phased; this is not
a single-cavity design or an optimized RF match. Initial maximum RF phase is
0.244 rad, giving about 1% deviation between sine and its linear approximation
at the edge. The trials neglect RF nonlinearities and transverse cavity fields.

First-turn rms length ranges:

| Case | Minimum / maximum sigma_z [mm] | Final sigma_z [mm] |
|---|---:|---:|
| No SC, no RF; same initial covariance | 22.361 / 24.639 | 24.210 |
| SC, no RF | 18.149 / 27.600 | 18.149 |
| SC, RF opposing the integrated longitudinal self-force | 22.361 / 25.790 | 24.878 |
| SC, opposite RF polarity | 17.945 / 29.643 | 24.681 |

The cancellation trial reduces the first-turn length range, but still gives a
15% increase at its maximum and does not close periodically. A true RF-matched
solution requires joint 6D matching, including dispersion and longitudinal
emittance, and stability tests over multiple turns. No cavity stability or
long-term performance claim is made here. The RF/slip-factor relation follows
the usual stationary-ring linearization; see
[Single-rf Longitudinal Stability](https://s3.cern.ch/inspire-prod-files-d/d45247da42345e3a46ca788286e466ff).

## Verification and limits

18 unit tests pass. Frozen-length tune changes under 100-fold tighter ODE
tolerance are below 3e-13. Coupled final rms sizes change by less than 2e-11
relative under a tenfold tighter tolerance. The largest observed relative
covariance-determinant drift is 5.1e-7; exact Hamiltonian linear moment evolution
would preserve it. These numerical checks do not remove the physical closure
approximation. The configurable 0.2 m maximum spatial semiaxis is a local-model
validity stop, not a beam-pipe aperture; no case reached it within the first turn.

Next steps are separate: repair/validate the OPALX PARALLEL bare reference
tracking before testing predicted SC tunes, and solve a genuinely periodic
coupled ellipsoid/RF moment match before treating the frozen-length prediction
as a stationary bunch tune. No production source or tolerance was changed.
