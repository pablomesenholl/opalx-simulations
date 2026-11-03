# Large uniform-ellipsoid initial-force benchmark

The isolated-bunch PIC force agrees with the exact uniform-ellipsoid model.
This is an initial-field validation, **not a ring tune or kinetic-equilibrium
result**. A bounded ring probe separately identifies a serious performance
limitation in synchronized particle-boundary refinement.

## Reproduce

From the repository root, using the OPALX Python environment:

```sh
~/.venv-h6/bin/python -m unittest discover -s sandbox/Regression-Tests/tune-shift-ring/analysis -p test_large_ellipsoid_benchmark.py -v
~/.venv-h6/bin/python sandbox/Regression-Tests/tune-shift-ring/analysis/large_ellipsoid_benchmark.py --run all --analyze
```

The runner prints its new result directory immediately. `--prepare-only` writes
all inputs before tracking; use `--study PATH --run all --analyze` afterward.
`--study PATH --ring-probe` launches a separate, automatically bounded 90 s ring
performance test. A prepared study refuses a changed executable or particle file.

Verified result directory:
[large-ellipsoid-rjg8zxr_](results/large-ellipsoid-rjg8zxr_/summary.json).
It includes a fresh DBA COF solve, input files, particle files, raw HDF5 fields,
source snapshots, binary hashes, timings, comparison CSV, sampled particle-field
CSV, JSON tensors, and publication-quality PNG/PDF convergence figures.

## Physical sample and field comparison

The bunch has 41.13101073 pC physical charge, independent of macroparticle count,
with semiaxes approximately 13.44224413, 5.54556635, and 50 mm. The reference
momentum comes from the separately executed DBA COF, p0/(mc)=0.073048230942655.

Spatial samples fill the ellipsoid uniformly in volume: a scrambled Sobol cube
is mapped with radius u^(1/3), uniform cos(theta), and uniform azimuth. Antithetic
partners set the centroid to zero. Tests verify the radial-volume CDF and full
covariance. Spatial points are not whitened after generation because that would
change the requested surface. Powers of two, 131072 and 524288, preserve Sobol
balance and correspond to the proposed order of 100000-particle starting sample
and a fourfold particle-count refinement.

The generated `ring-launch.txt` includes the transverse emittance and initial
covariance of the earlier fixed-length ellipsoid model. Conditional momenta are
Gaussian with the prescribed position-momentum correlations. This realizes the
specified moments; it is not an exact stationary Vlasov distribution. FROMFILE
uses absolute positions [m] and momenta p/(mc), so neither INITIALORBIT nor an
explicit BEAM.PC is set. OPALX rejects either combination with FROMFILE.

For the force test only, momentum spreads are zero. The bunch starts in a
field-free straight drift and receives one 1 ps ordinary Boris/PIC step. This
isolates the exact common-velocity Lorentz transformation and avoids ring
curvature, external fields and element-boundary retries. The solver is OPEN,
with one fixed GAMMAZ bin, no repartition, one MPI rank and one OpenMP thread.
The compared force per unit test charge is **E + v0 × B** [V/m]. The analytic
interior field uses the exact ellipsoid depolarization tensor in the rest frame
and the same boost. First-half-drift displacement is common to every particle,
so centered midpoint positions equal the input centered positions exactly.

Fits include all nine force-gradient coefficients and a free intercept. Both
all-particle and core results are reported; the core is ellipsoidal radius
r/a < 0.8. Whole-bunch RMS errors include field smoothing at the sharp density
surface and discrete-particle field noise. No particles are removed from the
all-particle statistics.

## Results

| Macroparticles | Grid | Whole-bunch force RMS error x/y/z | Maximum core gradient error | End-to-end time |
|---:|---:|---:|---:|---:|
| 131072 | 32³ | 1.60 / 1.62 / 1.82 % | 0.0112 % | 0.68 s |
| 524288 | 32³ | 1.59 / 1.62 / 1.81 % | 0.0120 % | 1.49 s |
| 131072 | 64³ | 1.33 / 1.37 / 1.44 % | 0.0422 % | 2.64 s |
| 524288 | 64³ | 0.648 / 0.659 / 0.718 % | 0.0020 % | 3.45 s |

The all-particle gradient bias improves from approximately -0.66% at 32³ to
-0.166% at 64³ with 524288 particles, consistent with a factor-four reduction
when mesh spacing halves. At 64³, increasing particle count fourfold reduces
core particle-wise force noise from 1.6–1.8% to 0.3–0.4%. This demonstrates why
mesh refinement alone is insufficient. Two grid spacings measure sensitivity;
they do not establish a complete asymptotic convergence order.

All particles survive, charge sums remain at the requested physical charge,
all output coordinates/fields are finite, and the maximum relative size change
is below 1e-11. Relativistic magnetic composition B=v0×E/c² agrees within
1.4e-11 relative. The physical momentum kick Δp/(mc)=e(E+v0×B)Δt/(m c) agrees
within 3.4e-8 relative, including longitudinal subtraction roundoff.

**Preexisting output metadata issue:** HDF5 ExUnit/EyUnit/EzUnit say `MV/m`, but
`H5PartWrapperForPT.cpp` writes the raw SI `pc->E` values without conversion.
Both the exact field comparison and independent momentum-kick check verify that
the actual values are **V/m**. The analysis uses V/m and records this discrepancy
in every case JSON. Source snapshots preserve the evidence.

![PIC force convergence](results/large-ellipsoid-rjg8zxr_/force-convergence.png)

## Bounded ring-performance result

The same 131072-particle finite-emittance launch was used for a 32³, 41.13 pC
ring probe requesting one nominal 62.5 ps step. The synchronized ordinary
Boris/PIC boundary controller was confirmed active. After 90.02 s, it had only
reached accepted substep 178 at 158.310 fs, approximately 0.2533% of the first
nominal interval. The task-owned process was terminated by its declared timeout;
its PID, return code, exact input and last progress lines are recorded in
`ring-performance-probe/provenance.json`.

The repeated global subdivisions resolve each individual particle's field-face
crossing down toward coordinate roundoff, repeatedly solving PIC. Therefore a
useful one-cell or many-turn large-bunch run is not feasible with this scheduling
policy. The short cold-drift force validation is complete. One-cell evolution,
stationary-bunch validation and tune-versus-charge measurements still require a
practical, explicitly convergence-controlled ring integration policy.

An earlier preparation, `large-ellipsoid-bg8i_vm4`, failed because its test drift
lacked an explicit placement. It is preserved as a setup failure, excluded from
all results above. The corrected input specifies the full lab pose.
