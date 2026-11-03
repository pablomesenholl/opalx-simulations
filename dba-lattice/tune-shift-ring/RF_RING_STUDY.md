# RF ring calibration and bare normal-mode study

The six physical `VARIABLE_RF_CAVITY` elements are active in ordinary device
Boris tracking. A one-particle synchronous orbit was calibrated successfully.
The 13-particle bare normal-mode probe run also passes all three mode checks
against the analytic RF map within 8e-7. No charged tune result is claimed here.

## Geometry and launch

The physical ring retains each complete 1 m D1 drift. The ring's element order
starts with C1_D1, while an explicit COF `SECTION` selects its midpoint. The
particle file contains absolute lab positions [m] and momenta p/(mc) around the
freshly solved COF state. This places the finite bunch entirely inside a straight
and avoids both an artificial passive-element boundary through the bunch and
curvature-induced centroid/momentum offsets. Analytic half-D1 transfers define
the observation section without creating extra physical elements.

Each D2 is physically replaced by 0.45 m drift, 0.10 m RF cavity, and 0.45 m drift,
with explicit inherited lab poses. Circumference and magnet geometry are unchanged.
Each cavity has a prescribed uniform longitudinal electric field; this idealized
RF element omits transverse RF gap forces. The corresponding analytic model uses
the same idealization.

The intended analytic family is
`results/ellipsoid-match-stable-long-d1center/`: harmonic 4, linear cavity slope
0.001 m^-1, and the family selected by a 50 mm zero-charge rms bunch length.
The model supplies a covariance and normal-mode basis at the D1 midpoint.
**A matched covariance alone does not establish a kinetic equilibrium.** The
separate kinetic-consistency study shows that uniform spatial density with
conditional Gaussian momenta changes shape even under a matched zero-charge map.

## Measured RF calibration

Verified study: `results/rf-ring-s3a01yyz/`.
Executable SHA256:
`e5b8b517eab9b20a4342809c0c5d8b2a9a9d065fbac12d805180163af0f442bb`.
One MPI rank and one OpenMP thread were used throughout calibration.

The zero-amplitude one-particle pass measures the revolution period and all six
cavity-centre arrival times. Cubic Hermite crossing localization uses position
and velocity brackets inside the straight RF gap, with the root solved in
normalized interval coordinates. RF settings are then

```text
f = h/T
k = 2*pi*f/(beta*c)
phi_i = pi - 2*pi*f*t_centre_i (mod 2*pi)
E0 = g*beta^2*gamma*(m*c^2/e) / [k*L*sinc(k*L/2)]
```

Here g=0.001 m^-1, h=4, L=0.10 m, and sinc(u)=sin(u)/u. Positive head displacement
therefore receives positive momentum slope delta_out-delta_in = g*z to linear
order. Constant polynomial models carry amplitude [MV/m], frequency [MHz], and
phase [rad]. Live-RF passes update the measured times and period; no assumed
circular orbit timing is used as final calibration.

The third powered pass (`synchronous-3`) gives:

- Frequency: **3.390639340542066 MHz**.
- Electric amplitude: **0.05121274671324947 MV/m**.
- Measured powered revolution period: **1.1797185127384405 us**.
- Reference return position error: **3.03 pm**.
- Maximum particle/reference position difference: **20.9 pm**.
- Maximum particle/reference momentum difference: **4.42e-13 mc**.
- Final synchronous energy error: **-4.17e-7 eV**.
- Peak sampled electric field: **2496.66 V/m**.

Energy rises by approximately 62.4 eV inside a zero-crossing cavity before
returning to its entrance value. This finite-transit excursion is expected and
is separate from the near-zero net energy error. The finite ideal-linear
analytic cavity approximates that small synchronous-energy variation.

Use `synchronous-3/rf.json` for the actual validated input settings, rather than
`calibration.json`'s proposed next iteration. Six individual phases, complete
source/input snapshots, the fresh COF JSON, binary checksums, raw HDF5 files,
logs and reference summaries are preserved in the study.

## Reproduce and inspect

```sh
~/.venv-h6/bin/python -m unittest discover -s sandbox/Regression-Tests/tune-shift-ring/analysis -p test_rf_ring_study.py -v
~/.venv-h6/bin/python sandbox/Regression-Tests/tune-shift-ring/analysis/rf_ring_study.py --calibrate
~/.venv-h6/bin/python sandbox/Regression-Tests/tune-shift-ring/analysis/rf_ring_study.py --study PATH --run-probes --probe-turns 32
```

The runner checks immutable executable provenance, active cavity registration,
nonzero fields for powered calibration, particle IDs, finite values, and bounded
child lifetimes. Thirteen unit tests cover geometry, straight and diagnostic circular
coordinate transforms, phase/sinc focusing sign, SI-safe crossing localization,
HDF5 ordering, reference diagnostics, normal-mode probes, section observations,
and task-owned timeout cancellation.

The bare probe consists of the central particle plus twelve antithetic pure-mode
quadrature probes, 13 particles in total. Tiny probes test the single-particle
RF map; they are not used as a PIC bunch. The maximum initial spatial component
is 10 um. All particle states are observed at common reference crossings of
successive D1 midpoint planes and expressed in each straight's fixed axes.
`section-observations.npz` provides six-dimensional phase coordinates, event
cells and times, and source IDs for independent phase/spectral tune analysis.

## Bare normal-mode results

`bare-probes-32turns-10um` completes 32 turns in 129.63 s on one MPI rank and
one OpenMP thread. It retains all 13 particles and produces 193 equivalent-cell
observations. Only each probe's deliberately excited mode enters the comparison;
all cross-mode fits are retained as leakage diagnostics.

| Mode | Analytic tune | Mean tracking phase tune | Maximum absolute phase error |
|---|---:|---:|---:|
| x | 0.764038385533 | 0.764038979986 | 5.99e-7 |
| y | 1.689536397020 | 1.689537181932 | 7.85e-7 |
| z | 0.069128128193 | 0.069127990300 | 1.94e-7 |

An independent complex spectral fit searches the complete sampled Nyquist range,
without constraining its frequency to the prediction. Its maximum errors are
3.15e-7, 4.70e-7 and 4.74e-7 for x, y and z. The maximum phase-versus-spectral
estimator difference is 3.16e-7. Coherent spectral fractions exceed
0.9999999998. This validates the RF phase sign, longitudinal focusing, x-z
normal-mode coupling and integer vertical tune for small bare probes. The
remaining differences combine timestep error and the ideal-linear RF model's
small finite-transit approximation; this single controlled timestep does not
establish a new convergence order.

The authoritative aggregate is
`bare-probes-32turns-10um/mode-analysis/primary-summary.json`, with individual
results in `primary-mode-tunes.csv`. The generic `mode-tunes.csv` includes weak
cross-mode signals and its unfiltered aggregate must not be used for pure-mode
probe validation. `section-observations.npz`, raw HDF5, exact launch vectors and
source labels are preserved for independent reanalysis.

Runtime cavity clones currently use the prototype name `VARIABLE_RF_CAVITY`.
Registration is therefore checked using six distinct finite 0.1 m RF body
intervals in the threaded map as well as nonzero physical fields. An initial
postprocessing check for individual C1_RF...C6_RF names falsely rejected the
completed probe run; its failure history is recorded in provenance and corrected
without rerunning or altering simulation output.

## Preserved failures

`rf-ring-cwhqepw6` exposed a missing `ParallelTracker::visitVariableRFCavity`
forwarding method: parsed cavities were absent from the tracking beamline.
Its powered cases incorrectly contained no RF field and are excluded.
`rf-ring-htx9ykln` repeated this result because a library target was rebuilt but
the `opalx_exe` executable had not yet been relinked. Both failures are retained
with exact hashes and inputs. The successful study above uses the relinked
executable and demonstrates nonzero RF action on host reference and device
particle tracking.
