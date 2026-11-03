# Cyclotron guard for bare-ring/device tracking changes

Date: 2026-09-16. Scope: establish that the bare DBA tracking/diagnostic revision
does not alter existing mapped cyclotron coasting, localized RF-gap acceleration,
or ordinary PIC sequencing. This is a before/after compatibility check, not a
new cyclotron convergence study.

## Frozen evidence

Task directory: `/Users/adelmann/git/opalx/tmp/cyclotron-guard-20260916-1789562042`.

The pre-change executable was copied from `omp-build/src/opalx` before any rebuild:

```
opalx-before
SHA256 2868bdde8d1088e4b4cd4f9912e6ae898937983c2b1b609b51c8521d2ec6e68e
```

`baseline.json` records the original path, timestamp and hash. Frozen copies of
all input decks, distributions, magnetic maps and RF maps are under `fixtures/`;
each case's `manifest.json` hashes its exact inputs and records modifications.
The live repository inputs are not used when repeating the runs.

`run_guard.py` runs one MPI rank and one OpenMP thread with all `OPALX_TEST_*`
environment variables removed. It refuses existing output directories, verifies
input hashes, limits each owned child to 180 seconds, terminates/reaps its own
process group on timeout/interruption, and records PID, runtime, return code,
binary hash and coverage markers in each `status.json`.

## Baseline results

All four cases completed normally, without the analytic-ring boundary controller.

| Case | Coverage and input difference | Final time [ns] | Energy [MeV] | Particles | Steps | Wall [s] |
|---|---|---:|---:|---:|---:|---:|
| coasting | Original 72 MeV input, one directed return | 118.669954781525 | 72.000000290492 | 1 | 721 | 1.865 |
| gap-stop | Existing acceleration deck, energy target reduced to 73 MeV; MAXSTEPS=20000; diagnostic sampling every 100 steps | 60.379048453759 | 73.273098424852 | 1 | 1468 | 0.248 |
| pic-midpoint | Original 262144 particles, 32 cubed OPEN mesh, single bin and charge; duration reduced from 720 to 12 steps | 1.974333662389 | 72.000880771450 | 262144 | 12 | 2.397 |
| pic-prestep | Same as above, historical PRESTEP field timing | 1.974333662389 | 72.000880173301 | 262144 | 12 | 2.461 |

The gap check encounters RF1, RF3, the third-harmonic RF2, and RF4, where it stops
after the complete kick. It covers harmonic and fundamental gap kicks plus the
physical energy-stop event; it does not cover the full 590 MeV acceleration.
The PIC checks preserve the complete original population and mesh but cover a
short initial arc, not a complete turn or an equilibrium calculation.

`comparison.json` contains exact final reference coordinates and momenta, charge,
step count and timing. `compare_guard.py` checks the directed-turn/energy-stop
markers and compares every HDF5 particle dataset, every step attribute (including
the reference state and times), and all numeric SDDS records before/after.
Timestamp/revision file headers and wall-clock timing are not physical outputs
and are excluded from equality.

Initial post-change executable frozen as `opalx-after`:

```
SHA256 07b0b6e80368de78bd0b796930dbb410060226a0e3bd86450ccbbd2399629b52
```

After the passive recorder's equality-only correction, the next executable was
separately frozen as `opalx-after-probe-fix`:

```
SHA256 fccb6e76ec0a4fc9610a5aff8873641a972dae9e85ca6067d3d6a4965e326252
```

The final executable also includes prepared-occurrence ordering for callbacks
only when `spatialRing_m && bareTracking_m`. Source inspection confirms that
cyclotron and charged tracking preserve the existing candidate iteration order.
This executable is separately frozen as `opalx-after-order-fix`:

```
SHA256 c26084e30f195c5a5bf1ff4f5e23ea3abd50094058c7f1ba9c8f9f91f158587c
```

All four identical frozen cases completed successfully with all three candidate
executables. The final repetition lives under `after-order-fix/`; earlier
`before/`, `after/`, and `after-probe-fix/` evidence was preserved untouched.
All recorded physical outputs are **exactly equal**, including NaN-aware
comparison of single-particle undefined statistics. No numerical tolerance was
introduced:

| Case | HDF5 datasets | HDF5 step attributes | Numeric SDDS values | Differences |
|---|---:|---:|---:|---:|
| coasting | 7220 | 19494 | 33934 | 0 |
| gap-stop | 150 | 405 | 705 | 0 |
| pic-midpoint | 10 | 27 | 564 | 0 |
| pic-prestep | 10 | 27 | 564 | 0 |

In particular, the two PIC endpoint datasets compare all 262144 particles each.
Coasting retains one directed return and the gap check retains the identical
complete-kick energy stop. No case enters the analytic-ring boundary controller.
`comparison-after-order-fix.json` is the final machine-readable evidence,
reproduced by `compare_guard.py after-order-fix` with exit status zero.
`comparison.json` and `comparison-after-probe-fix.json` preserve earlier
successful comparisons.

## Analytic OPEN dispatch smoke

An additional independent case under `fixtures/open-smoke` is derived from the
saved full-population DBA one-cell input. It retains the OPEN solver, 32 cubed
mesh, single GAMMAZ bin, 160 pC charge and 62.5 ps step; it uses the first 64 saved
particles and stops after four steps. The manifest records input hashes and the
limited purpose. This sparse bunch is only a dispatch smoke, not a physical
space-charge/tune-accuracy result.

The rebuilt executable completed at 250 ps with all 64 particles and finite phase
space. Its log explicitly selects spatial device field evaluation with ordinary
fixed Boris/PIC steps and reports:

```
Boris boundary trials=0, rejected=0
Total FieldSolver calls: 4
```

Thus OPEN no longer activates repeated boundary/PIC trials in this compatible
analytic-ring case. The smoke was also repeated with the final executable;
see `after-order-fix/open-smoke/{run.log,verification.json}`. The stored endpoint
time is one floating-point ulp above the requested 250 ps; its duration check
allows four ulps for four clock increments. This clock-only check does not alter
the exact baseline comparisons or introduce any physics tolerance.

## Isolation requirements and test coverage

`ParallelTracker::execute()` allows the analytic descriptor/controller only when
the lattice supports it; `DeviceExternalField::Builder` rejects CYCLOTRONSECTOR.
Cyclotron RF gaps additionally exclude the path via `hasCyclotronGaps()`.
`SINGLEGAP` retains its existing NONE-solver, one-particle/one-rank restrictions.
The parent task removed the newly introduced unconditional spin rejection under
TURNS, which otherwise changed unrelated cyclotron behavior. The existing
spin restrictions within actually unsupported tracking modes remain scoped there.

Added unit test
`DeviceExternalFieldTest.CyclotronSectorRetainsItsMappedTrackingPath` in
`unit_tests/Algorithms/TestDeviceExternalField.cpp`: constructs a real sector with
a small synthetic PSI map; verifies descriptor eligibility rejects it and build
throws without mutating the lattice; confirms its ordinary mapped scalar field
remains available. This test launches no OPALX executable and changes no API.
This newly added test passed on MPI1 and MPI2 after rebuilding. The MPI2 log
`tmp/bare-ring-device-field-mpi2.log` records all eight descriptor tests passing on
each rank, including this case.

Baseline existing-binary validation:

- MPI1 CTest: TestCyclotronSector, TestCyclotronRF, TestQuaternion,
  TestSpectralTunes, TestOneTurnMap and TestTrackReferenceStep: 6/6 pass, 2.65 s.
- MPI2 unit-only checks: TestCyclotronSector (7 cases per rank) and TestCyclotronRF
  (6 cases per rank) pass. Scientific runs remain on one rank.

Fresh rebuilt validation performed by the parent task and inspected here:

- `tmp/bare-ring-ctest.log`: all 18 focused existing suites pass, including the
  descriptor, sector, RF, reference, quaternion, binning and PIC algorithm suites.
- `tmp/bare-ring-{cyclotron-sector,cyclotron-rf,reference}-mpi2.log`: freshly
  linked TestCyclotronSector (7), TestCyclotronRF (6), and TestTrackReferenceStep
  (11) pass on each rank. These are unit-only MPI checks.
- After the final ordering correction, `tmp/bare-ring-order-ctest.log` and
  `tmp/bare-ring-order-mpi2.log` show TestDeviceExternalField passing on MPI1
  and all 10 cases passing on each MPI2 rank, including reversed-allocation
  ordering and mapped-cyclotron eligibility checks.

The three existing cyclotron regression package `results/` directories were empty
when inspected; their README result descriptions are historical and were not
used as current proof. Fresh baseline evidence above is retained in this task.

## Reproduction

The completed final run used the following invocation (the runner deliberately
refuses overwriting these now-existing output directories). The comparison
command can be rerun directly:

```sh
/Users/adelmann/.venv-h6/bin/python tmp/cyclotron-guard-20260916-1789562042/run_guard.py after-order-fix coasting gap-stop pic-midpoint pic-prestep open-smoke
/Users/adelmann/.venv-h6/bin/python tmp/cyclotron-guard-20260916-1789562042/compare_guard.py after-order-fix
```

The original fixtures remain at
`sandbox/Regression-Tests/ring-cyclotron-{coasting,accelera,spaccha}-1`.
No historical results were overwritten; no physics tolerance was changed.
