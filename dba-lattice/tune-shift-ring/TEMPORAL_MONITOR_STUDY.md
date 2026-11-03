# Temporal D1 monitor diagnostics

Existing `TYPE=TEMPORAL` monitors provide accurate simultaneous whole-bunch snapshots at the six field-free D1-centre sections for bare tracking. Their HDF5 **group TIME** must be used; the per-particle `time` dataset contains a different integration time. Final directed return can produce a duplicate event. The supplied offline reducer handles both details. No C++ changes or new public input APIs were introduced.

## Evidence

Both tests used the calibrated six-cavity RF lattice, thirteen pure-mode probes, two turns, one MPI rank and one OpenMP thread. Physical magnet/RF poses and the original25.766370614359 m nominal circumference were preserved. Binary SHA256: `e5b8b517eab9b20a4342809c0c5d8b2a9a9d065fbac12d805180163af0f442bb`.

| Test | Boundary controller | Time step | Runtime | Maximum position difference | Maximum momentum difference | Primary-mode phase difference |
|---|---|---:|---:|---:|---:|---:|
| `results/temporal-monitor-pkjciq1y` | enabled |62.5 ps|15.79 s|4.60e−15 m|2.23e−16 p0|1.61e−9 rad|
| `results/temporal-monitor-fbf4cfcq` | ordinary fixed step (`REPARTFREQ=10`) |100 ps|4.24 s|1.85e−14 m|1.51e−15 p0|9.21e−10 rad|

Every monitor snapshot was compared particle-by-particle, in ID order, with dense H5 output from the **same run**, interpolated in lab coordinates within the field-free D1. Group times agree with independent reference plane crossings to≤4.3e−22 s. The first test has12 raw passages, the fixed-step test13 raw passages but12 unique events. These are output-validation comparisons; the fixed-step test is not itself a tune-convergence claim. The RF calibration/bare-tune study remains documented separately in `RF_RING_STUDY.md`.

Each result directory contains the exact deck, FROMFILE launch, source snapshot/diff, binary/input hashes, task-owned PID and timeout, comparison CSV, coordinate snapshots, final analysis sources, tests and checksums. The controller run initially failed our postprocessing RF assertion because monitors split RF supports into smaller contiguous intervals. Source output was valid. The assertion now merges those intervals, and the original assertion status is retained in provenance history. Neither simulation was rerun to fix that assertion.

## Geometry and output details

`OpalMonitor::update` enforces nominal length≥.01 m even if L=0. Ring nominal circumference includes this length. Therefore each existing1 m D1 is replaced by.495 m drift +.01 m MONITOR +.495 m drift, with monitor plane at the original D1 centre and downstream drift at centre+.005 m. All six triples sum to the original length. The monitor's actual support is±.005 m around its plane. Original field-producing elements are unchanged.

Monitor planes have unlimited transverse extent. Opposite-side monitor supports therefore appear in the RF IndexMap intervals; their reverse reference crossings do not trigger these temporal snapshots. Those additional passive boundaries can make the all-particle boundary controller more expensive. This input arrangement should not be interpreted as curing that controller's large-N cost.

The particle state is at the Boris endpoint when the reference midpoint callback invokes a temporal monitor. `driftToCorrectPositionAndSave` corrects positions by `(tau−1/2) dt` to the reference plane time. In a force-free straight this exactly explains the verified position agreement. Monitor local positions are relative to the reference at that plane; the group RefPartR/RefPartP attributes reconstruct lab coordinates and the desired static D1 section frame.

Two existing output issues are documented, without changing their C++ implementation:

1. The row `time` dataset uses the bunch endpoint time. It differs from group TIME by up to31.02 ps in the62.5 ps test and49.18 ps in the100 ps test. Use group TIME for all particles in a temporal snapshot.
2. The final directed return can be recorded from a preceding midpoint extrapolation and again from the shortened final step. In the fixed test the two group times differ by4.2e−22 s and describe the same event. Deduplicate consecutive same-monitor times within64 floating-point epsilon relative tolerance, keeping the latest stored group.

## Offline reduction and disk budget

`analysis/reduce_temporal_monitors.py` reads one full snapshot at a time **after the run closes its files**. It computes centroid, full6×6 covariance, whitened radial CDF error, outside fraction, fourth radial moment and core-density ratios using every particle. Optional model covariance supplies a second density comparison. Only after those calculations does it retain a reproducible seeded sample of up to1024 fixed IDs, chosen independently of position/amplitude/tune. Missing IDs are explicitly flagged. The same IDs are used at all passages.

The reduced output includes:

- `section-observations.npz`: fixed-ID6D coordinates, group times, cell/turn coordinates, survival mask, plus full-bunch centroid/covariance arrays.
- `full-bunch-diagnostics.csv`: whole-bunch density diagnostics and population per passage.
- `raw-first-last.h5`: first and last unique full snapshot at each of the six monitors.
- `sampling.json`: exact selected IDs, seed, method, time and duplicate conventions.
- `manifest.json`: input/output checksums, byte counts, geometry and provenance; exact input and run metadata are copied.

The reducer **does not delete inputs** or edit actively written H5 files. Raw intermediate snapshots can be released only after the reducer manifest and retained output have been checked. Keep the FROMFILE launch and initial/final ordinary H5 snapshots separately as the t=0/end reference. A run starts at C1's plane and normally first samples C2; C1's first monitor passage is one turn later.

Example:

```sh
~/.venv-h6/bin/python analysis/reduce_temporal_monitors.py \
  results/temporal-monitor-fbf4cfcq results/temporal-monitor-fbf4cfcq/reduced \
  --study results/rf-ring-s3a01yyz --count 1024 --seed 90217
```

Measured datasets occupy80 bytes per particle per monitor passage, before small HDF5 headers. For six samples per turn:

| N | Turns | Raw monitor payload | Retained12 full endpoint snapshots | Fixed1024-ID6D sample payload |
|---:|---:|---:|---:|---:|
|131072|64|3.75 GiB|.117 GiB|18 MiB|
|524288|32|7.50 GiB|.469 GiB|9 MiB|
|524288|64|15.0 GiB|.469 GiB|18 MiB|

The current≈12 GiB free disk cannot hold the524288-particle64-turn raw output. Even32 turns requires checking current free space and allowing margin for initial/final H5, logs, checkpoint files and other concurrent tasks. Reduce completed cases before launching the next large case. Avoid a high-frequency ordinary H5 dump for full bunches; temporal monitors already provide the cell samples.

## Scope and checks

Thirteen monitor/reduction unit tests pass, including rotation, exact straight interpolation, circumference preservation, row/group time separation, terminal duplicate handling, fixed-ID reproducibility, missing-ID masks, whole-bunch moments and complete synthetic offline reduction. No unit test invokes OPALX. Fourteen RF-runner unit tests pass, including the merged-support assertion. Scripts and generated inputs were reviewed.

The present evidence validates bare field-free straight snapshots. With space charge, temporal momenta are retained at the integration endpoint rather than interpolated to the crossing time, and the position correction uses that momentum. Their error must be checked in the PIC timestep convergence study; bare agreement does not establish it. Curved-section monitors and general three-dimensional frame rotation were not validated. These limitations do not prevent using the existing monitor output with explicit conventions and convergence checks, but they must remain part of the numerical uncertainty.
