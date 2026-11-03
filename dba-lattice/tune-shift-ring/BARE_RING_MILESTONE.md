# Bare DBA ring milestone

Status: **the bare magnetic DBA milestone passes all 23 validation gates**.
The final frozen study establishes a common geometric closed orbit and bare
tunes from fresh COF maps and device tracking, with passive selected-particle
measurements. Diagnostics off, three probes, and all nine probes produce
identical final numerical state bytes and logged accepted-step schedules over
32 directed reference turns. Earlier failed attempts are preserved below.
No space-charge or RF runs are included in the DBA numerical study.
The lattice and analytic reference are specified in the
[LaTeX note](../TeX/main.tex), `sec:hamiltonian`; comparison tables below retain
the analytic row alongside each historical numerical result.

## Intended comparison

Use the same static magnetic DBA for a fresh host COF solve, the COF numerical
one-turn map, and device Boris tracking. Observe selected native particle IDs at
the six equivalent D1 midplanes, without changing fields, particles, or the
accepted-step sequence. Start from the actual solved COF position and momentum.

The experiment contains a central particle and symmetric horizontal/vertical
position probes at 10 and 5 micrometres (nine particles in total). Device cases
use 32 turns and 62.5/31.25 ps nominal timesteps. The coarse case is repeated with
diagnostics disabled, three IDs selected, and all nine IDs selected. All runs use
one MPI rank and one OpenMP thread. Host maps use both nominal timesteps and
30/15 micrometre position differences, with momentum differences scaled by the
nominal reference momentum.

The reference section is the middle of C1_D1, 0.5 m from either adjacent magnet.
The ring's physical drift remains unbroken. Six equivalent sections per turn
allow the full device tunes to be measured without imposing their integer parts.
Matrix eigenvalues supply fractional phases; their integer labels are explicitly
assigned from the independent hard-edge cell optics.

## Final validated study

Directory: [bare-ring-milestone-wp5fpjfd](results/bare-ring-milestone-wp5fpjfd/).
Authoritative outputs are the [complete 23-gate report](results/bare-ring-milestone-wp5fpjfd/summary.json),
[individual probe tunes](results/bare-ring-milestone-wp5fpjfd/tunes.csv),
[manifest](results/bare-ring-milestone-wp5fpjfd/manifest.json), and
[strict four-turn repeatability check](results/bare-ring-milestone-wp5fpjfd/repeat4/comparison.json).

The frozen executable SHA256 is
`c26084e30f195c5a5bf1ff4f5e23ea3abd50094058c7f1ba9c8f9f91f158587c`.
The source baseline is `288d4327cedc742e8bf4c32878070d32a2f5e15e` with the
preserved local edits. All 40 frozen changed-source files match the validated
working tree. Each scientific case uses one MPI rank and one OpenMP thread;
independent device cases run concurrently, so elapsed times are not performance
measurements. The saved `run_gated.py` requires the short strict-repeatability
test to pass before launching the full study. Every requested case completed,
and every owned child was reaped.

### Closed orbit and tunes

The central particle is compared at its own directed plane crossings, using the
actual COF solution rotated into the six symmetry-equivalent D1 frames. Maximum
errors over all observed planes and turns are **0.4107 nm** in position and
**`8.3433e-12 mc`** in mechanical momentum. Both pass the unchanged 1 nm and
`1e-10 mc` gates. The largest final reference return error is **0.01172 nm**.

| Calculation | Qx | Qy |
|---|---|---|
| Independent hard-edge analytic optics | 0.7701075691329 | 1.6895363970202 |
| COF matrix, 62.5 ps / 15 µm FD | 0.7701083748691 | 1.6895374072269 |
| COF matrix, 31.25 ps / 15 µm FD | 0.7701077711904 | 1.6895366504457 |
| Device, 62.5 ps, mean over four probes per axis | 0.7701083767371 | 1.6895374081212 |
| Device, 31.25 ps, mean over the same probes | 0.7701077716181 | 1.6895366505326 |

The refined device tunes range from `0.770107767060657` to
`0.770107777627855` horizontally and from `1.68953665053013` to
`1.68953665053497` vertically. The largest individual difference from the finest
COF matrix is **`6.44e-9` in Qx and `8.93e-11` in Qy**. All matrix timestep and
finite-difference comparisons, device timestep/amplitude comparisons, and
analytic comparisons pass the `2e-6` tune gate. The maximum half-record tune
difference is `5.13e-8`; the maximum difference from the independent real-signal
frequency estimator is `1.14e-8`.

Each device run completes **32 directed reference turns**. There are 191 or 192
own-plane observations per off-orbit probe, depending on whether it arrives at
the final launch section before the reference stops. The central probe has 191
crossings: 31 at its launch section and 32 at each other section. All recorded
crossings enter the orbit check; none is discarded to improve agreement.

### Passive measurement and repeatability

The short off/off/three/three/all check gives **96,925 accepted steps in every
case**, identical raw endpoint arrays and reference attributes, identical logged
step schedules, and identical original CSV row bytes for common selected IDs.
It includes repeated diagnostics-disabled runs, so the former intrinsic
repeatability failure is independently covered.

The full 32-turn coarse cases give:

| Diagnostics | Accepted-step log records | Final numerical state bytes | Shared CSV row bytes |
|---|---|---|---|
| Off | 789,148 | Baseline | No probe output |
| IDs 0, 1, 3 | 789,148 | Identical to off | Identical to corresponding all-ID rows |
| All nine IDs | 789,148 | Identical to off | Identical to corresponding three-ID rows |

All three schedule hashes are
`745473abbbd902f80798abb1e8ec6737290a86e95fdf0eca5d9a2f4f18aa0187`.
Array comparisons require matching shape, dtype and raw bytes, distinguishing
positive from negative zero. CSV comparisons use original header/row bytes,
without floating-point parsing. Schedule hashes include every emitted accepted
step index and printed time/unit; their time precision is the log's precision.
Source/unit checks of the read-only recorder and exact final-state comparison
provide independent passivity evidence. The refined case completes 1,392,610
accepted steps.

The repair that restored repeatability restricts the new deterministic element
callback order to supported bare rings. Pointer-address iteration previously
changed floating-point frame-transform ordering between processes; the short
and full comparisons now repeat exactly. The selected diagnostic IDs never
select particles for integration refinement.

### Limits and separate timing observation

The central particle's **equal-time longitudinal displacement** is a timing
observable, distinct from its geometric position at its own plane crossing.
It is −17.330 nm (equivalent to −0.7935 fs) at 62.5 ps and −26.929 nm (−1.2329 fs)
at 31.25 ps. Its magnitude increases by a factor of 1.554 with timestep
refinement, so convergence of this small timing discrepancy is not established.
The existing geometric orbit and tune tolerances have not been loosened.

Some human-readable step lines label microsecond-scale numbers as `[ms]`.
Physics analysis uses the CSV/HDF5 SI timestamps, not those formatted labels;
fixing that display issue is outside this study. The observed execution backend
is Kokkos OpenMP; CUDA/HIP hardware was not tested. This milestone does not
validate space-charge tune shifts, a stationary distribution, or RF dynamics.

### Tests and cyclotron non-interference

The final [validation evidence](results/bare-ring-milestone-wp5fpjfd/validation/evidence.json)
records **20 passing CTest suites, six passing dedicated MPI2 unit suites,
15 passing Python tests**, their log hashes, the executable hash, and the 40
matching source snapshots. Scientific studies remain single-rank.

The final [cyclotron comparison](/Users/adelmann/git/opalx/tmp/cyclotron-guard-20260916-1789562042/comparison-after-order-fix.json)
exactly matches the baseline for coasting, accelerating gap-stop, and
262,144-particle midpoint/prestep PIC fixtures. See
[CYCLOTRON_GUARD.md](CYCLOTRON_GUARD.md) for fixture scope and provenance.
These are measured non-interference checks; they do not claim exhaustive
coverage of every cyclotron configuration.

## Earlier second study: repeatability failure, now superseded

The following measurements are preserved history from before the deterministic
bare-ring callback-order fix. They do not describe the final validated binary.

Directory: [bare-ring-milestone-p67acdol](results/bare-ring-milestone-p67acdol/).
Read the complete [gate report](results/bare-ring-milestone-p67acdol/summary.json),
[individual tunes](results/bare-ring-milestone-p67acdol/tunes.csv), and
[run manifest](results/bare-ring-milestone-p67acdol/manifest.json).

Executable SHA256:
`fccb6e76ec0a4fc9610a5aff8873641a972dae9e85ca6067d3d6a4965e326252`.
The four independent device cases ran concurrently, each on one MPI rank and one
OpenMP thread, with a task-owned 900-second limit. All completed and were reaped.
Elapsed times are therefore not performance measurements. The study-local
`run_concurrent.py` records this execution; the normal runner reproduces the
same experiment sequentially. Fresh COF maps match the table below.

### Orbit and tune results

All central-particle own-section observations pass the unchanged 1 nm position
and `1e-10 mc` momentum gates. The maximum over all six sections and all recorded
turns is **0.3980 nm** and **`8.1824e-12 mc`**. Each recorded device case contains
191 central crossings: 31 at the initial section and 32 at each other section.
Tracking stops at the reference's 32nd return; the slightly delayed central
particle has not yet reached its own final launch-section crossing.

| Calculation | Qx | Qy |
|---|---|---|
| Independent hard-edge analytic optics | 0.7701075691329 | 1.6895363970202 |
| COF matrix, 31.25 ps / 15 µm FD | 0.7701077711904 | 1.6895366504457 |
| Device, 62.5 ps, mean of four signed/amplitude probes per axis | 0.7701083760483 | 1.6895374081200 |
| Device, 31.25 ps, same probes | 0.7701077698266 | 1.6895366505323 |

At 62.5 ps, Qx ranges from `0.7701083707867` to `0.7701083807180`;
Qy ranges from `1.6895374081180` to `1.6895374081219`.
At 31.25 ps, Qx ranges from `0.7701077670846` to `0.7701077707998`;
the vertical values span `2.745e-13`.
The maximum refined-device difference from the finest COF matrix is
**`4.106e-9` horizontally and `8.66e-11` vertically**. Matrix DT/FD refinement,
device DT/amplitude refinement, half-record consistency, and independent
sinusoid fitting all pass the `2e-6` tune gate.

The equal-time longitudinal offset is a separate timing diagnostic. For the
all-probe cases it is **−18.132 nm (−0.830 fs)** at 62.5 ps and
**−25.251 nm (−1.156 fs)** at 31.25 ps. Its magnitude increases by a factor of
1.393 when DT is halved; convergence of this small timing discrepancy is not
established. These offsets are not geometrical closed-orbit errors. The
all-section orbit comparison and transverse endpoint gates remain unchanged.

### Passivity and repeatability failure

| Coarse 32-turn case | Accepted-step log records | Exact endpoint versus diagnostics off |
|---|---|---|
| Diagnostics off | 789,356 | Baseline |
| IDs 0, 1, 3 | 789,296 | Different |
| All nine IDs | 789,351 | Different |

The schedule hashes differ, and shared three-ID CSV records are not bitwise
equal to the corresponding all-ID rows. Maximum final particle-position
differences versus diagnostics off are approximately 1.414 nm for three probes
and 1.153 nm for all probes. The first divergence visible in the thinned logs
occurs at accepted step 10,999: 561.875 ns with diagnostics off versus 562.000 ns
in both recorded cases. The previous retained step, 9,999, agrees.

Four additional four-turn cases preserve the exact launch and nominal timestep;
they change only the turn limit. These are repeatability diagnostics, not
replacements for the 32-turn study. Their
[comparison](results/bare-ring-milestone-p67acdol/repeat4/comparison.json) shows:

| Pair | Accepted-step counts | Maximum particle-position difference |
|---|---|---|
| Off / off | 97,025 / 96,914 | 0.1113 nm |
| Three probes / three probes | 96,884 / 96,935 | 0.08252 nm |
| Off / three probes | 97,025 / 96,884 | 0.09742 nm |

Thus the non-repeatability exists with diagnostics disabled. These data do not
prove recorder feedback, and they also do not prove bitwise passivity. The
bitwise gates remain failed. Investigation of field ordering and floating-point
edge decisions is separate from the successful tune comparison; no causal
mechanism has yet been established by this numerical study.

### Regression evidence

The frozen [validation evidence](results/bare-ring-milestone-p67acdol/validation/evidence.json)
records 20 passing CTest suites, six passing dedicated two-rank unit suites,
14 passing Python tests, and verification of all 40 frozen changed-source files.
The execution backend is Kokkos OpenMP; CUDA/HIP hardware was not tested.

The final executable's independent
[cyclotron before/after comparison](/Users/adelmann/git/opalx/tmp/cyclotron-guard-20260916-1789562042/comparison-after-probe-fix.json)
matches the baseline exactly for the coasting, accelerating gap-stop, and
262,144-particle midpoint/prestep PIC fixtures. Its small OPEN dispatch smoke test
is explicitly not force or tune convergence evidence. These checks support the
non-interference requirement for the exercised cyclotron infrastructure.

## First frozen attempt

Directory: [bare-ring-milestone-aukgh9a_](results/bare-ring-milestone-aukgh9a_/)

Executable SHA256:
`07b0b6e80368de78bd0b796930dbb410060226a0e3bd86450ccbbd2399629b52`.
The directory preserves the executable, full dirty-source snapshot, inputs,
particle launch, analysis scripts, logs, run status, and outputs. The study did
not change the tracking source or numerical tolerances.

### Fresh COF maps

| Nominal DT | FD position step | Qx | Qy |
|---|---|---|---|
| 62.5 ps | 30 µm | 0.7701083746754863 | 1.6895374070438538 |
| 62.5 ps | 15 µm | 0.7701083748691482 | 1.6895374072268818 |
| 31.25 ps | 30 µm | 0.7701077710385854 | 1.6895366502626330 |
| 31.25 ps | 15 µm | 0.7701077711904131 | 1.6895366504456684 |

All four COF cases completed. The finest residual is approximately
`2.34e-14 m` horizontally and `2.08e-16 mc` in horizontal mechanical momentum;
vertical residuals are zero.

### Device observations

The diagnostics-disabled coarse case completed 32 directed turns in
192.28 seconds with 788,299 accepted-step log records. All nine particles remain.
Its final reference return is `2.09e-12 m` from the COF launch.

The central particle's simultaneous reference-frame displacement is
approximately `(2.585e-10, 0, -1.65664e-8) m`. The longitudinal component is
larger than the first runner's 1 nm equal-time-distance gate. Review identified
that this mixed two observables: geometric orbit closure and longitudinal phase
at the same laboratory time. The revised analysis retains the existing 1 nm
own-section orbit gate for all planes and turns, checks the transverse endpoint
offset and reference closure at 1 nm, and reports the simultaneous longitudinal
slip and its equivalent flight time separately. A geometrically closed orbit
can have a slightly different numerical time of flight. This is an explicit
correction of the comparison observable, with no change to the position,
momentum, or tune tolerances. The refined-timestep trend remains to be measured.

The three-ID recorder case stopped after 94,435 accepted-step log records with
`PassiveRingProbe: Invalid passive observation for particle ID 0`. It completed
69 valid selected-particle crossings, including 23 central-particle crossings.
Those central crossings agree with the COF state rotated to each equivalent D1
section to a maximum `3.984e-11 m` in position and `8.017e-13 mc` in momentum.
This partial record is insufficient for the requested 32-turn validation.

A likely cause of the recorder failure is the non-increasing represented-time
check: at a few microseconds, an accepted boundary substep can be smaller than
one floating-point spacing of absolute time. This is a diagnosis from the
controller/recorder interfaces, not yet a confirmed repair. Passive diagnostics
must accommodate this without changing the live integration.

The all-nine-ID and refined-timestep device cases were not started after the
recorder failure. Consequently, device tune agreement, full-duration passivity,
and refined device-orbit agreement remain unproven.

## Reproduction and checks

From this directory:

```text
~/.venv-h6/bin/python -m unittest discover -s analysis -p test_bare_ring_milestone.py -v
~/.venv-h6/bin/python analysis/bare_ring_milestone.py --binary /absolute/path/to/opalx
~/.venv-h6/bin/python analysis/bare_ring_milestone.py --run PRINTED_STUDY --timeout 900
```

Fifteen pure unit tests pass; none executes OPALX. They cover input generation,
actual-COF launches, matrix modes, six-plane phase extraction, invalid/missing
records, intermediate central-orbit errors, bitwise comparisons, streamed log
hashes, and task-owned process termination. The runner hashes every printed
accepted step/time while retaining one such log line per 1,000 steps. The hash
uses OPALX's printed time precision, so bitwise final states and independent
recorder tests are also required as passivity evidence. New tests explicitly
distinguish a pure longitudinal phase slip from a transverse orbit error.

Predeclared study gates are `2e-6` tune agreement/convergence, `1 nm` position
agreement, and `1e-10 mc` momentum agreement. The tune gate resolves the intended
order-0.1 space-charge shifts by a factor of 50,000; the position gate is 1e-4 of
the larger probe amplitude and much smaller than the original millimetre
discrepancy. No gate establishes cyclotron compatibility; that is covered by
separate source review and regression tests outside this numerical study.
