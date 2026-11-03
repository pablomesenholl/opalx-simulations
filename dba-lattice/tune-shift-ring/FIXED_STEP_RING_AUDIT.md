# Independent fixed-step RF-ring audit

No tested ordinary fixed Boris timestep is acceptable for the coupled tune
benchmark at an absolute tune error target of 0.001. Increasing the probe
amplitude from 10 µm to 1 or 5 mm does not remove the failure. This is a failed
zero-charge prerequisite, not a measurement of a physical space-charge shift.

## Retained finite-amplitude controls

Data: `results/finite-amplitude-fixed-rf-qft9g9ng/`. Six runs use 13 symmetric
normal-mode probes, 32 turns, one MPI rank and one OpenMP thread. The physical
RF calibration, D1-centre launch, six equivalent sections, and normal-mode basis
are shared with `results/rf-ring-s3a01yyz/`. `REPARTFREQ=10` selects the existing
ordinary fixed Boris path; no new numerical parameter is introduced.

Analytic bare RF tunes are (0.764038385533, 1.689536397020, 0.069128128193).
The table reports the **largest absolute phase-fit error among four primary
probes per mode**, retaining opposite-probe disagreement instead of cancelling
it in an average.

| Amplitude | DT | Max x error | Max y error | Max z error | Max centre–tracked reference separation |
|---|---:|---:|---:|---:|---:|
| 1 mm | 400 ps | invalid sections | invalid sections | invalid sections | 0.649 m |
| 1 mm | 200 ps | 0.012276 | 0.000477 | 0.023138 | 0.330 m |
| 1 mm | 100 ps | 0.016104 | 0.000353 | 0.006553 | 0.092 m |
| 5 mm | 400 ps | invalid sections | invalid sections | invalid sections | 0.549 m |
| 5 mm | 200 ps | 0.021345 | 0.000755 | 0.010394 | 0.175 m |
| 5 mm | 100 ps | 0.020422 | 0.000857 | 0.028714 | 0.148 m |

The 400 ps simulations complete, but each has 15 unordered equivalent-section
transitions. The observation guard is unchanged and no tune is assigned.
The valid 200/100 ps cases each have 193 events. Their horizontal/longitudinal
signals have low spectral coherence in some probes, and phase/spectral estimates
disagree by as much as 0.0615. Even the 5 mm/100 ps vertical spectral error reaches
0.001146. No monotonic convergence claim is supported.

An independent least-squares fit of one stationary cell matrix, in conditioned
normal coordinates and after centre subtraction, has relative RMS prediction
errors of 0.096–0.133 in these four valid runs. This is separate evidence that
the failure cannot be assigned only to the phase estimator. Opposite horizontal
probes differ by as much as 0.0250 in inferred tune.

![Retained tune errors and reference divergence](results/finite-amplitude-fixed-rf-qft9g9ng/fixed-step-audit.png)

## Which orbit is moving?

The last column above is taken directly from raw `local_r[:, centre]`: it is
the central particle minus the concurrently advanced tracking reference. It is
distinct from deviation of either trajectory from the nominal magnetic COF.
For these 1/5 mm controls, the reference itself has maximum horizontal section
offsets of 37.6–96.0 mm. All controls start with tiny centre/reference differences;
these grow rapidly over the first few turns. The raw histories are retained.

The original 10 µm/400 ps case is independently audited in
`results/fixed-boris-audit/10um-400ps.json`. Its centre has 79.2 mm maximum absolute
horizontal section offset, the reference has 88.3 mm, and their maximum raw 3D
separation is 285.1 mm. At the first cell the reference is already 2.54 mm off the
nominal design while the two trajectories agree to about 2e−13 m. Their difference
first exceeds 1 nm at 1.36 turns, 1 µm at 2.56 turns and 1 mm at 3.66 turns. Both
use the fixed Boris path: this audit does not attribute the growth to different
named integrators. The initial discrepancy and subsequent amplification must
both be addressed.

## Observation and model checks

The observation uses static, equivalent D1 straight sections, with coordinates
(x, px/p0, y, py/p0, z_head, pz/p0−1). Six sections per turn retain the vertical
integer tune. Primary modes are selected from the excitation labels; weak
cross-mode leakage fits are not counted as independent validation. The explicit
central probe is subtracted for frequency extraction, while absolute and relative
orbit errors are audited separately.

The finite-amplitude controls store phase-space every approximately 8 ns
(20/40/80 steps). These brackets are about 0.175 m, wholly inside the 1 m straight
for the nominal section. Particle positions interpolate linearly in the
force-free straight. The strict geometry/event ordering checks reject trajectories
for which a nominal section sequence is no longer defensible.

The boundary-resolved 10 µm RF control in
`results/rf-ring-s3a01yyz/bare-probes-32turns-10um/` agrees with the linear model
within 5.99e−7, 7.85e−7 and 1.94e−7 in the primary x, y and z modes. Its calibrated
synchronous reference also closes. That establishes a viable observation/model
control. It does not validate fixed steps or the finite-amplitude dynamics.
The deliberately small initial RF phases cannot explain the percent-scale errors
in the fixed controls; once a trajectory wanders by centimetres, its RF dynamics
also changes and a simple small-amplitude tune interpretation is no longer valid.

For tiny probes, their hard-edge arrival differences are much smaller than a
100–400 ps interval. Midpoint support sampling can assign an identical kick
duration to neighbouring trajectories and miss the correct first variation of
the physical edge map. The larger-amplitude experiment tests whether beam-scale
amplitudes avoid this issue; the observed failures reject that workaround for
these timesteps. No production timestep or large-PIC result follows from this
study. No additional resolved 5 mm control was run because the corresponding
fixed-step results already fail the proposed acceptance criterion.

## Reproduction and preservation

Executable SHA256:
`e5b8b517eab9b20a4342809c0c5d8b2a9a9d065fbac12d805180163af0f442bb`.
Each case retains its input, particles, launch labels, provenance, log, reference
history, and raw `case.h5.gz`. Compression was verified against the raw HDF5 SHA256
before removing that task-owned uncompressed copy. Valid cases also retain
`section-observations.npz` and all mode-fit outputs. Initial failed-run logs and
the original runner are preserved; no numerical tolerance was relaxed.

The independent summaries can be regenerated without OPALX execution:

```sh
cd sandbox/Regression-Tests/tune-shift-ring/analysis
~/.venv-h6/bin/python -m unittest test_fixed_probe_audit.py
~/.venv-h6/bin/python summarize_fixed_step_controls.py \
  ../results/finite-amplitude-fixed-rf-qft9g9ng ../results/rf-ring-s3a01yyz
```

Three pure Python tests cover reference/particle distinction, recovery of a known
linear cell map, and preservation of cancelling per-probe errors. All pass.
Publication PNG and PDF were rendered and inspected. No production source was
changed by this audit. The proposed residence correction has separate mathematical
requirements described in `EDGE_AVERAGING_INVESTIGATION.md`.
