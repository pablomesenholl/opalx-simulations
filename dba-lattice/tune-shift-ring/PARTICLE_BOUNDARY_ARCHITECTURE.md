# Particle boundary architecture audit

Status: source audit completed; **explicit approval for the new API is pending**.
See [PARTICLE_BOUNDARY_PROPOSAL.md](PARTICLE_BOUNDARY_PROPOSAL.md). This audit made
no production edits and launched no simulations. It does not establish numerical
convergence or authorize implementation.

## Feasibility and scope

Independent device particle substeps can retain ordinary `element->apply(pc)`
and `BorisPusher::kick`. They need a scheduling layer around those operations,
with one common predictor-midpoint PIC solve per macro interval. Frozen gathered
self fields are an explicit approximation; this is not repeated self-consistent
PIC at the individual substep times.

The initial scope should remain the current one-rank, one-container ring with
supported analytic elements, fixed binning, no ongoing emission and no spin.
`DeviceExternalField` may supply geometric support tests; its independent
transport oracle must not replace production tracking or duplicate field models.

## Reuse and state sequence

1. Hold the tracking/reference frame fixed during a macro interval `H`. Predict
   all particles to a common midpoint, set every `pc.dt=H`, and perform the
   existing PIC solve once. Save gathered self E/B in device scratch arrays in
   the restored tracking frame, then restore the starting particle state.
2. Maintain device state per particle: accepted R/P, elapsed time, trial duration,
   pending dyadic intervals and active/accepted/rejected status. Preserve both
   children after a rejected interval; retain the existing coordinate-roundoff
   floor, depth limit, finite-state checks and a bounded work count.
3. In each batch, perform a masked half drift in SI coordinates, restore frozen
   self E/B, and call ordinary element field kernels. Test the trial endpoint
   against geometric support. Accept and kick/push each qualifying particle;
   restore and split each rejected particle independently.
4. End every particle at the same macro endpoint. Restore `pc.dt=H`, advance the
   global clock/reference and frame once, then compute physical bunch moments,
   emit output and flush particle deletion.

The existing [kickParticles](/Users/adelmann/git/opalx/src/Algorithms/ParallelTracker.cpp:1727)
uses each particle's `dt` (line 1727 in the audited source). The present drift
(line 1693) scales coordinates by `1/(c*dt)` before pushing, so it cannot handle
completed particles with `dt=0`. Use a masked SI half drift. An unchanged Boris
kick can use zero duration for completed/rejected particles only while their
state and working fields remain finite.

## Masking and likely cost

| Approach | Consequence |
|---|---|
| Unchanged full-container field calls | Smallest first implementation: transform/evaluate all N each batch, restore completed R/P exactly, and zero completed working E/B. Cost remains proportional to N times the batch count. |
| Internal active-mask guards | Better practical target: guard transforms, candidate selection and supported field kernels while retaining `apply(pc)` and its field formulas. Mask scans remain, but inactive field work and memory traffic fall. |
| Compact scratch ParticleContainer | Possible but more invasive: device pack/scatter requires stable indices; construction, layout/state registration, collective allocation, counts and generated IDs need isolation. Do not change the live physical population to express the work population. |

[ParticleContainer::transformBunch](/Users/adelmann/git/opalx/src/PartBunch/ParticleContainer.hpp:230)
(line 230) transforms all R/P/E/B and marks moments dirty. The current
[field iteration](/Users/adelmann/git/opalx/src/Algorithms/ParallelTracker.cpp:1281) (line 1281)
also computes full-container bounds before its spatial branch. Both are major
reuse/performance points.

`apply(pc)` is a host virtual call that launches device kernels. Preserving it
means host-driven batches with device particle states and acceptance decisions;
only counts/flags need return to the host. It does not permit calling the whole
virtual element interface inside one persistent device kernel.

Scratch storage at roughly 500,000 particles is expected to be tens to a few
hundred MiB, depending on saved states and stack representation. The intended
gain is one PIC solve per macro step and batches governed by the most demanding
individual trajectory, instead of the union of every particle's boundary events.
Actual batch and transform cost remains unmeasured.

## Normalization, RF and side effects

- **PIC normalization:**
  [ParticleMeshFieldTransfer::scatterScaledTimeStep](/Users/adelmann/git/opalx/src/SpaceCharge/CartesianPIC3D/ParticleMeshFieldTransfer.cpp:57)
  (line 57) deposits `dt*Q`. Individual substep durations must not be installed
  before the common solve, or they change charge weighting. Save self E/B before
  external fields are added; restore this cache before every trial field gather.
- **RF:** [VariableRFCavity::apply(pc)](/Users/adelmann/git/opalx/src/AbsBeamline/VariableRFCavity.cpp:115)
  (line 115) evaluates amplitude, integrated frequency and phase on the host at
  the common bunch midpoint. Keeping bunch T/dT fixed during batches preserves
  these kernels and freezes the RF temporal value at the macro midpoint. This
  approximation needs timestep convergence, especially at cavity entry/exit;
  it is not exact substep RF timing. Reference comparisons should use the same
  convention or include the discrepancy in the numerical uncertainty.
- **Exact asynchronous RF:** no general per-particle absolute-time attribute
  exists. [AbstractTimeDependence](/Users/adelmann/git/opalx/src/Algorithms/AbstractTimeDependence.h)
  is a host virtual interface. Exact per-particle RF timing would require
  device-readable polynomial/sinusoidal/spline models or a restricted supported
  subset. Emission birth-time inventories are not a general tracking clock.
- **Moments and frames:** no physical bunch moments or PIC solve may use a cloud
  whose particles represent different times. Keep the frame fixed through all
  batches; cache fields in that frame and rotate working fields through the
  ordinary element transforms.
- **Monitors:** never invoke side-effecting monitor callbacks on trials.
  [Temporal monitors](TEMPORAL_MONITOR_STUDY.md) assume a common timestep and
  endpoint. Independently refined reference callbacks do not supply synchronized
  bunch snapshots; output must be tied to common events or adapted explicitly.
- **Apertures:** mark losses only from accepted states; defer deletion until the
  macro endpoint. Rejected trials must not create irreversible marks. Endpoint
  checks alone can miss an aperture crossed earlier in the interval.
- **MPI and ordering:** current kick/drift wrappers contain MPI barriers, which
  cannot remain inside rank-local loops with different iteration counts. PIC
  migration can reorder particles and invalidate cached views; preserve ID
  association and reacquire views after a solve. Scratch views require deliberate
  handling if migration, emission, deletion or resizing is later supported.

## Required validation before production use

Retain ordinary field-kernel/Boris comparisons, verify constant physical charge
and impulse normalization, test accepted/rejected/completed masks and exact
macro-duration completion, and establish macro-timestep convergence for RF,
collective forces and tune extraction. Check conservation and the consistency of
host reference tracking with device particles. Numerical tolerances, parallel
ordering changes and any expanded scope need explicit justification. No such
validation was performed by this architecture-only audit.
