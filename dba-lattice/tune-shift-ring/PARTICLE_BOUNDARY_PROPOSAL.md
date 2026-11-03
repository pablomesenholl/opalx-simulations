# Proposal: independent particle boundary steps with midpoint PIC

Status: numerical prototype passed; OPALX implementation and public input
addition are not authorized yet. Point 5 of the tune-shift study is incomplete.
This proposal replaces the earlier unimplemented BOUNDARYDT proposal.

## Why this change is needed

The current collective boundary controller reproduces the bare closed orbit
and tunes, but a 131,072-particle charged trial spent 90 seconds advancing only
158 fs of its first 62.5 ps interval. Every particle's crossing can trigger a
new collective PIC solve. Ordinary fixed steps of 100/200/400 ps failed the
ring's orbit/tune checks. Field-residence averaging has unresolved derivative
and combined-field errors. Neither is suitable for the charge scan.

## Proposed input and default

Add one opt-in RUN attribute:

    BOUNDARYMODE="COLLECTIVE"        // default: current validated behavior
    BOUNDARYMODE="PARTICLE_MIDPOINT" // proposed independent particle refinement

The existing TRACK DT is the common interval H, in seconds. No new tolerance,
charge convention, field boundary condition, particle-count-dependent switch,
or replacement particle pusher is proposed. Invalid combinations should fail
explicitly rather than silently disabling requested refinement.

Initial eligibility is the current aligned analytic ring with one container,
dynamic OPEN or NONE, fixed single bin, completed emission, no spin, checkpoint,
image charge or fixed Cartesian boundary. Scientific runs remain on one MPI
rank. A dedicated MPI regression must check collective ordering separately.

## Integration and reuse

1. With all particle dt equal to H, use the ordinary SI half-drift predictor
   R*=R_n+H v(P_n)/2 and the existing common-time PIC solve/gather. Preserve
   charge normalization: deposition currently depends on dt*Q.
2. Cache the returned self E [V/m] and B [T] in device memory and restore the
   starting particle state. Reacquire views after any solver-owned migration.
3. Resolve each particle's external support crossings independently, using
   ordinary additive element field kernels and the existing combined Boris
   kick. Each substep contains both self and external fields. Rejected trials
   split only that particle's dyadic interval; both halves are retained.
4. Keep particle state and decisions on the Kokkos execution space. Host-driven
   batches may dispatch existing element kernels; only activity/error flags
   return to the host. The external-only test-oracle pusher is not used.
5. Finish all particles at the same endpoint before updating physical moments,
   the global clock, the reference frame, losses and output. Never deposit a
   population whose particles represent different times.

Use masked SI drift rather than the ordinary unitless-position conversion:
completed particles have zero working dt, which that conversion cannot divide
by. An internal active mask in frame/candidate/field work avoids repeatedly
processing finished particles. Field formulas and the normal apply(pc) entry
points remain shared. Rejected trials must not record monitors or delete rows.

## Approximations that must remain explicit

R* is a synchronized predicted midpoint, not generally the midpoint reached by
the final refined trajectory. Its error is O(H^2) for bounded forces. The gathered
self fields are held fixed for this common interval; the current test-particle
velocity still enters each Boris magnetic rotation. This is a change to the
numerical method, not just element selection or fewer identical PIC solves.

PIC's beam-frame conversion also depends on source mean momentum and gamma.
The present predictor uses P_n; position accuracy alone cannot establish a
general second-order self-field approximation. The implementation must either
provide a justified midpoint source-momentum predictor or explicitly retain the
quasistatic approximation and demonstrate its error in this low-beta ring.

The bounded [source-direction sensitivity](ELLIPSOID_BOOST_SENSITIVITY.md)
now quantifies one contribution: in the continuum 160 pC moment-family shapes,
holding gamma and laboratory covariance fixed, the worst local relative force
bound is 1.66e-6 at 62.5 ps and 1.06e-5 at 400 ps. This supports testing the
existing approximation first; it is not a full PIC or tune-error bound.

For the existing VARIABLE_RF_CAVITY kernels, the smallest implementation freezes
their time-model values at the common midpoint during the batches. This is not
exact RF evaluation at each particle's substep time. Apply the same temporal
convention to reference trials, and compare against the independently resolved
RF reference while refining H. General asynchronous time-model support is not
part of this proposal.

The method is not claimed to be symplectic or exactly reversible. Magnetic-only
Boris retains momentum-norm conservation; electric self fields require measured
energy/emittance drift. Floating-point ordering, execution scheduling, memory
traffic and MPI collective ordering will change inside this opt-in mode.

## Evidence already available

The standalone prescribed-field test in
results/frozen-midpoint-replay-jnmlakkz resolves a hard-edge magnetic slab and
a sector-like exit plane. It includes a spatial linear electric field and a
nonparallel constant self B. The independent DOP853 event/variational oracle
includes crossing derivatives. Both state and full 6x6 initial-condition
Jacobian converge approximately as H^2; tightening the oracle tenfold changes
its Jacobian by less than 2e-13. Seven pure numerical unit tests pass.

This evidence covers a prescribed smooth self field, not PIC source motion,
RF, MPI, GPU hardware or practical OPALX throughput. It supports implementation
testing, not a production tune-shift claim.

## Required gates before the charge scan

1. Device schedule, charge normalization, additive fields, rejected-trial
   restoration, active masks and failure handling unit tests, without OPALX
   execution. Existing collective-mode regressions remain unchanged.
2. Actual ring/reference agreement and three bare modes for tiny and finite
   probes, with interval refinement and comparison to the validated collective
   run. Check RF energy/phase and diagnostic interpolation separately.
3. One complete large-N charged cell: timing, particle/charge conservation,
   fields, moments and density versus H, particle count and mesh. Repeat the
   established straight force/short-arc controls as applicable.
4. Many-turn stability and a charge scan with individual mode tunes, independent
   estimates, time-window checks and particle/mesh/interval convergence.
5. Plot tracked tune shifts and their numerical uncertainty against the
   applicable analytic prediction. Monitor density deformation: the uniform
   ellipsoid moment closure is not an exact stationary kinetic distribution.

Implementation approval does not waive any gate or establish that the eventual
PIC scan will agree with the uniform-density closure.

## Approval boundary

Repository AGENTS.md states: "Do not change public APIs without explicit
approval." The new RUN attribute is a public input API. Approval is requested
for this concrete opt-in extension and its documented numerical behavior;
the source change has not been made. No commit or push is requested.
