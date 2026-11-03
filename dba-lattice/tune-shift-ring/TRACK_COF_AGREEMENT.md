# Device TRACK / host COF agreement — 10 September 2026

The no-space-charge DBA baseline now agrees using Boris in COF and TRACK.
This change applies to explicit `RUN, TURNS=...` on a supported static magnetic
ring. The rejected host particle implementation has been replaced: positions,
momenta, spatial field selection and boundary-resolved Boris evolution all run
in the particle views' Kokkos execution space. COF and the replicated reference
remain on the host. The ordinary collective-field tracking path still needs
equivalent spatial selection and boundary handling.

## Numerical evidence

2.5 MeV proton, one rank, one OpenMP thread, one turn, named COF launch.
The zero coordinates are the solved closed orbit, not a substituted launch.

| DT | Previous final transverse offset | New maximum particle/reference distance | Reference return distance |
|---|---:|---:|---:|
| 100 ps | 40.788 mm | 0.1051 nm | 1.254 pm |
| 25 ps | 7.254 mm | 0.4317 nm | 1.506 pm |

Maximum distances use every saved HDF5 snapshot (118 and 472 respectively),
not just the endpoint. COF transverse closure residuals are 0.247 pm and
0.00221 pm. The small particle/reference difference increases with the number
of reference-frame transforms; this is consistent with accumulated roundoff,
not the former millimetre-scale edge error. No numerical tolerance was loosened.
The regression requires the full recorded particle/reference distance <1 nm.

The device runs reproduce all recorded particle/reference positions of the
rejected host implementation at both timesteps. This establishes numerical
agreement of the revised transport on the available OpenMP backend; it does not
establish picometre physical accuracy against an exact trajectory.

Adding four spectators at x=+/-0.1 mm and z=+/-0.1 mm changes the central
particle trajectory by at most **0.661 pm** (100 ps). The spectator horizontal
excursions reach 0.387 mm: particles are integrated independently, not forced
onto the reference orbit. Both one- and five-particle cases preserve all particles.

## Implementation

`DeviceExternalField::Builder` uploads immutable element geometry, aperture and
field coefficients once, after lattice initialization. It never reads particle
arrays. Sector and rectangular bends reuse `BendFieldModel` with the same inputs
as their existing host/device field evaluation, including fringes. Dipole and
quadrupole descriptors capture the existing normalized coefficients.

`Ring::deviceBoris` executes a Kokkos parallel reduction in the particle views'
execution space. Every particle transforms into laboratory coordinates, tests
the exact spatial support of the descriptors, sums fields, advances its own
Boris substeps, and transforms back. Descriptors are scanned directly; there is
no path-index approximation or incomplete broad-phase candidate set.

Support changes at the start/midpoint/end cause subdivision using the host COF
policy. The iterative depth-first traversal uses a fixed 65-byte stack of depths
per particle, avoiding recursion and dynamic allocation. It shares the Boris
kick and compensated addition arithmetic with the host solver. The existing
element-length step cap and boundary resolution floor are retained. This is
boundary subdivision, not adaptive control of smooth-field truncation error.

No particle arrays are mirrored to host in this path. Only the failure count is
reduced to a host scalar and collectively checked. Endpoint E=0 and B diagnostics
are written into device views in the bunch frame. Reference and terminal-event
trials use the host `advanceResolvedInBeamline` wrapper. Trials never record
monitors; accepted reference substeps do. Aperture processing remains separate.

Positions remain metres, momenta p/(mc), time seconds, magnetic fields tesla.
There is no energy normalization, closed-orbit projection, new field model, or
change to the magnetic Boris rotation. The step partition and floating-point
operation order do change. Pure magnetic energy conservation is unit-tested.

The local build uses Kokkos OpenMP; CUDA, HIP and SYCL are disabled. The code
executes in device-callable Kokkos kernels, but GPU hardware compilation/execution
was not available and has not been validated. There is no host particle fallback.
Direct element scans and divergent subdivision have not been performance-tuned.

Explicit TURNS still requires FS=NONE, one container, no restart or continuing
emission. Supported descriptors: drifts, passive markers/monitors, SBEND, RBEND,
and normal/skew dipole/quadrupole multipoles. Nonzero higher multipoles, other
element types, explicit misalignments and spin are rejected. These restrictions
avoid silently using unsupported fields or host transport. Misalignment support
also needs reconciliation with COF's nominal-placement field query. Arbitrary
aperture-loss cases and multi-rank execution were not validated in this one-rank
study. Space-charge transport is not fixed by this change.

## Runs and reproduction

- [100 ps device run](results/pr553-tracking-63jqo0zd/cof_initialorbit/)
- [25 ps device run](results/pr553-tracking-cgql3l5o/cof_initialorbit/)
- [One/five-particle device comparison](results/resolved-spectators-3kzoixy5/summary.json)
- [Figure](results/device-track-cof-agreement-20260910/track-cof-agreement.png)
- [Figure PDF](results/device-track-cof-agreement-20260910/track-cof-agreement.pdf)

Each run retains input, log, HDF5 and tabulated trajectory. Parent folders hold
COF residuals and executable hashes. The spectator regression retains the source
diff. HEAD was `288d4327c` on `544-closed-orbit-finder-for-opalx`; changes remain
uncommitted. The final rebuild adds explicit unsupported-misalignment rejection
and its tests, without changing transport of this aligned lattice. Source snapshots
including new untracked files are retained with the device comparison figure.

From the repository, using `/Users/adelmann/.venv-h6/bin/python`:

```sh
python sandbox/Regression-Tests/tune-shift-ring/analysis/check_pr553_tracking.py --cof-integrator BORIS --launch cof_initialorbit --dt 1e-10
python sandbox/Regression-Tests/tune-shift-ring/analysis/check_pr553_tracking.py --cof-integrator BORIS --launch cof_initialorbit --dt 2.5e-11
python sandbox/Regression-Tests/tune-shift-ring/analysis/verify_track_cof_spectators.py <100-ps-output-folder>
```

Rebuilt OPALX and relevant tests. Single-rank ctest passed:
TestDeviceExternalField, TestTrackReferenceStep, TestClosedOrbitInitialState,
TestClosedOrbitSolver, TestOrbitThreader. Device tests verify the analytic thin
magnet impulse, energy conservation, finite spatial support, spectator
independence, rotated frames, reverse drift, fields of placed bends with fringes,
normal/skew quadrupoles, host/device transport agreement, and unsupported-input
rejection. The thin-magnet fixture initially failed because the existing n=0
setter halves its argument; the fixture was corrected, with no tolerance change.
Reference tests verify passive-monitor suppression during trials.
`git diff --check` passes; source diff reviewed.

Doxygen and TURNS input help describe the new behavior and limitations. The
configured physics/user manual directories are absent on this machine.
