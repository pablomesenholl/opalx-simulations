# Ordinary Boris/PIC tracking of the DBA ring

Updated 2026-09-10. Branch: `544-closed-orbit-finder-for-opalx`; base commit:
`288d4327cedc742e8bf4c32878070d32a2f5e15e`. Changes are local and uncommitted.
**Transport, solver coupling, and the relevant unit tests pass.**

**Thirty-two particles are insufficient for quantitative space-charge results.**
The charged N32 runs below verify solver/tracker coupling, charge response, and
MPI consistency only. They do not validate a space-charge tune shift. The bare
N32 run has self-fields disabled and checks independent single-particle tunes.
A substantially larger matched bunch, with particle-count and mesh convergence,
is required before comparing a predicted space-charge shift with tracking.

## Implementation

The ring now uses the existing production sequence in
[ParallelTracker.cpp](../../../src/Algorithms/ParallelTracker.cpp): device half
drift → existing midpoint PIC solve → additive existing element fields → one
combined Boris kick → device half drift. `TURNS` supplies a stopping condition;
it no longer selects a separate external-field-only particle transporter. COF
and the replicated reference remain on the host. Particle transport remains in
the selected Kokkos execution space.

[DeviceExternalField](../../../src/Algorithms/DeviceExternalField.h) supplies
immutable geometry for spatial candidate selection and support checks. The
existing bend/multipole device kernels still calculate fields. A register-only
[midpoint trial](../../../src/Algorithms/BorisMidpoint.h) uses the actual total
electric and magnetic fields without modifying live momentum. If any particle
or the external-field reference changes support, MPI makes one shared
accept/split decision. Rejection reverses only the uncommitted first drift on
the solver's current particle ownership; acceptance performs the ordinary kick
and second drift once. Monitor output occurs only after acceptance.

The [dyadic scheduler](../../../src/Algorithms/BorisStepControl.h) retains both
children, bounds work/depth, and sums accepted time without dropping a small
remainder. Finite-state and root-only trial failures are propagated collectively.
`MAXSTEPS` counts completed nominal intervals; diagnostics/global step counts
refer to accepted substeps, and solver counters include attempted solves.

No public input attribute or physical unit changed. Detailed equations,
roundoff-floor definitions, and conservation assumptions are authoritative in
the [physics manual](../../../../opalx-manual/physics/overview/index.qmd#ring-boris-pic);
configuration is documented in the
[tracking guide](../../../../opalx-manual/user-guide/tracking.qmd#ring-boundary-refinement).

## Supported scope and conditioning fixes

Refinement currently requires an aligned, supported analytic static `RING`, one
container, positive `DT`, completed emission, no spin, and no checkpoint/restart.
The solver must be dynamic `OPEN` or `NONE`, without image/fixed boundaries or
periodic ORB repartitioning. Binning is absent or fixed at one bin with file/table
diagnostics disabled. Set `REPARTFREQ=0` explicitly; its default of 10 disables
refinement. A fixed `GAMMAZ` bin retains the existing relativistic self E/B
composition. Unbinned mode retains the existing electrostatic model.

Other configurations keep ordinary fixed steps. Compatible ring geometry still
uses spatial field selection independently of retry eligibility; general field
maps and unsupported elements retain their established path. Startup output
states the selected mode and any refinement restrictions.

Three numerical corrections are necessary for this sequence:

- Host/device field support agrees at inclusive entrances and exclusive exits;
  field support and aperture-loss ownership remain distinct.
- [Quaternion.cpp](../../../src/Algorithms/Quaternion.cpp) computes small
  rotations using `atan2(cross magnitude, dot)`, avoiding cancellation that
  previously left the moving frame behind the reference. The existing
  `1e-12` parallel/antiparallel threshold is unchanged.
- Eligible dynamic solves retain the mean-momentum rotation but use zero PIC
  frame translation. Translating a micrometre bunch by accumulated path length
  introduced particle-coordinate roundoff, even with `TYPE=NONE`. Open dynamic
  solves are translation invariant; all excluded configurations retain their
  previous frame handling. Cold fixed-bin/zero-span assignment also avoids a
  division by zero in [AdaptBins](../../../src/PartBunch/Binning/AdaptBins.tpp).

The N32 diagnostic that exposed the frame issue is preserved in
[the stencil audit](results/gaussian-bare-62.5ps-x8e8z4dv/audit_stencil_report.md)
and [raw bend-invariant audit](results/gaussian-bare-62.5ps-x8e8z4dv/audit_raw_lab_frames.json).
Conditioned interpolation did not remove its horizontal discrepancy. N5 probes
skip the solver below the default `MINBINEMITTED=10`; an N32 check is therefore
essential. No analysis tolerance was relaxed to hide that failure.

## Validation

Unless stated otherwise, simulations use one MPI rank and one OpenMP thread.
The numerical build used for the completed results below has SHA-256
`53b5cb2985b14e63f91444cdc3b6e0c59e97969af98f29a828ac69ac1558e516`.
Per-run provenance preserves executable/source/input hashes and commands.

| Check | Result and evidence |
|---|---|
| One particle, COF launch, one turn, 100 ps | Maximum particle–reference distance **2.9303 pm**; reference return error **2.0745 pm**; COF position residual **0.2469 pm**. [Input](results/gaussian-bare-100ps-6so7oxpy/case.in), [checks](results/gaussian-bare-100ps-6so7oxpy/orbit-agreement.json). |
| Weak finite-bunch PIC coupling | All five cases pass. Charge doubling relative residual **1.7655×10⁻⁵**; response change on timestep halving **0.0024748**. Nonzero self E and self B verified. [Charged input](results/boris-pic-gexxdv3n/q1/case.in), [summary](results/boris-pic-gexxdv3n/summary.json). |
| Dedicated one-/two-rank PIC regression | Maximum component differences **8.7129×10⁻¹⁴ m** and **2.4980×10⁻¹⁵ mc**; equal final time. Fixed `1e-12` position/momentum gates pass. [Physical-identity comparison](results/boris-mpi-b0j3scpl/summary-file-identity.json), [fixture and mapping](results/boris-mpi-b0j3scpl/README.md). |
| N32 Gaussian bare probes (`TYPE=NONE`), 32 turns, 62.5 ps | All 32 survive. Mean Qx=**0.770108381022**, Qy=**1.689537407930**. Max |TRACK−COF|: **6.31×10⁻⁸** in x, **1.36×10⁻⁹** in y. Phase/frequency difference ≤**4.51×10⁻⁸**. All eight checks pass with the original 2 µm minimum signal (30 x / 26 y fits). [Input](results/gaussian-bare-62.5ps-1xm939m2/case.in), [comparison](results/gaussian-bare-62.5ps-1xm939m2/comparison.json). |
| Five probes, 32 turns, 125/62.5 ps | All six checks pass. Max |TRACK−COF| **1.10×10⁻⁸** coarse, **6.74×10⁻⁹** fine. Halving DT reduces analytic error by **3.97x/3.98y**; two-point dt² extrapolation agrees within **1.46×10⁻⁸**. [Coarse input](results/bare-tunes-125ps-m1b4x2wl/case.in), [fine input](results/bare-tunes-62.5ps-gb9k1ydp/case.in), [summary](results/boris-pic-bare-comparison/summary.json), [plot PNG](results/boris-pic-bare-comparison/bare-tunes.png)/[PDF](results/boris-pic-bare-comparison/bare-tunes.pdf). |
| Final source rebuild after interface wrapper/new frame unit test | **19 relevant C++ suites and 12 Python tests pass.** Final executable SHA-256 `ee9e50e96911a75b9ed71aae4af67181e7aa7083145203e5c112c0fe0ac83c51`; only wrapper routing, tests and comments changed after the numerical-core runs. [Final source/test provenance](results/boris-pic-integration-validation/summary.json); [final-binary N1 check](results/gaussian-bare-100ps-2kmpovap/orbit-agreement.json) is identical. |

The five-probe decks run COF separately and write its longitudinal momentum to
FROMFILE after verifying a zero transverse closed orbit. They do not exercise
`INITIALORBIT`; the N1/N32 Gaussian decks explicitly use `INITIALORBIT=ic`.

The weak PIC fixture has 32 Gaussian particles, RMS transverse size `50/3 µm`,
RMS longitudinal size 1 mm, an OPEN `16³` mesh, and one fixed `GAMMAZ` bin.
It compares bare, `1e-18 C`, and `2e-18 C` runs over 0.625 ns at 62.5 ps, plus
bare/charged repeats at 31.25 ps. The pre-existing 5% scaling and 10% timestep
gates are coupling smoke checks, not precision tune tolerances or evidence of
asymptotic convergence order.

Ten targeted C++ suites passed on this numerical build: `TestBendRep`,
`TestQuaternion`, `TestTrackReferenceStep`, `TestDeviceExternalField`,
`TestBorisStepControl`, `TestBorisMidpoint`, `TestBinning`, `TestSpaceChargeSolver`,
`TestCartesianPIC3DComponents`, and `TestCartesianPIC3DAlgorithms`. Dedicated
two-rank helper/cold-bin tests also passed. The MPI simulation compares physical
input-file rows, not rank-dependent native IDs. Accepted schedules differ across
rank counts, so bitwise reproducibility is not claimed.

The new PIC-frame tests separately verify translation invariance with exactly
representable geometry and the actual quantization of picometre offsets translated
by 825 m. This avoids confusing physical equivalence with a deliberately
ill-conditioned coordinate representation; existing E/B tolerances are unchanged.

Both edited manual pages rendered successfully and source validation passed.
Site-wide HTML validation is limited by the unrelated missing generated book
PDF; the separate documents repository checkout is absent.

## Reproduction

Run from `sandbox/Regression-Tests/tune-shift-ring`. Runners create fresh result
directories and print their paths; they enforce one rank/one thread and archive
provenance. Substitute those printed directories when analysing new runs.

```sh
~/.venv-h6/bin/python analysis/run_gaussian_bare.py --particles 1 --turns 1 --dt 1e-10 --sigma-xy 1e-12 --sigma-z 1e-12
~/.venv-h6/bin/python analysis/run_gaussian_bare.py --particles 32 --turns 32 --dt 6.25e-11
~/.venv-h6/bin/python analysis/run_bare_tunes.py --turns 32 --dt 1.25e-10
~/.venv-h6/bin/python analysis/run_bare_tunes.py --turns 32 --dt 6.25e-11
~/.venv-h6/bin/python analysis/check_boris_pic.py --binning one --run all --analyze
```

Reanalyse saved data without tracking:

```sh
~/.venv-h6/bin/python analysis/analyze_bare_tunes.py results/bare-tunes-125ps-m1b4x2wl results/bare-tunes-62.5ps-gb9k1ydp
~/.venv-h6/bin/python analysis/analyze_bare_tunes.py results/gaussian-bare-62.5ps-1xm939m2
~/.venv-h6/bin/python analysis/compare_gaussian_bare.py results/gaussian-bare-62.5ps-1xm939m2
~/.venv-h6/bin/python analysis/summarize_bare_tunes.py results/bare-tunes-125ps-m1b4x2wl results/bare-tunes-62.5ps-gb9k1ydp --output results/boris-pic-bare-comparison
~/.venv-h6/bin/python analysis/check_boris_pic.py --study results/boris-pic-gexxdv3n --analyze
~/.venv-h6/bin/python results/boris-mpi-b0j3scpl/compare_file_identity.py
```

## Remaining limits

This validates transport and weak finite-bunch coupling. It does not establish
a matched space-charge distribution, a substantial tune shift, long-time
collective stability, large-bunch performance, or GPU-hardware accuracy. Kokkos
OpenMP is tested; CUDA/HIP hardware is not. Shared refinement can require many
Poisson solves, and hard-edge accuracy still requires timestep/mesh studies.
The N32 bare run used 2,193,692 trial intervals, 794,838 rejected, in 187.7 s;
`NONE` performs no Poisson backend solve. This runtime does not estimate a
large charged-bunch run. Profile the shared-boundary solve cost before a long
particle/mesh convergence campaign.
The next physics milestone is a matched finite bunch and an analytic-versus-
tracking tune-shift comparison within a validated beam model.
