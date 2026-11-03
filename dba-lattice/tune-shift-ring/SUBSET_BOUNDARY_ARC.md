# Shared boundary-refinement control-subset test

This test evaluates the user-approved approximation in which a small stable-ID
particle subset may request a smaller *shared* Boris/PIC timestep. Every one of
the 131,072 particles still deposits charge, receives its own space-charge and
element fields, and advances with the accepted common timestep. The reference
particle and all-particle finite-state checks remain active. The test does not
measure a tune and does not establish a matched kinetic equilibrium.

## Completed run

The reproducible study is
`results/subset-boundary-arc-b1j77tzi/`. It uses the frozen executable whose
SHA256 is
`2868bdde8d1088e4b4cd4f9912e6ae898937983c2b1b609b51c8521d2ec6e68e`, one MPI
rank, eight OpenMP threads, 160 pC total charge, a 32^3 open PIC mesh, and the
same fixed Sobol/ellipsoid launch for each case. The common endpoint is 32.5 ns,
just after the first quadrupole exit (entry 22.8926 ns, exit 32.0496 ns), before
the next RF element. The FROMFILE phase-space round trip is 3.39e-16 in the
stored normalized coordinates.

Both cases used 62.5 ps nominal steps:

| case | crossing-control particles | elapsed | boundary trials | rejected trials | final time |
|---|---:|---:|---:|---:|---:|
| `control4-dt62p5ps` | 4 | 83.50 s | 1,098 | 289 | 32.500 ns |
| `control32-dt62p5ps` | 32 | 308.66 s | 4,166 | 1,823 | 32.500 ns |

The controls are nested stable-ID SplitMix64 samples: stride 16,384 selects 4
IDs and stride 4,096 selects 32 IDs. The startup log verifies both the stride
and the realized count. The 32-particle sample costs 3.70 times the 4-particle
sample and produces 3.79 times as many boundary trials.

The final phase spaces are compared particle-by-particle in the same initial
D1 Frenet frame, with no per-case centroid subtraction. The whitening metric is
fixed by the initial covariance. Comparing the 4- and 32-particle controls:

| metric | result |
|---|---:|
| whitened covariance difference, \(\|C_0^{-1/2}(C_a-C_b)C_0^{-1/2}\|_F/\sqrt{6}\) | 1.9216e-4 |
| largest centroid difference / initial RMS component | 5.92e-6 |
| particle error p99, all 131,072 particles | 2.6892e-2 |
| particle error p99, the 4 selected particles | 6.0608e-7 |
| particle error p99, unselected particles | 2.6892e-2 |
| largest particle error over the full bunch | 6.8225e-2 |

The selected particles agree closely because both runs refine around those
same IDs. The unselected particles do not: their finite-step edge treatment
changes as the control set changes. Thus a control sample is useful for reducing
the cost of shared refinement, but it cannot be treated as a tune-accuracy
certificate for the rest of the bunch. The two full-bunch density diagnostics
are nearly unchanged (outside-ellipsoid fraction 0.0438614 versus 0.0438538),
but that agreement does not remove the particle-level edge error.

The particle differences above are six-dimensional displacements whitened by
the initial covariance and divided by sqrt(6). In particular, 0.026892 is not a
2.7% tune error or a relative coordinate error. Two samples at one timestep do
not establish whether a subset can or cannot produce converged tunes. They
also do not establish that full-population boundary triggers are mandatory for
the final study. Timestep/control-size convergence is the missing evidence.
The previous years-long full-run estimates extrapolated 4/32-control timings
far beyond the measured range and are not reliable runtime forecasts.

## Default-behavior smoke test

In `results/subset-boundary-arc-7igow9u6/`, 64-particle solver-free runs with
the environment unset and with `OPALX_TEST_BOUNDARY_CONTROL_STRIDE=1` are
bitwise identical at 40 ns: all six phase-space components, boundary counts,
charge, and density diagnostics match exactly. This verifies that the default
path and stride 1 retain the previous behavior.

## Remaining cases

The 139-particle and full-population cases are prepared in the 32.5 ns study but
were not needed to establish the first sensitivity result. An exploratory 40 ns
139-particle run in `results/subset-boundary-arc-7igow9u6/` was intentionally
stopped at about 23.5 ns before its final snapshot; it is recorded as
`interrupted`, not as a physics result. A tune scan still requires timestep and
control-size convergence over multiple cells/turns.

A separate full-population one-cell case was subsequently run on the same
MPI1/OMP8 configuration with the same initial particles. It reached its 600 s
wall limit at 18.844 ns of 196.625 ns, during the bunch head's first quadrupole
entrance crossing. No final snapshot was produced. That deck omits RF; see
[the bounded feasibility result](results/full-population-one-cell-rlrm1m2z/RESULT.md)
for the exact configuration and interpretation limits.

Raw endpoints and machine-readable tables are in `cases.csv`,
`comparisons.csv`, `comparisons.json`, each case's `analysis.json`, and the
corresponding `case.h5` files. The runner is
`analysis/subset_boundary_arc.py`.
