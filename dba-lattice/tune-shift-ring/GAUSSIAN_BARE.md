# Gaussian distribution launched from COF: bare tunes

The single input executes `ic: COF` and then `TRACK, INITIALORBIT=ic` with a
generated `DISTRIBUTION, TYPE=GAUSS`. Overall tunes agree with the preceding
five-particle benchmark, but several horizontal per-particle precision checks
fail. This is a successful launch/comparison, not a claim that all tune-quality
checks pass.

Input: [bare-gaussian-cof.in](bare-gaussian-cof.in).
Completed run: [case.in](results/gaussian-bare-62.5ps-gro2v5p1/case.in).
This input is self-contained: it uses neither FROMFILE nor particles.txt.
Distribution/source/beam definitions precede COF; their particles are generated
when TRACK runs after COF. INITIALORBIT consumes the solved state directly.

Parameters: sigma_x=sigma_y=50e-6/3 m; CUTOFFX=CUTOFFY=3; sigma_z=1e-12 m
(a numerical thin slice); SIGMAPX=SIGMAPY=SIGMAPZ=0; 32 particles; seed20260910;
32 turns; DT62.5ps; FSNONE; one MPI rank and one OpenMP thread. Kokkos particle
transport, same executable hash as the preceding benchmark. No CUDA/HIP hardware
was tested. Charge is 32 elementary charges and has no self-field effect with
FSNONE. This is a cold, unmatched distribution, not a matched space-charge bunch.

COF converged at x=px=y=py=0 with maximum positional residual4.29e-14m. Launch
logging confirms zero transverse momentum spread, mean transverse position
consistent with zero, and sampled RMS x=19.09015um/y=15.44518um. The requested
16.6667um is the parent Gaussian sigma; finite sampling and truncation change the
realized RMS. Gaussian sampling subtracts its sample mean after applying cutoff,
so the cutoff is not a strict final-coordinate bound after recentering. The first
recorded snapshot nevertheless lies within ±50um: x[-32.889,+34.172]um,
y[-32.145,+37.935]um. This snapshot is after12.3125ns, not exactly launch.

All32 particles survive through the last of3067 HDF5 snapshots. The planar
reference remains y=0. As before, tune extraction uses each particle's own six
cell crossings and local interpolation of deviations from the circular bend.
There is no artificial central probe in this Gaussian ensemble.

| Method | Qx | Qy |
|---|---:|---:|
| Analytic |0.770107569133|1.689536397020|
| Previous five-particle TRACK,62.5ps |0.770108373744|1.689537408351|
| Gaussian TRACK mean |0.770108295518|1.689537408282|
| Gaussian TRACK median |0.770108381684|1.689537408282|

Before examining results, a2um first-snapshot signal threshold was chosen to
exclude near-zero-amplitude phase fits. This retained30 horizontal and26 vertical
fits; all32 particles were still tracked. The full exclusions and results are in
[comparison.json](results/gaussian-bare-62.5ps-gro2v5p1/comparison.json) and
[tunes.json](results/gaussian-bare-62.5ps-gro2v5p1/tunes.json).

Mean-minus-previous is -7.82e-8 in x and -6.85e-11 in y. Vertical individual tunes
span only1.75e-10. Horizontal individual tunes span0.770106524420–0.770109840305.
The maximum tracking/COF discrepancy is1.85e-6, independent-fit difference2.28e-6,
and first/second-half difference4.53e-6. These fail the preselected per-particle
checks (1e-6,1e-6,2e-6 respectively). Interpolation order agrees within1.44e-10,
and relative energy variation is below6.69e-10. The cause of the larger horizontal
scatter has not been isolated; do not assert that it is a physical tune spread
or that low amplitude alone fully explains it.

For context only, restricting to first-snapshot amplitudes>=10um, comparable to
the previous probes, leaves19 x and17 y fits and gives mean Qx0.770108383372,
Qy1.689537408283. Maximum x tracking/COF difference is4.58e-7, half-record
difference1.01e-6. This secondary subset does not replace the original failed
checks. No thresholds were relaxed and no C++ transport changes were made.

Changed files: input deck above; new analysis/run_gaussian_bare.py and
analysis/compare_gaussian_bare.py; analysis/analyze_bare_tunes.py extended to
handle a generated ensemble without assuming particle0 is central. The existing
five extraction unit tests pass. Python compilation checks pass. The comparison
script intentionally exits nonzero because three precision checks fail, while
writing the complete report data.

Reproduce from this directory using /Users/adelmann/.venv-h6/bin/python:

```sh
python analysis/run_gaussian_bare.py
python analysis/analyze_bare_tunes.py results/gaussian-bare-62.5ps-gro2v5p1
python analysis/compare_gaussian_bare.py results/gaussian-bare-62.5ps-gro2v5p1
```

The runner prints a fresh directory; substitute it for subsequent analysis.
The existing input can also be copied to a fresh directory and executed with
OMP_NUM_THREADS=1 /Users/adelmann/git/opalx/omp-build/src/opalx case.in.
Keep the completed benchmark output unchanged.

Next possible investigation: resolve the horizontal per-particle scatter using
longer records and/or independent section-map diagnostics before claiming the
previous benchmark's precision. A finite bunch length, matched momenta, realistic
charge, and suitable solver still need to be specified for space-charge studies.
