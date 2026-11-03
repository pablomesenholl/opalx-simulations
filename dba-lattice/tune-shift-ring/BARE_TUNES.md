# DBA bare-tune verification

The hard-edge analytic model and device particle tracking agree. Both tune errors
fall by approximately four when the internal Boris timestep is halved.

| Method | Qx | Qy |
|---|---:|---:|
| Analytic hard-edge cell maps | 0.770107569133 | 1.689536397020 |
| Device TRACK, 125 ps | 0.770110773280 | 1.689540418975 |
| Device TRACK, 62.5 ps | 0.770108373744 | 1.689537408351 |
| Host COF, 125 ps | 0.770110784311 | 1.689540428352 |
| Host COF, 62.5 ps | 0.770108375122 | 1.689537407920 |

TRACK entries average four probes. These digits identify the recorded result;
they do not imply twelve-digit accuracy. The largest first/second-half tune
difference is 1.30e-7. Two-step, leading-dt² extrapolation differs from the
analytic tunes by at most 3.01e-8 across all probes. This is a two-step consistency
check, not a high-order convergence study.

## Ring and launch

The baseline lattice and analytic tune convention are specified in the
[LaTeX note](../TeX/main.tex), `sec:hamiltonian`. The tracking settings are
FSNONE; Boris; 32 turns; one MPI rank and one OpenMP thread.
Section beta_x=7.7381560003 m, beta_y=1.7591155260 m and alpha≈0 in both planes.

Five physical particles: one at the COF origin and four with x=y=±10 or ±50 µm,
zero transverse momentum and the same COF longitudinal momentum. The symmetric
centroid leaves the tracker reference on the planar closed orbit. Both planes
are excited simultaneously. This checks small-amplitude tunes, not arbitrary
large-amplitude dynamic aperture. The central particle is excluded from tune fits.
All five particles survive. Maximum central particle/reference displacement:
77.3 nm at 125 ps and 173.4 nm at 62.5 ps after 32 turns; this small accumulated
tracking/reference difference remains and is not claimed to be identically zero.

## Measurement and checks

HDF5 contains reference-relative particle coordinates. Reconstruct laboratory
positions and momenta using the planar reference pose, then locate each particle's
own forward crossings of the six equivalent Ck_B1 entrance planes. Do not sample
all particles at the reference return time: arrival offsets mix with bending motion.

At a cell boundary, interpolate small radial and momentum deviations from the
known radius-2-m circular design trajectory of the adjacent bends. They equal
Cartesian transverse position and momentum at the plane. Use local fifth-degree
polynomials, not global splines across hard edges. Discard crossings without a
complete interpolation stencil. Fit unwrapped phase of x+i(alpha*x+beta*x')
(or y analogue), with sign chosen for increasing betatron phase. Six samples per
turn resolve both full tunes without assigning the tracking integer from theory.
The analytic beta/alpha only normalize phase-space coordinates.

An independent windowed sinusoid fit to section positions agrees within 6.56e-8.
Third- versus fifth-degree interpolation changes the tune by at most 1.44e-10.
Tracking–COF differences are below 2.30e-8. Maximum relative kinetic-energy
variation is 6.84e-10. The finer run's horizontal tune spread across ±10/±50 µm
probes is 4.27e-8; the vertical spread is 3.42e-10. Five Python unit tests pass,
including synthetic known full tunes and geometric crossing/frame reconstruction.
No C++ physics code was changed for this measurement.

COF comparison uses the argument of each matrix eigenvalue, selecting the
orientation from M12 and the integer branch from cell optics. The coarse COF
JSON labels its finite-difference matrix NON_UNIT_CIRCLE: the horizontal modulus
defect is 1.70e-8 (2.51e-9 at the finer step). Do not interpret that strict flag
alone as physical instability; the finite-length tracked trajectories remain
bounded. No COF stability tolerance was changed.

## Reproduction and files

Run from this directory with /Users/adelmann/.venv-h6/bin/python:

```sh
python analysis/run_bare_tunes.py --dt 1.25e-10 --turns 32
python analysis/run_bare_tunes.py --dt 6.25e-11 --turns 32
python analysis/analyze_bare_tunes.py results/bare-tunes-125ps-5cm30zvp results/bare-tunes-62.5ps-ijgjxg2t
python analysis/summarize_bare_tunes.py results/bare-tunes-125ps-5cm30zvp results/bare-tunes-62.5ps-ijgjxg2t --output results/bare-tune-comparison
python -m unittest discover -s analysis -p test_bare_tune_tools.py -v
```

New runner invocations print fresh directory names; substitute those for analysis.
Each accepted run includes case.in, particles.txt, case.h5, run.log, COF input/JSON,
config, source diff, new device source files, binary/revision/analysis hashes and
tunes.json. Binary: omp-build/src/opalx, SHA256
338c59b7ca7545df2ec65f63c9e57204b8f95ad645eb96734e5172fbf6a02953;
base revision 288d4327cedc742e8bf4c32878070d32a2f5e15e with existing uncommitted
device-tracking changes. Kokkos OpenMP execution was tested; CUDA/HIP hardware was
not tested. All study runs use one MPI rank.

Accepted runs:
- [125 ps](results/bare-tunes-125ps-5cm30zvp/case.in)
- [62.5 ps](results/bare-tunes-62.5ps-ijgjxg2t/case.in)
- [Comparison data](results/bare-tune-comparison/tune_comparison.csv)
- [Validation checks](results/bare-tune-comparison/summary.json)
- [Plot](results/bare-tune-comparison/bare-tunes.png), [PDF](results/bare-tune-comparison/bare-tunes.pdf)

Exploratory asymmetric 0,+10,+50 µm runs are retained but excluded from the
accepted table: their moving reference follows an off-axis centroid. This changes
the output frame, not the laboratory launch positions. Their 500/250 ps nominal
steps both subdivide to 125 ps internally, so that pair cannot test internal-step
convergence. Initial 100/50 ps tracking runs were canceled; only their COF runs
completed. Initial exploratory runs also dumped every step because COF time_s
was mistakenly read as a period; it is launch time. Final runs use circumference
and speed and dump approximately 96 samples/turn. These issues were confined to
the experimental setup/analysis, not changes to OPALX transport in this task.

