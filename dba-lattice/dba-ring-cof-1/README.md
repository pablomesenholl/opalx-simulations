# Stable hard-edge DBA closed-orbit benchmark

This issue #544 experiment is the historical 250 MeV electron closed-orbit
benchmark. The common geometry and optics are described in the
[LaTeX note](../TeX/main.tex), Section 2. Its input uses `K1=2.84 m^-2`,
DOP853, `DT=5 ps`, one Richardson level, and a 3e-5 finite-difference
position/slope scale.

`analysis/generate_analytic.py` independently scans the one-family exact hard-edge optics,
selects the published stable point, and writes the maps, principal phases,
periodic dispersion and circumference. No OPALX tracking is used in that solution.

`./run.sh` regenerates the analytic result and runs both the nominal and displaced COF seeds on one
MPI rank with one OpenMP thread.  Each tracked 4x4 mechanical-momentum map is converted to slope
coordinates before comparison.  Use a new `RUN_NAME` to preserve existing results.

## Fresh baseline

`results/baseline-20260909` at revision `fd675e885` reproduces both seeds as `STABLE`.  The nominal
case has a 6.31e-9 maximum matrix error and 3.28e-11 maximum normalized-momentum closure residual.
The displaced seed converges back to the origin with a 4.42e-9 matrix error and 2.60e-11 maximum
residual.  The retained analytic principal fractional phases are 0.229892430867 and 0.310463602980.
