# PR553 tracking recheck — 2026-09-10

Goal: determine whether current PR553 removes the small-ring tracking blocker.

Remote `refs/pull/553/head` and local branch
`544-closed-orbit-finder-for-opalx` both resolve to
`288d4327cedc742e8bf4c32878070d32a2f5e15e`. No checkout change was needed.
The experiment has moved to `sandbox/Regression-Tests/tune-shift-ring`.

Relevant changes in this head include final directed return-plane event timing
(`2db119cda`) and named COF results consumed by TRACK INITIALORBIT (`288d4327c`).
The return-plane correction is restricted to static magnetic, single-container,
space-charge-off tracking. It does not itself validate space-charge tracking.

Reconfigured and rebuilt `omp-build`. The initial incremental build retained
the old embedded revision label; reconfiguration rebuilt Util.cpp and relinked
the executable. Final run logs explicitly report `288d4327c`.
TestTrackReferenceStep and TestClosedOrbitInitialState pass, each on one rank.

Reproducible diagnostic: `analysis/check_pr553_tracking.py --dt <seconds>`.
It retains exact input, binary SHA256, revision, HDF5, trajectory CSV and summary.
Each case uses one proton, one MPI rank, one thread, 2.5 MeV, FS=NONE, one turn.
Two launches are compared: the previous FROMFILE design-orbit particle, and a
near-point generated distribution placed using a freshly solved named COF.
The latter uses 1e-12 m Gaussian widths and zero momentum spread, complying with
the current INITIALORBIT restriction against FROMFILE distributions.

Final 100 ps runs: `results/pr553-tracking-a3vk_gbw/`.
Final 25 ps runs: `results/pr553-tracking-rnj2oy0v/`.
Earlier duplicate runs with the old embedded banner are retained in
`results/pr553-tracking-dbuc2ngp/` and `results/pr553-tracking-ar9jky8o/`.

| DT | Launch | Final particle x relative to reference | Final local z |
|---|---|---:|---:|
| 100 ps | FROMFILE | 40.788500 mm | −256.629352 mm |
| 100 ps | INITIALORBIT | 40.788429 mm | −256.629294 mm |
| 25 ps | FROMFILE | 7.253902 mm | −9.743710 mm |
| 25 ps | INITIALORBIT | 7.253900 mm | −9.743711 mm |

At 100 ps, both cases already exceed 1 mm horizontal separation by a saved
path position of 5.4603 m, and reach about 83.06 mm near s=21.4043 m. Thus the
separation develops well before the final return event. The final reference
does reach the launch z-plane to roundoff, which demonstrates the timing fix;
it does not eliminate the particle/reference trajectory discrepancy.

The named COF converges to zero transverse coordinates with x residual8.44e-14m
and STABLE linear map. Its input therefore provides the intended orbit.
The new launch feature does not materially change this ring's result.

Conclusion: PR553 addresses initialization and final-plane timing, but the
small-ring particle/reference tracking discrepancy remains at these timesteps.
The previously identified field selection/support-boundary treatment remains
a candidate cause; this recheck does not re-instrument fields or prove the
precise current cause. Do not use these runs as validated SC tune measurements.

No production source, physics inputs outside this diagnostic, or tolerances
were changed. Existing files preserved. Next step: diagnose/fix the interior
particle/reference field/step disagreement before the SC comparison.

## Boris in both COF and TRACK

User-requested controlled rerun: COF TIMEINTEGRATOR=BORIS and TRACK's Boris
particle pusher, both with nominal DT=25 ps, one rank/one thread, FS=NONE.
Input and results: `results/pr553-tracking-4zims0pl/cof_initialorbit/`.

COF converges at zero transverse coordinates with residual x=−2.21035e-15 m,
px=−2.99371e-15 in p/(mc), and STABLE map. TRACK still ends at
x=7.253900264 mm, z=−9.743711107 mm relative to its moving reference.
These TRACK values are identical to the earlier DOP853-initialized run because
both COF solutions provide the same launch state. The DOP853/Boris choice is
therefore not sufficient to explain this discrepancy. Nominal timesteps now
agree, but field selection and boundary substepping remain different code paths.

Reproduce with:
`~/.venv-h6/bin/python analysis/check_pr553_tracking.py --dt 2.5e-11 --cof-integrator BORIS --launch cof_initialorbit`.
The diagnostic now exposes integrator/launch options and records COF residuals
alongside TRACK offsets. Production code and existing runs were unchanged.
