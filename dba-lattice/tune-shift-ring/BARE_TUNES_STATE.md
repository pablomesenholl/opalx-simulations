Completed bare-tune verification. Authoritative report: BARE_TUNES.md.
Accepted runs: results/bare-tunes-125ps-5cm30zvp and bare-tunes-62.5ps-ijgjxg2t.
Analytic reference: [LaTeX note](../TeX/main.tex), `eq:bare-tunes`.
32turn symmetric0,±10,±50um x=y probes, FSNONE, one rank/thread, Kokkos OpenMP.
Tracking mean62.5ps Qx=.770108373744 Qy=1.689537408351.
Tracking–COF max2.30e-8; dt-halving error≈4x; extrap error<=3.01e-8.
Extraction uses own six cell crossings, local radial-deviation interpolation,
phase unwrapping, independent real-sinusoid fit; max third/fifth order1.44e-10.
Five Python unit tests pass. Summary checks all pass; plot visually reviewed.
No C++ edits this task; previous device changes preserved. GPU not tested.
Exploratory canceled/asymmetric runs excluded; see report for setup corrections.
Remaining: final response with report and plot links. Next physics work could
compare space-charge tune shifts against a bare baseline at the same timestep.
