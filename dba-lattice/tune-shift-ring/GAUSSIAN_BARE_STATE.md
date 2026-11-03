Completed single-input COF -> GAUSS INITIALORBIT launch and32turn benchmark.
Input bare-gaussian-cof.in; results/gaussian-bare-62.5ps-gro2v5p1 completed32particles.
See GAUSSIAN_BARE.md for authoritative results and limitations.
Qx mean.770108295518 vs previous.770108373744; Qy1.689537408282.
All particles survive;5 Python tests pass. Comparison precision checks NOT allpass:
horizontal individual/COF1.85e-6, independent2.28e-6, half-record4.53e-6.
Mean tune comparison passes; report retained failures, no tolerance changes.
No C++ changes,one rank/thread,Kokkos OpenMP; sigma_z1pm thin slice,FSNONE.
Next optionalphysicswork: diagnose horizontal scatter before highprecisionclaim.
