# Periodic ellipsoid analysis: handoff

Analytic matching and observable tools are complete. This agent changed no
production C++. It subsequently ran six zero-charge OPALX controls, each on one
MPI rank and one OpenMP thread, as described below.

Primary data: `results/ellipsoid-match-stable-long-d1center/`. Use its D1-centre
launch and observation matrices. The chosen family has zero-charge rms length
50 mm, six 0.1 m cavities at D2 centres, integrated slope +0.001 /m and harmonic 4.
Analytic frequency is 3.390638746 MHz, effective voltage 5119.24559 V, and initial
transit-corrected peak field 0.05121275555 MV/m. Root calibrates the actual RF.
Charges are 0, 10, 20, 40, 80, 120 and 160 pC. At 160 pC the conditional predicted
shifts are −0.154480565 horizontally and −0.203879307 vertically.

NPZ files contain full covariances, cell maps, trajectories, normal-mode bases,
signed phases and section transformations. Strong dispersion makes normal-mode
projection essential; a raw horizontal FFT may select the synchrotron mode.

Eleven matching tests and eight observable tests pass. Collective moment modes
are stable through 160 pC: multiplier moduli differ from unity by at most 1.4e−9.
Independent 32-turn moment evolution stays within 5e−9 scaled covariance error.
Tightening ODE tolerance 100 times changes tunes by less than 8.2e−12.

**Physical limitation:** uniform spatial density with conditional Gaussian
momenta is not a kinetic equilibrium, even with exact prescribed linear fields
and at zero charge. At 160 pC after one cell, 13.22% of particles lie outside the
rms ellipsoid and core density reaches 1.73 times the uniform expectation, while
covariance error remains only 0.00015. Repeating with 65k, 131k and 262k particles
gives 13.22–13.35% outside. More particles cannot fix this intrinsic deformation.

Independent Coulomb-pair whole-bunch gradient changes after one cell are
+1.42±0.18% in x, −5.40±0.58% in y and +0.76±0.10% in z. Errors are batch standard
errors. These are not central force gradients or particle tune shifts. Data and
visually checked PNG/PDF figures: `results/ellipsoid-kinetic-consistency/`.

Root has accepted retaining the uniform closure as a conditional prediction
while measuring actual PIC density and tune footprints. Do not claim an exact
matched kinetic equilibrium or use covariance agreement as proof of uniformity.

Runner tools are `analysis/ellipsoid_observables.py` and
`analysis/analyze_ellipsoid_modes.py`. Input NPZ contains `phase` with shape
(events, particles, 6), `turn_coordinate`, and `ids`. Apply the shifted inverse
mode basis; the complex signal q−ip has positive x/y and negative z phase. Supply
an explicit centroid when needed. For the paired probe deck only, CLI option
`--central-id 0` subtracts its reference particle. Independent phase and spectral
estimates, residuals, amplitudes, coherence and inactive-mode flags are retained.
The runner agent has been informed.

Study report: `MATCHED_ELLIPSOID.md`. Remaining root work: calibrate RF bare
tracking, validate PIC forces and convergence, and measure the many-particle
charge-dependent tune footprint with density evolution recorded.

## Independent fixed-step controls (2026-09-10)

Completed 1 mm and 5 mm pure normal-mode probes at 400/200/100 ps, 13 particles,
32 turns: `results/finite-amplitude-fixed-rf-qft9g9ng/`. Both 400 ps cases fail the
unchanged section ordering guard. Every valid 100/200 ps case fails the 0.001
horizontal/longitudinal tune-error target; maximum individual errors reach 0.029,
and centre/tracked-reference differences reach 9–33 cm. No tested timestep is
validated, and no resolved 5 mm redo is justified by these failures. Root agrees.
No active simulation started by this agent remains. Full raw HDF5 is preserved
as SHA-verified gzip; inputs, logs, provenance, sections and reference histories
remain immediately accessible.

Report: `FIXED_STEP_RING_AUDIT.md`. Reproducible independent audit/plot scripts:
`analysis/audit_fixed_step_observations.py`,
`analysis/summarize_fixed_step_controls.py`. Three associated unit tests pass.
The chart PNG/PDF was visually inspected. Root's original 10 µm/400 ps run was
audited independently, separating nominal-COF departure from particle/reference
disagreement; both are substantial and the latter grows from roundoff-scale
differences over the first few turns. No tolerance or estimator masking was used.

Completed bounded mathematical review: `EDGE_AVERAGING_MATH_AUDIT.md` and
`results/edge-average-independent-audit/`. The original weighted map reverses to
roundoff in the oblique test; its endpoint-moment variant has O(h²) momentum
round-trip error. Exact known-time nonparallel magnetic switching independently
shows an O(h²) missing commutator impulse in one averaged combined kick, with a
measured O(h³) remainder. Position-only correction cannot remove it. Three new
oracle tests and the prototype's eight tests pass. No production implementation
or further method search is recommended on this evidence. Root and boundary
agent have the result; no source changes or running jobs remain for this agent.

Final geometry-only control by the boundary agent is favorable for a face normal
to the reference orbit: `results/edge-average-prototype-dor56wcq/` recovers
approximately second-order full-slab position/momentum derivatives with the
endpoint-moment variant. This table was independently inspected and added to the
math audit. The oblique-face, reversal and noncommuting-field limits remain;
normal incidence must not be generalized to arbitrary supports or finite-beam
trajectories without further evidence.

New bounded read-only audit (same date): `FROZEN_MIDPOINT_REPLAY_AUDIT.md` assesses
one shared predicted-midpoint PIC solve followed by independent combined Boris
replay through external boundaries with particle-labelled frozen self fields.
It has a credible second-order argument for regular, accurately centred self
fields, but the deposit is not the replay's exact midpoint. Current BINS=ONE
also uses unchanged P_n to construct source mean velocity/gamma and Lorentz
fields, so position accuracy alone does not establish second order when source
momentum changes. A source-moment predictor or explicitly limited approximation
must be specified. The proposal is generally not exactly reversible. Root and
boundary agent were informed; the latter's event/saltation Jacobian oracle for
E=A R and constant self B was checked independently. No implementation or OPALX
simulation was performed for this new audit.

## Source boost algebra (2026-09-11)

Completed the bounded composition-only direction-lag audit in
`SOURCE_BOOST_AUDIT.md`, `analysis/source_boost_audit.py` and
`results/source-boost-audit/`. Five regression checks pass. For fixed common-axis
E', gamma and nominal co-moving test velocity, exact relative-force operator
bounds are 0.911725/1.823479/5.836250/14.604137 ppm at 62.5/125/400/1000 ps for
beta=.0728541 and rho=2 m. These exclude potential/stretch/frame changes, PIC
noise, source-energy/trajectory error and accumulated tune error. The general
Lorentz map permits distinct source/test velocities; the numerical table uses
the nominal cold reference relation. Root's separate continuum ellipsoid
extension was checked read-only for normalization, signs, dimensions and frames;
no inconsistency was found. No production source or OPALX simulation changed.
