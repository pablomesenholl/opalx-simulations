# Independent mathematical audit of the edge-average candidate

The residence average removes the finite kick discontinuity of point support
sampling and shows second-order state errors in the bounded slab test. It does
not establish second-order first variations, or a DBA timestep adequate for
0.001 tune accuracy. The tested endpoint first-moment correction improves one
entrance derivative but does not resolve this limitation for an oblique exit.
A final control with an exit face normal to the reference orbit does recover
second-order derivatives, so the limitation is geometry dependent. The variant
also loses the original weighted map's exact reversibility in an oblique
crossing test.

This is a bounded review of `EDGE_AVERAGING_INVESTIGATION.md` and
`analysis/edge_average_prototype.py`, including its `weighted_moment` variant.
No OPALX production source, public input API or PIC method was changed. No broad
integrator search was performed. The independent script and results are in
`results/edge-average-independent-audit/`; the reviewed variant's own convergence
study is `results/edge-average-prototype-7jb_q8_a/`.

## 1. A field average cannot fix both ordinary drifts

Consider a particle with constant longitudinal speed, with constant transverse
acceleration **a** switched on at time τ in a step of duration h. This is an
exact elementary counterexample and is also the leading local expansion for a
non-grazing magnetic edge. The exact impulse and transverse displacement are

    Δv = a (h−τ),
    Δr_exact = h v0 + a (h−τ)² / 2.

If the averaged combined kick supplies that correct impulse, the unchanged
drift–kick–drift position update necessarily gives

    Δr_DKD = h v0 + h a (h−τ) / 2,
    Δr_DKD − Δr_exact = a τ(h−τ) / 2.

With τ≈−z0/vz, differentiating this O(h²) state error with respect to launch
position gives an O(h) first-variation error. This does not contradict
second-order orbit convergence. The error coefficient changes as an edge moves
through the timestep, over a launch-position scale proportional to h.

For a force varying over the step, the leading missing position correction is
its first temporal moment, schematically

    δr = ∫₀ʰ (h/2−t) a_ext(t) dt.

For relativistic momentum coordinates the leading expression uses the velocity
Jacobian Dv/Dp times the corresponding force moment. A constant midpoint self
field has zero first temporal moment. This observation motivates the prototype's
endpoint correction, but does not prove consistency of the whole method.

The implemented correction is

    δr = (h/2 − t̄_active) [v(p_combined) − v(p_SC_only)],

where the active-time centroid comes from the original two uncorrected
half-drift segments. The difference of velocities isolates the external force
only to leading order; the residence predictor and PIC midpoint stay unchanged.
The variant therefore changes the position update while retaining an earlier
uncorrected path to calculate its own edge timing.

The variant's slab results confirm the limited benefit. At one entrance both
position and momentum launch-z derivatives improve to second order. Across the
complete entrance/body/exit map, both remain first order. At h=0.005, the maximum
position derivative error changes from 0.002389 to 0.000413, while the momentum
derivative error remains 0.000329. This is a useful improvement, not a complete
edge-map correction.

The boundary agent's final geometry-only control,
`results/edge-average-prototype-dor56wcq/`, changes the exit to the plane normal
to the exact exiting orbit, n=(−sin α,0,cos α) with α=asin(BL/pz), without changing
the candidate algorithm. I independently inspected its retained table. Here
the endpoint-moment variant has approximately second-order position and momentum
derivatives over the full slab: at h=0.005 their maximum errors are 2.22e−6 and
9.74e−7. Its momentum state error is approximately cubic; position state error
remains quadratic. This favorable case is relevant to sector faces normal to
a design orbit. It does not cover finite off-orbit incidence, general field
supports, noncommuting forces or the reversal test below. The ordinary residence
average without the moment correction remains first order in derivatives.

## 2. A position-only correction also loses reversibility

The original weighted DKD is algebraically symmetric when its residence fixed
point is unique and converged. Reversing the step swaps the same two half-drift
segments, so the same support fraction and midpoint field are available. This
statement assumes static fields and consistent time reversal; it is not a
symplecticity proof or an arbitrary-geometry convergence guarantee.

The independent check reverses momentum and magnetic fields while leaving the
electric field unchanged, which implements physical time reversal for the test.
It uses an oblique entrance, p=(0.2,0.05,1) in units mc, body By=0.8, self
B=(0.01,0.03,0.02), and spatial E(r)=−0.1r in the prototype's dimensionless units.
The maximum is over entry fractions 0.2, 0.5 and 0.8.

| h | Weighted momentum round trip | Endpoint-moment momentum round trip | Endpoint-moment position round trip |
|---:|---:|---:|---:|
| 0.08 | 0 | 4.4898e−5 | 1.4836e−6 |
| 0.04 | 1.11e−16 | 1.2003e−5 | 1.9527e−7 |
| 0.02 | 1.11e−16 | 3.0980e−6 | 2.5031e−8 |
| 0.01 | 1.11e−16 | 7.8669e−7 | 3.1681e−9 |
| 0.005 | 1.11e−16 | 1.9819e−7 | 3.9846e−10 |

After the endpoint correction the reverse first drift reaches a different
midpoint and predicts a different support interval. A single final coordinate
adjustment does not retain the original map's symmetry. The table establishes
this concrete defect; it does not by itself quantify long-term emittance growth.

## 3. Even exact residence leaves a noncommuting impulse term

A separate oracle removes geometry and crossing prediction entirely. A known
switch at τ changes a constant magnetic generator A0 into A0+A1. Relativistic
magnetic motion preserves |p|, so γ is constant and these are exact linear
rotation generators satisfying A(B)p = p×B/γ in the prototype's units.

The exact momentum map and the leading averaged-kick map are

    M_exact = exp[(h−τ)(A0+A1)] exp[τ A0],
    M_average = exp[h A0 + (h−τ) A1] + O(h³).

Expansion through second order gives

    (M_exact − M_average)p
        = τ(h−τ) [A1,A0]p / 2 + O(h³),
    [A1,A0] = A1 A0 − A0 A1.

Nonparallel magnetic fields give a nonzero commutator. A combined Boris kick
using the exact average has the same leading O(h²) defect; its ordinary smooth
Cayley error enters at O(h³). A position-only correction cannot alter that
momentum defect. Differentiating τ with respect to launch position again gives
an O(h) first-variation term. Parallel/commuting special cases do not test this
general interaction.

The independent example uses p=(0.2,0.3,1), self B=(0.7,0,0.1), body B=(0,0.8,0),
and τ=0.3h. At h=0.02, 0.01 and 0.005 the impulse errors are 4.9113e−6,
1.2455e−6 and 3.1359e−7. Subtracting the displayed commutator term leaves
1.5748e−7, 1.9704e−8 and 2.4642e−9: the expected cubic remainder. These field
strengths are a mathematical counterexample, not a quantitative model of this
ring's space-charge magnetic field. Electric/magnetic time ordering requires
similar care; it has not been claimed numerically validated here.

## Bounded recommendation

Do not describe either candidate as a validated replacement for the exact-edge
ring method. A first-order tangent error may eventually meet a tune tolerance at
a small enough timestep; the present mathematics does not prove that it cannot.
It also supplies no evidence of a sufficiently large practical timestep for
100k particles. That requires ring-specific first-variation and tune controls.

If the two ordinary drifts and one combined averaged kick must remain literally
unchanged, the position counterexample rules out simultaneous correction of
state impulse and edge displacement by changing the field average alone. A
simple endpoint first-moment addition is insufficient for the more general
requirement: corrected path timing, inverse-map consistency and noncommuting
momentum effects must also be addressed.

Addressing these effects does not mathematically require splitting external
transport from space charge; a more complete integration method could retain a
single common midpoint field solve and keep all particle work on device. But it
would be a numerical-method change, with a new consistency argument and tests,
not merely additional element selection. No particular untested method is
recommended by this audit. Avoid production implementation until a bounded
candidate passes complete entrance/exit state and derivative tests, reversibility
characterization and the three-mode DBA control at an adequate timestep.

The standalone oracle and its three unit tests can be rerun without OPALX:

```sh
cd sandbox/Regression-Tests/tune-shift-ring/analysis
~/.venv-h6/bin/python -m unittest test_edge_average_audit.py
~/.venv-h6/bin/python audit_edge_average_math.py \
  ../results/edge-average-independent-audit
```

All three tests pass. The reviewed prototype's eight tests also pass. Units,
field definitions and source snapshots accompany the retained CSV data. Curved
supports, grazing faces, RF time dependence, self-consistent PIC, MPI and device
performance remain outside this mathematical audit.
