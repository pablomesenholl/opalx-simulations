# Frozen midpoint PIC fields with particle-local Boris replay

The proposed method has a credible second-order consistency argument for smooth,
accurately centred self fields. It could remove the expensive requirement that
one particle's edge crossing triggers another collective PIC solve. It is a
**material integration change**, not just field selection, and presently has no
validated production timestep. This audit changed no OPALX source or simulation.

## Method and conditional accuracy

For one common interval h, predict every source position as
R*=R_n+(h/2)v(P_n), solve/gather self E*,B* once, restore R_n,P_n, then replay each
particle over h with boundary-resolved ordinary Boris substeps. Each substep
combines the frozen self fields with live external fields at its own midpoint
and physical time. All particles synchronize again at t_n+h.

This freezes **gathered, particle-labelled vectors**, not a mesh regathered along
the path. The Boris kick must use the current test-particle velocity in v×B*.
Do not rebuild B* from that velocity or apply another source Lorentz conversion.
Keep stored vectors in fixed tracker axes, or transform them consistently if the
reference axes change.

With bounded acceleration, the ballistic position predictor has O(h²) error at
the common physical midpoint, including a finite hard-edge crossing. A smooth
self-field functional then has O(h²) evaluation error. Freezing an accurately
centred material field over h cancels its leading odd-in-time variation, giving
O(h³) local impulse error. Combined-field substeps also retain external/self
force ordering through an edge, avoiding the missing commutator of one averaged
external kick. Under stable, regular conditions one expects separate O(h²)
self-refresh and O(δ²) smooth Boris errors, plus event and PIC errors, where δ is
the largest local substep. This is not a proof for noisy or nonsmooth PIC maps.

The deposit is a **synchronized predicted midpoint**. It generally differs by
O(h²) from the state actually reached halfway through the replay. It is a valid
explicit predictor stage, but is not an exact intermediate state of the accepted
trajectory. A requirement to deposit exactly on that trajectory would require a
corrector. With no internal substeps and a position-only self force, the proposal
reduces to the ordinary single-step Boris/PIC update.

## Actual OPALX qualification: source momentum is also required

Our BINS=ONE solve depends on momentum as well as position. In
`src/SpaceCharge/CartesianPIC3D/ParticleBinTraversal.cpp`, the bin mean P and mean
gamma come from the live particle array. `RelativisticFieldComposer.cpp` forms

    E = gamma E' − (gamma−1)(E'·vhat) vhat,
    B = gamma (v_source × E') / c²,
    v_source = c <P> / <gamma>.

The mesh is stretched by gamma, and the solve's spatial alignment also uses mean
momentum. After the ordinary first drift those momenta remain P_n. When source
energy/direction changes, they are generically only O(h)-accurate midpoint data.
An O(h) field error produces an O(h²) local impulse error and potentially a
first-order global term. Special symmetries or low beta can suppress the
coefficient; midpoint positions alone cannot establish second order.

A general second-order claim needs O(h²)-accurate midpoint source moments/frame
quantities, or a stated fixed-velocity approximation and convergence evidence.
A predictor might avoid a second PIC solve, but its initialization and response
to hard-edge partial impulses must be specified. A naive momentum Euler
predictor can miss an O(h) partial impulse and its launch derivative. No such
predictor is introduced or validated here.

## Boundaries, reversibility and conservation

For a spatial face normal n, the exact variational jump is

    S = I + (F_after−F_before)(n,0)^T / [(n,0)^T F_before].

A finite magnetic edge changes acceleration, while position and velocity remain
continuous. Its jump acts through normal-position perturbations. Boundary
location must follow the **combined-field trajectory**, including the frozen
self force. A support schedule based only on external forces can miss a shifted
crossing. Validate the first variations; orbit convergence alone does not
establish tune convergence. Grazing faces, losses/emission, discontinuous self
boundaries, mesh/interpolation kinks and adaptive bins need additional analysis.

The macro method is generally not exactly time reversible: reconstructing a
ballistic midpoint from the final replay state need not recover the original
predictor. At a bounded force edge, their O(h²) position difference yields an
O(h³) self-impulse discrepancy for a regular position-only field. This permits
second-order accuracy but gives no exact symmetry or symplecticity. Source
momentum lag can cause the larger error above; local adaptive step selection has
its own reversal error. Each magnetic-only Boris substep preserves |P|, but that
does not establish total self-field energy conservation or prevent numerical
emittance growth once electric self fields act.

## Efficiency and required evidence

One collective PIC solve per macro interval is compatible with independent
on-device edge loops and no collective at intermediate particle times. That is
a credible cost advantage, not a measured speedup. Gathered fields and original
states must follow particle identity through any PIC migration; cached views
must be reacquired. Prediction must not commit loss/wrap/monitor/emission side
effects. Charge deposition retains the common dt normalization; individual
Boris intervals sum to h and do not rescale the stored fields.

The bounded standalone test with live E=A R, constant nonparallel self B and a
hard body B is appropriate. Its exact oracle uses

    Dv = (I−vv^T)/gamma,
    DF = [[0,Dv],[A,−[B]_cross Dv]],
    H = gamma − R^T A R/2  (symmetric A).

These formulas and the saltation matrix were independently checked. Positive A
is defocusing, so keep the test bounded. Apply each event jump once, exclude
grazing/end-on-face ambiguity, and recompute the candidate's predictor/E* for
each perturbed launch in finite differences. This test intentionally omits
changing source boosts and full self-consistent PIC.

Before production, separate macro-step from local/event errors; check all three
DBA bare modes, source-frame evolution, finite-amplitude tune footprints,
energy/moment drift and h/N/mesh convergence. The existing SCSOLVEFREQ input is
not evidence of an implemented reuse mode. Any proposed opt-in behavior needs a
concrete reviewed specification.

An external-half / self-kick / external-half composition instead applies a
separate self map at a transported midpoint; symmetry depends on its constituent
maps. The present proposal keeps external and self forces together in every
Boris substep. It does not require that operator split and inherits none of its
conservation claims. The outcome here is a conditional basis for a proposal,
not approval of an implementation or timestep.
