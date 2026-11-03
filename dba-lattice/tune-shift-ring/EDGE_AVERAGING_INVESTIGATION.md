# Field residence averaging: investigation, not an available method

The large-N exact boundary scheduler is too costly. Ordinary fixed-step ring
tests also fail: a400ps step gives a horizontal tune error about0.021 for tiny
bare probes, with eventual centimetre-scale orbit/reference separation; the
200ps test fails the ordered-section check. These runs do not validate a
production timestep. A finite BOUNDARYDT alone would inherit this coarse-edge
problem unless sufficiently small; that proposed input is not implemented.

## Candidate numerical correction

Keep the existing common drift–kick–drift, actual midpoint PIC solve, and
combined Boris kick. Replace the point-sampled external field near a hard
edge by an average over the fraction of this step spent inside the element.
This is a numerical-method change and requires independent tests. It is not
the already validated exact-boundary method.

For one element, a predicted path r(tau), tau in[0,h], gives

    Fbar_ext = (1/h) integral_0^h indicator_support(r(tau)) F_ext(r(tau),t+tau) d tau.

Here F denotes E or B, h is seconds, positions are metres, E is V/m and B
tesla. The instantaneous self field remains the existing solve at the common
midpoint; it is neither advanced through a separate particle subcycle nor
recomputed on particle-specific clocks. Each particle has its own external
support fraction, but all particles share the same accepted physical interval.

The two ordinary Boris half-drift segments provide a cheap piecewise-linear
predicted path. Its final momentum depends on the combined kick, so a bounded
corrector or an independently justified predictor is needed. Element crossings
on this predicted path split the quadrature intervals. Existing element field
kernels should evaluate the interior sample points; do not invent a second
production magnet/RF field model. The support descriptor supplies geometry only.
Overlapping external fields still add before the ordinary kick.

An implementation could temporarily stage device positions at the interior
quadrature points, use the existing additive element kernels with isolated
field buffers, weight the resulting fields and restore the physical midpoint
positions. PIC is called only at the actual physical midpoint. Passive monitor
side effects must be excluded from trial field queries. This must be measured
against a simpler implementation; extra copies or host particle transport are
not acceptable by default. The host reference must use the same approximation.

## Expected properties to establish, not assume

- A magnetic-only Boris rotation preserves momentum norm algebraically even
  with a state-dependent averaged B. That alone does not prove an accurate map.
- Smooth regions should reduce to the existing second-order midpoint update.
- At an isolated non-grazing face, residence weighting should remove the O(h)
  missed impulse. A finite number of O(h^2) crossing errors could retain global
  second-order accuracy. Verify position, momentum AND first variations.
- A field kick at the common midpoint does not reproduce the exact position
  through an arbitrarily placed edge, even when its integrated impulse is exact.
- Grazing faces, transverse aperture loss, multiple/disconnected supports,
  corners, curved charts, and rapidly varying RF require explicit treatment.
  The current shortest-body guard is not a general proof against all such cases.
- A corrector must not change the PIC deposit position, introduce species/rank
  dependent clocks, or silently reuse a field across physical accepted steps.
- Neither symplecticity nor exact energy conservation with RF/space charge is
  claimed. Characterize numerical heating, reversibility, and tune convergence.

## Required evidence before production use

1. Standalone/unit hard-edge uniform-field tests against an exact solution,
   including crossings at different fractions, signs and near-face perturbations.
2. Smooth-field reduction and additive self-field tests; consistent host/device
   results, nonfinite-state handling, and no mutation from rejected predictors.
3. DBA N1/reference closure and all three bare normal-mode tunes versus DT,
   including tiny and millimetre probes. Keep failed fixed-step evidence.
4. Large-N short-cell PIC, all particles retained, moments/density versus mesh,
   particle count and timestep. Dedicated MPI tests separate from one-rank sweeps.
5. Only then a many-turn charge scan and individual tune footprints.

No new public input API is assumed by this investigation. Any eventual public
API addition needs explicit approval. Current production source is unchanged
by this document and the sandbox prototype work.

## Related established alternative

Qiang's [2017 multiparticle splitting model](https://doi.org/10.1103/PhysRevAccelBeams.20.014203)
separates external transport and the space-charge map. Its symplectic claim
requires symplectic constituent maps; it does not establish symplecticity of
our ordinary Boris/PIC implementation or the averaging candidate above.
The [2018 PIC comparison](https://doi.org/10.1103/PhysRevAccelBeams.21.054201)
also illustrates why long-term emittance and timestep convergence must be
checked independently. A new operator splitting would be a separate design
decision, not a description of the existing combined-field kick.
