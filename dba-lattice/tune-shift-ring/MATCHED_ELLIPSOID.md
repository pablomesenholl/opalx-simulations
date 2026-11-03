# Periodic isolated-bunch model for the DBA ring

A stable, periodic **six-dimensional moment-closure** family is available at
0, 10, 20, 40, 80, 120 and 160 pC. It includes dispersion, horizontal–longitudinal
coupling, longitudinal space charge and RF confinement. At 160 pC the predicted
incoherent tune shifts are **−0.154480565 horizontally and −0.203879307 vertically**,
relative to the same RF lattice at zero charge.

These are analytic-field/moment predictions. A finite-emittance sample with uniform
spatial density is not thereby a stationary Vlasov equilibrium. Density preservation,
PIC convergence and the actual tune footprint still require particle tracking.

## Selected physical configuration

The magnetic lattice follows the [LaTeX note](../TeX/main.tex),
`sec:hamiltonian`. The middle 0.1 m of each 1 m D2 straight is replaced by an ideal buncher,
leaving 0.45 m drift on either side. There are six independently phased cavities.
The reference particle has zero integrated energy gain in each cavity.

- Linear integrated RF slope: `delta_out − delta_in = g z`, `g=+0.001 /m`.
- Coordinate convention: `z` points toward the bunch head, `delta=(p−p0)/p0`.
- Harmonic 4: analytic frequency 3.3906387456 MHz.
- Effective gap voltage: 5119.245591 V per cavity.
- For a 0.1 m prescribed sinusoidal longitudinal field, the straight-reference
  transit factor is 0.9996036213. Initial peak amplitude: 0.05121275555 MV/m.
- Actual OPALX RF frequency and phase must use its measured closed-orbit return
  time and finite-gap trajectory. The analytic values are calibration starting
  values, not a substitute for the zero-reference-gain check.
- Rms normal-mode emittances, held fixed across charge: 3, 3, 33.54373179 µm.
- The zero-charge rms length is 50 mm at the original B1 section, corresponding
  to a full uniform-ellipsoid extent of 223.6068 mm. Each charge is matched
  separately; the finite-charge length is an output.

The observation and launch section is the **middle of D1**, so the complete bunch
fits inside a straight. This avoids treating a Cartesian ellipsoid at a curved
bend entrance as a correctly centred Frenet distribution. It changes neither the
ring nor the matched family: the original covariance is transported through B1
and half of D1 and the one-cell map is conjugated by that transfer map.

Use the matrices in `results/ellipsoid-match-stable-long-d1center/`. Files in
`results/ellipsoid-match-stable-long/` retain the original B1 section for provenance.
The cyclic cell at the launch section is

```text
D1b(0.5) – Q(0.2) – D2a(0.45) – RF(0.1) – D2b(0.45)
        – B2(π/3) – B1(π/3) – D1a(0.5)
```

Lengths are metres. The finite analytic RF lens uses `delta' = (g/L_RF) z` inside
RF; the existing prescribed-Ez particle model and this closure both omit RF
transverse gap forces. The closure additionally neglects the small reference-energy
excursion inside a zero-net-gain cavity. The maximum bunch-edge RF phase over
this family is 0.109162 rad, giving at most 0.1985% sine-versus-linear deviation.
The reference transit phase also varies over the finite gap, with the transit
factor above. These approximations should be assessed against the calibrated RF
bare tracking before assigning all observed differences to space charge.

## Predicted frequencies

| Charge (pC) | Qx | Qy | Qs | ΔQx | ΔQy |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.764038386 | 1.689536397 | 0.069128128 | 0 | 0 |
| 10 | 0.746726193 | 1.671009278 | 0.084396157 | −0.017312193 | −0.018527119 |
| 20 | 0.730908537 | 1.652823586 | 0.100638854 | −0.033129849 | −0.036712811 |
| 40 | 0.703407495 | 1.618698429 | 0.133271737 | −0.060630891 | −0.070837968 |
| 80 | 0.661448358 | 1.562207112 | 0.189130669 | −0.102590027 | −0.127329285 |
| 120 | 0.631578732 | 1.519290153 | 0.229974739 | −0.132459654 | −0.170246244 |
| 160 | 0.609557821 | 1.485657090 | 0.259718680 | −0.154480565 | −0.203879307 |

The RF itself changes the zero-charge horizontal tune from 0.770107569 without RF
to 0.764038386. Charge differences must use the RF zero-charge baseline.

At 160 pC the periodic rms ranges through a cell are approximately
11.264–16.654 mm horizontally, 2.487–3.743 mm vertically, and 40.558–41.035 mm
longitudinally. The horizontal size includes dispersion; it is not a pure betatron
size. At the original B1 section the x–delta correlation increases from 0.805 at
zero charge to 0.960 at 160 pC. A raw x FFT may therefore select a synchrotron
component instead of the horizontal betatron mode.

## Matching and observable definitions

For `X=(x,px/p0,y,py/p0,z,delta)`, evolve the covariance `C=<XX^T>` as

```text
C' = F(C,s) C + C F(C,s)^T
M' = F(C,s) M
```

The curvature terms, external focusing and isolated-ellipsoid field tensor are
those in `analysis/ellipsoid_model.py`. The moment solver retains all x–z tilt
terms and uses a positive-definite block-Cholesky parametrization consistent with
midplane symmetry. Periodic shooting constrains the three symplectic
rms eigenemittances. Covariance periodicity alone leaves those emittances free.

The particle map `M` uses the frozen self-field along the matched periodic moment
trajectory. Its normal modes predict incoherent particle frequencies within this
linear closure. **Collective** stability is checked separately by differentiating
the nonlinear one-cell covariance map with respect to all 21 symmetric moments,
including cross-plane perturbations absent from the symmetric match.

Each `match-<charge>pc.npz` contains:

- `covariance`: matched 6×6 covariance at D1 centre.
- `matrix`: particle one-cell map at that section.
- `normal_mode_basis`, `normal_mode_inverse`: a real symplectic basis `W` and its
  inverse, in x/y/z canonical mode order.
- `signed_cell_phases`: positive x/y phases and negative longitudinal phase above
  transition. Frequencies in the table use absolute phase times `6/(2π)`.
- `s`, `covariances`, `particle_maps`: the one-cell moment trajectory and transfer
  maps at common sample points.
- `transfer_from_b1`: map used to shift the observation section.

For particle phase extraction at successive equivalent cell sections, subtract
the beam centre and transform `U=W^-1 X`. The complex signal
`U[2*j] − i U[2*j+1]` advances by `signed_cell_phases[j]` in this model. Sample at
cell crossings to resolve the integer part of Qy; turn-only sampling aliases it.
These basis matrices describe the analytic closure. Particle tune footprints
should also be checked with an independent frequency estimator.

## Numerical verification and rejected design

Eleven Python unit tests pass, including symplecticity, exact zero-charge map
agreement, nonzero-charge eigenemittance conservation, positive periodic matching,
normal-mode rotation, full-moment stability and section-map conjugation.

For all seven selected charges:

- Tightening the ODE tolerance by 100 changes any predicted tune by less than
  8.2e−12.
- Full 21-moment collective multipliers have modulus within 1.4e−9 of one;
  halving the finite-difference step confirms this result.
- Independent 32-turn moment integration changes the periodic covariance by less
  than 5.0e−9 in scaled components; relative eigenemittance drift is below 3.1e−9.
- After shifting to D1 centre, one-cell closure is below 4.1e−10; conjugated and
  independently propagated cell maps differ by less than 1.6e−11 in mode coordinates.

The original 100 mm full-length, weak-RF family was rejected for a concrete
physical reason: although positive periodic covariances and stable particle maps
exist, the full moment closure is collectively unstable above 20–30 pC. Its
41.131 pC maximum multiplier is 1.106851 per cell, an e-folding time of about
1.64 turns. That is not a numerical tolerance failure. The longer selected bunch
avoids this instability while reaching a vertical tune shift near −0.2.

## Kinetic consistency fails for the current particle sample

The moment match must **not** be described as a matched six-dimensional particle
equilibrium. A separate calculation transported 131,072 particles through the
exact prescribed periodic linear cell maps, without OPALX or a PIC mesh. The
initial spatial sample was a uniform ellipsoid with conditional Gaussian momenta.
After only one cell, at 160 pC:

- Radial CDF error rises from 0.000030 to 0.1322.
- 13.22% of particles lie outside the rms-defined ellipsoid boundary.
- The density averaged inside radius 0.25 of that ellipsoid is 1.73 times the
  uniform-density expectation.
- The maximum scaled covariance error remains only 0.00015.

The same failure occurs at zero charge: after one cell 14.54% lie outside the
ellipsoid. It is intrinsic to the chosen finite-emittance distribution, not a
PIC convergence error. Repeating the 160 pC one-cell calculation with 65,536,
131,072 and 262,144 particles and two independent Sobol scrambles gives an outside
fraction of 13.22–13.35% and core-density ratio 1.71–1.73.

An independent Coulomb-pair virial calculation estimates the whole-bunch
least-squares self-force gradient. At 160 pC after one cell, its changes relative
to the uniform-ellipsoid prediction are +1.42±0.18% horizontally,
−5.40±0.58% vertically and +0.76±0.10% longitudinally. Errors are one batch
standard error from 1,048,576 point pairs. These are **whole-bunch** fits,
not central gradients or particle tune shifts. The anisotropic core force cannot
be inferred merely by multiplying the uniform force by the central-density ratio.

For the rest-frame pair separation d, the mesh-independent identity used is

```text
<E'_i r'_j> = Q/(8 pi epsilon_0) <d_i d_j / |d|^3>
G'_fit = <E' r'^T> <r' r'^T>^-1
```

The code then applies the same Lorentz and normalized-momentum factors as the
ellipsoid force model. Uniform-sphere and charge-scaling tests check the estimator.

The current uniform closure therefore remains a **conditional reference** for
PIC comparison. The actual density evolution and individual tune footprint must
be measured. Agreement of rms sizes alone is insufficient evidence of equilibrium.
The data, sample-size checks and figures are in
`results/ellipsoid-kinetic-consistency/`; the figure is `kinetic-consistency.png`.

## Tracking observable tools

`analysis/ellipsoid_observables.py` provides density/CDF tests, moment diagnostics,
Coulomb-pair whole-bunch force fits and individual normal-mode tune extraction.
The tune routine uses a supplied section basis and optional explicit beam centroid.
It retains both an amplitude-weighted phase fit and an independent Hann-spectrum
peak search over the complete signed Nyquist interval. It does not constrain
frequencies toward the prediction. Inactive numerical mode leakage is flagged;
phase residual, spectral coherence and disagreement of estimators are retained.

For the runner's `section-observations.npz` format (`phase`, `turn_coordinate`,
`ids`), use:

```sh
python analysis/analyze_ellipsoid_modes.py section-observations.npz \
  results/ellipsoid-match-stable-long-d1center/match-160pc.npz mode-analysis
```

For the paired probe deck only, add `--central-id 0` to subtract its explicit
central reference particle. Do not identify an arbitrary macroparticle as a
centroid in a physical many-particle bunch. Eight additional tests cover the
observable functions, including x/z separation in a dispersive basis, signed
longitudinal phase, the integer part of Qy from cell sampling, independent
spectral fitting and the runner's NPZ/CSV interface.

## Reproduce

Use `~/.venv-h6/bin/python` from the experiment directory:

```sh
python analysis/match_ellipsoid_ring.py --output results/new-match \
  --sigma-z .05 --rf-slope .001 --harmonic 4 \
  --charges-pc 0 10 20 40 80 120 160
python analysis/validate_matched_ellipsoid.py results/new-match
python analysis/shift_matched_ellipsoid_section.py \
  results/new-match results/new-match-d1center
python -m unittest discover -s analysis -p test_match_ellipsoid_ring.py -v
```

`results/ellipsoid-match-stable-long/validation.json` retains the full numerical
checks. Source snapshots are included with both section datasets. RF and bunch
length scans, including unstable cases, are retained for comparison.

## Physical scope

This extends the [uniform-ellipsoid internal-field calculation](ELLIPSOID.md).
Rms envelope closure is an established approximation for bunched beams; it is not
an exact density-preserving particle distribution:
[Sacherer, CERN-SI-Int-DL-70-12](https://cds.cern.ch/record/322516/).
Exact positive finite-emittance uniform three-dimensional solutions do exist for
special isotropic focusing, for example the
[Kurth solution discussed by Mitchell, Hwang and Ryne](https://accelconf.web.cern.ch/ipac2021/papers/wepab248.pdf).
The unequal DBA normal modes and negative longitudinal Hamiltonian signature do
not meet those isotropic assumptions. Thus a sample with uniform spatial density
and conditional Gaussian momenta realizes the prescribed covariance, but its
uniform density still needs verification during PIC tracking.
