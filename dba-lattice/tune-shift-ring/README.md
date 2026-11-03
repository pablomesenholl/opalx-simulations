# Coasting KV benchmark: runners and experiment records

The [LaTeX note](../TeX/main.tex) is the authoritative description of the
baseline lattice (`sec:hamiltonian`), KV model (`sec:kv`), charge normalization,
operating points and FFT2D5 comparison (`sec:fft2d5`). This README records the
runner interface, file formats and historical experiment checks.

The isolated-bunch extension and trial buncher results are documented in
[ELLIPSOID.md](ELLIPSOID.md). Run that calculation with `./run_ellipsoid.sh`.

## Run the reference calculation

From this directory:

```sh
~/.venv-h6/bin/python analysis/run_model.py
```

For the targeted reference tests, use:

```sh
~/.venv-h6/bin/python -m unittest discover -s analysis -p 'test_kv*.py' -v
```

The historical `./run.sh` wrapper uses `~/.venv-h6` (override with `OPALX_PYTHON`)
and discovers all experiment tests; some diagnostic tests require archived
OPALX outputs. NumPy, SciPy, pandas and Matplotlib are required. Each calculation
creates a fresh directory in `results/`. The analytic model uses no stochastic
sampling.

## Configuration and outputs

`config.json` supplies the kinetic energy, **geometric rms** emittances, cell
count, hard-edge lengths/focusing and requested vertical shifts.

Each run contains:

- `config.json`, `source/`, `run.log`: input, source snapshot and provenance.
- `current_scan.csv`: current, line charge, total ring charge, populations,
  tunes, sizes, matched initial envelopes, periodic residuals and envelope stability.
- `targets.csv`: the requested vertical-shift operating points.
- `envelope-*.csv`: rms sizes and accumulated phases over one cell; repeat six
  times for the full ring, adding the phase advance at each cell.
- `summary.json`: global parameters and numerical convergence checks.
- `fft2d5_reference.json`: local field-gradient and force-normalization targets.
- `tune_shift.png/pdf`, `envelopes.png/pdf`: figures.

## Implementation-specific checks and initial scan

The model includes no cooling, aperture or loss model and does not reproduce IOTA.
For the configured proton beam, the rigidity is 0.2286218828 T m and the dipole
field magnitude is 0.1143109414 T.

The constant-focusing round limit provides an exact independent test, with
edge emittance E and perveance K defined in the LaTeX note:

```text
a^2 = [K + sqrt(K^2 + 4 k E^2)]/(2 k)
depressed wavenumber = E/a^2
```

For particle initialization, the matched covariance has <x^2>=a^2/4,
<xx'>=aa'/4, and <x'^2>=(a'^2+E_x^2/a^2)/4, with analogous vertical expressions.

The initial seven model tests cover proton units, exact maps and Twiss functions,
the zero-current ring, exact constant focusing, weak-current and similarity
scaling, tighter integration, independent envelope variational derivatives, and
direct particle-matrix phase agreement. All tests avoid OPALX execution.
Integration uses DOP853 with exact element boundaries. Target tune changes under
100-fold tighter relative integration tolerance are below 6e-12; scaled periodic
closure residuals are below 5e-13 in the initial scan. The solver acceptance
thresholds are 2e-8 for scaled closure and 1e-7 for the run's tune convergence
check; these are conservative rejection thresholds, not claimed physical accuracy.

The envelope-stability check addresses linear envelope modes only; it does not
establish freedom from kinetic modes, imperfection-driven resonances or losses.
The principal finite-current tunes lie near half-integers, so magnet errors may
matter in later tracking studies.

Additional theory reference used for the original experiment:
[Franchetti, CERN CAS lecture, slides 46–70](https://cas.web.cern.ch/sites/default/files/lectures/prague-2014/franchettisc.pdf).
The configured local physics-manual repository was unavailable at setup time.

## Proton lattice and sampled distribution records

`bare-cof.in` implements the proton lattice. The reference momentum is
0.068539116183 GeV/c; normalized momentum p/(m_p c)=0.073048230844.

The single-rank DOP853 closed-orbit calculation converges at (0,0,0,0), with
horizontal residual 8.44e-14 m. Its full tunes, after assigning the integer and
complement branches from the analytic cell phases, are
**0.770107568476 / 1.689536396825**. Differences from analytic optics are
−6.52e-10 / −1.95e-10. This verifies a one-turn linearization about the closed
orbit; it does not resolve the earlier PARALLEL multi-turn tracker issues.

Reproduce that check with `./run_bare.sh`. The binary can be selected with
`OPALX_BINARY`. Archived output and binary provenance are in
`results/bare-proton-lzzekti5/`. `analysis/bare-reference.json` is a copy of that
COF JSON used as a unit-test fixture for the comparison/acceptance code.

Generate the matched distribution with:

```sh
~/.venv-h6/bin/python analysis/prepare_kv.py
```

This generates **65,536 macroparticles**, seed **546544**, targeting Delta Qy=-0.2.
Signed and cyclic symmetry within blocks of 64 gives exactly centered samples
and exact transverse second moments up to roundoff, while preserving the KV
shell. Each transverse projection has uniform elliptical density. The sampled
geometric rms emittances are 3.000000 micrometres in both planes.

- `kv-section.txt` is a FROMFILE-formatted observation-plane section, with
  x,px,y,py,z,pz in metres and p/(m_p c). All z=0. Its physical interpretation
  and restrictions are specified in the LaTeX note, `sec:fft2d5`.
- `matched-kv.json` records matching, moments, charge normalization and checksums.
- Each fresh `results/matched-kv-*/` contains both these files, source snapshots,
  phase-space plots and `kv-ring.csv`. The latter samples 1024 equally spaced
  longitudinal slice centers with 64 matched transverse particles per slice.
  Its s,x,x′,y,y′ are **local ring coordinates**, not global Cartesian coordinates.
  Each particle has charge 1.872e-13 C approximately.

### Historical solver compatibility note

The original OPALX `FieldSolverCmd::constructBCHandler` rejected mixed boundary
conditions, including open/open/periodic. That experiment established the matched
coasting initial conditions while a compatible solver/coordinate treatment
remained unresolved. This records the original implementation, not a new check
of the current solver.
