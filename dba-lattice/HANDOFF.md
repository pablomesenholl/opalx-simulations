# DBA lattice task state — 2026-11-01

The [LaTeX note](TeX/main.tex) is authoritative for the physics model, benchmark
results, figures, limitations and design task. [README.md](README.md) provides
commands and file navigation.

## Current artifacts

- `TeX/main.tex` and `TeX/main.pdf`: consolidated scientific note, with both figures.
- `model/`: cell specification, machine generator, expanded machine and local optimizer.
- `designs/demo-qx0.8-qy1.7/`: worked design and its compatible KV configuration.
- `REFERENCE_RUNS.json`: baseline and optimized scan locations.
- `tune-shift-ring/analysis/run_model.py`: copied runner extended with explicit
  charge columns, FFT2D5 reference exports and standalone source-hash provenance.
- `TeX/make_figures.py`: figure reproduction; hashes in `TeX/figures/provenance.json`.

## Completed verification

These checks were completed before the present Markdown cleanup:

- Seven machine/design/artifact tests and ten existing KV tests passed in `~/.venv-h6`.
- Baseline and optimized scans completed. Tighter integration changed tunes by
  at most 5.32e-12 and 5.50e-12, respectively; no tolerances were relaxed.
- Baseline geometry closure was 4.39e-15 m and the symplectic residual was
  7.55e-15. All 30 placements matched the copied proton deck within 2e-14.
- Exported laboratory fields independently reproduced the KV defocusing terms
  after Lorentz-force conversion at every target in both scans.
- The worked design's rematched principal current is 10.019502 mA and its
  circumference is 26.23649521009 m; the full scan is indexed by `REFERENCE_RUNS.json`.
- The 33 plot charges matched `I*C/(beta*c)` within 7.11e-15 nC. The plot-only
  unity tolerance is 1e-8, separating <1.5e-12 integration noise from >5e-3
  observed multiplier growth; no solver acceptance threshold changed.
- The final nine-page document compiled without warnings, unresolved references
  or overfull/underfull boxes. Both figures and all pages passed visual review.

## Provenance and review records

The initial packages were copied on 2026-10-30 from
`/Users/adelmann/git/old-opal/opalx`; copied files were SHA-256 verified.
The three explanatory Markdown notes were consolidated on 2026-11-01, then
rewritten with the manuscript-writing-review skill and its local extension.
Independent editorial review found no substantive omission or overstatement.

- `tmp/scientific-rewrite-20261101/`: original TeX, writing review and retention checks.
- `tmp/figures-20261101/`: pre-figure TeX, renderings and compilation/check metadata.
- `tmp/md-dedup-20261101/`: Markdown cleanup inventory, source backups and review record.

Historical diagnostic reports retain their source revisions and archived paths.
No production OPALX source was changed or pushed. Resume solver integration from
the procedure in Section 4.3 of the LaTeX note; historical diagnostic runs do not
replace validation of the current solver.

## Markdown cleanup completed — 2026-11-01

Reviewed all 30 active Markdown files. Removed duplicated exposition from ten
files and replaced it with links to the LaTeX note. Retained commands, file
formats, acceptance criteria, distinct finite-bunch/RF models and historical
results. Archived source snapshots and diagnostic reports were preserved.

Reviewed the documentation diffs and checked new links, code-fence balance and
checksum entries for edited package Markdown. The LaTeX source and model code
were unchanged; no physics or MPI runs were needed. Four existing Markdown
checksum entries were refreshed. Older manifests in `tune-shift-ring` and
`dba-ring-cof-1` already referenced missing archives and stale files; those
unrelated entries remain historical and are recorded in the cleanup audit.
The portable `dba-cof-1` package manifest verifies completely.

Inventory, non-Markdown checksum notes and per-file checks are recorded in
`tmp/md-dedup-20261101/checks.json`; source backups use the `.md.before` suffix
under that directory's `before/` tree.
