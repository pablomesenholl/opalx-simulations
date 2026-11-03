# DBA lattice: KV solver reference and AI machine design

The [LaTeX document](TeX/main.tex) is the authoritative description of the model, benchmark results, figures, validation scope, and AI design task. Open `TeX/` as a LaTeX project; `main.tex` is the main document. This README covers navigation and reproduction.

## Entry points

| File | Purpose |
|---|---|
| `model/cell.json` | Machine-readable baseline Hamiltonian coefficients and conventions |
| `model/machine.json` | Expanded 30-element machine, placements, maps, dispersion and tunes |
| `model/machine.py` | Generate a machine from a cell |
| `model/optimize.py` | Local two-knob reference tune optimizer |
| `designs/demo-qx0.8-qy1.7/` | Worked optimized machine and compatible KV configuration |
| `REFERENCE_RUNS.json` | Paths to reproduced baseline and optimized KV scans |
| `tune-shift-ring/config.json` | Baseline proton KV beam and cell parameters |
| `tune-shift-ring/analysis/run_model.py` | Immutable scan, figures, stability and FFT2D5 reference export |
| `tests/test_machine.py` | Hamiltonian, symplecticity, geometry, tune and design checks |

## Reproduce

Run from this directory with the local OPALX Python environment:

```sh
~/.venv-h6/bin/python -m unittest discover -s tests -v
~/.venv-h6/bin/python -m unittest discover -s tune-shift-ring/analysis -p 'test_kv*.py' -v
~/.venv-h6/bin/python tune-shift-ring/analysis/run_model.py
~/.venv-h6/bin/python model/machine.py --output /tmp/dba-machine-new.json
~/.venv-h6/bin/python model/optimize.py --qx .8 --qy 1.7 --output-dir /tmp/dba-design-new
~/.venv-h6/bin/python tune-shift-ring/analysis/run_model.py --config /tmp/dba-design-new/kv-config.json
~/.venv-h6/bin/python TeX/make_figures.py
```

The machine generator and design exporter refuse to overwrite outputs. Each analytic scan creates a new results directory and records source hashes, versions and input. Git provenance is optional, so the standalone copy runs without initializing a repository. Required Python packages: NumPy, SciPy, pandas and Matplotlib. The targeted tests above exercise the current reference; historical diagnostic tests may require archived OPALX outputs.

## Historical inputs

Copied 2026-10-30 from `/Users/adelmann/git/old-opal/opalx`, related to [PR #553](https://github.com/OPALX-project/OPALX/pull/553): `dba-cof-1/` is the self-contained electron COF package; `dba-ring-cof-1/` is the original PR fixture; `tune-shift-ring/` contains the proton KV study and subsequent tracking diagnostics. Historical absolute paths and source snapshots retain their provenance.

