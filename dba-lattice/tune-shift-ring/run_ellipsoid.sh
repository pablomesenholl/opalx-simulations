#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${OPALX_PYTHON:-$HOME/.venv-h6/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PYTHON" -m unittest discover -s "$ROOT/analysis" -p 'test_*.py'
exec "$PYTHON" "$ROOT/analysis/run_ellipsoid.py"
