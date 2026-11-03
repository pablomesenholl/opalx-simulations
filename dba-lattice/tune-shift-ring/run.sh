#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${OPALX_PYTHON:-$HOME/.venv-h6/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PYTHON" -m unittest discover -s "$ROOT/analysis" -p 'test_*.py' -v
exec "$PYTHON" "$ROOT/analysis/run_model.py" "$@"
