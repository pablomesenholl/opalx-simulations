#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${OPALX_PYTHON:-$HOME/.venv-h6/bin/python}
BINARY=${OPALX_BINARY:-$ROOT/../../../omp-build/src/opalx}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$ROOT/results"
OUT=$(mktemp -d "$ROOT/results/bare-proton-XXXXXXXX")
cp "$ROOT/bare-cof.in" "$ROOT/config.json" "$OUT/"
"$PYTHON" - "$BINARY" "$OUT" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path
binary, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
meta = dict(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=out, text=True).strip(),
            mpi_ranks=1, omp_threads=1, command=[str(binary), 'bare-cof.in'])
(out/'provenance.json').write_text(json.dumps(meta, indent=2)+'\n')
with (out/'run.log').open('w') as log:
    subprocess.run(meta['command'], cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
PY
"$PYTHON" "$ROOT/analysis/check_bare.py" "$OUT"
printf 'Results: %s\n' "$OUT"
