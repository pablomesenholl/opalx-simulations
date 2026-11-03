#!/bin/sh
# Run the portable COF benchmark on one MPI rank and one OpenMP thread.
set -eu

if [ "$#" -ne 0 ]; then
    echo 'Usage: OPALX_BINARY=/path/to/opalx [OPALX_PYTHON=python3] [RUN_NAME=name] ./run.sh'
    exit 0
fi

case_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_binary=${OPALX_PYTHON:-python3}
opalx_binary=${OPALX_BINARY:-opalx}

if ! python_binary=$(command -v "$python_binary"); then
    echo 'Python not found. Set OPALX_PYTHON to a Python interpreter with NumPy and pandas.' >&2
    exit 1
fi
if ! opalx_binary=$(command -v "$opalx_binary"); then
    echo 'OPALX not found. Set OPALX_BINARY to the executable or put opalx on PATH.' >&2
    exit 1
fi
if [ ! -x "$opalx_binary" ]; then
    echo "OPALX is not executable: $opalx_binary" >&2
    exit 1
fi
"$python_binary" -c 'import numpy, pandas'

run_name=${RUN_NAME:-run-$(date -u +%Y%m%dT%H%M%SZ)}
case "$run_name" in
    .|..|*[!A-Za-z0-9_.-]*)
        echo 'RUN_NAME must be a single name containing letters, digits, dots, underscores or hyphens.' >&2
        exit 1
        ;;
esac
run_root="$case_dir/results/$run_name"
mkdir -p "$case_dir/results"
if ! mkdir "$run_root"; then
    echo "Choose a new RUN_NAME; existing results are preserved: $run_root" >&2
    exit 1
fi
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
export OMP_DYNAMIC=false
export PYTHONDONTWRITEBYTECODE=1

if ! "$python_binary" "$case_dir/analysis/generate_analytic.py" \
    --output "$run_root/analytic" > "$run_root/analytic.log" 2>&1; then
    echo "Analytic calculation failed; inspect $run_root/analytic.log" >&2
    exit 1
fi
if ! "$python_binary" "$case_dir/analysis/run_cases.py" --executable "$opalx_binary" \
    --analytic "$run_root/analytic" --output "$run_root/opalx" > "$run_root/opalx.log" 2>&1; then
    echo "Benchmark failed; inspect $run_root/opalx.log and the per-case run.log files." >&2
    exit 1
fi
cat "$run_root/opalx.log"
echo "PASS: $run_root"
