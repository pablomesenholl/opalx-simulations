#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$case_dir/../../.." && pwd)
opalx_binary=${OPALX_BINARY:-"$repo_dir/omp-build/src/opalx"}
python_binary=${OPALX_PYTHON:-"/Users/adelmann/.venv-h6/bin/python"}
run_name=${RUN_NAME:-run-$(date -u +%Y%m%dT%H%M%SZ)}
run_root="$case_dir/results/$run_name"

test ! -e "$run_root"
mkdir -p "$run_root/analytic"
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
export OMP_DYNAMIC=false

"$python_binary" "$case_dir/analysis/generate_analytic.py" \
    --output "$run_root/analytic" > "$run_root/analytic/run.log" 2>&1
"$python_binary" "$case_dir/analysis/run_cases.py" --executable "$opalx_binary" \
    --analytic "$run_root/analytic" --output "$run_root/opalx" > "$run_root/opalx.log" 2>&1

git -C "$repo_dir" rev-parse HEAD > "$run_root/git-revision.txt"
shasum -a 256 "$opalx_binary" > "$run_root/binary.sha256"
echo "PASS: $run_root"
