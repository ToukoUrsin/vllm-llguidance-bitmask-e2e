#!/bin/bash
# Session 032 driver, inside vllm/vllm-openai:cu129-nightly-5555a2d8f7116b2b4877f66516f4128176cf722a.
# Usage: run.sh OUTDIR [bench args...]
set -euo pipefail
cd "$(dirname "$0")"
OUT=$1; shift
mkdir -p "$OUT"
# The cu129 nightly ships a torchcodec built against CUDA 13 (OSError on import); audio is unused here.
python3 -m pip uninstall -y -q torchcodec 2>&1 | tail -1 || true
python3 -m pip install -q jsonschema pytest 2>&1 | tail -1 || true
SO=$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')/v1/structured_output
# Installed files must equal vLLM main 6849ca69 before the PR overlay.
if ! cmp -s "$SO/__init__.py" vllm-orig/__init__.py || ! cmp -s "$SO/backend_guidance.py" vllm-orig/backend_guidance.py; then
  cmp -s "$SO/__init__.py" vllm-patched/__init__.py && cmp -s "$SO/backend_guidance.py" vllm-patched/backend_guidance.py \
    || { echo "installed structured_output differs from main 6849ca69"; diff "$SO/__init__.py" vllm-orig/__init__.py | head; exit 2; }
fi
cp vllm-patched/__init__.py vllm-patched/backend_guidance.py "$SO/"
python3 -m pytest -q test_structured_output_manager.py -p no:cacheprovider 2>&1 | tail -15 | tee "$OUT/pytest.log" || true
python3 -u bench_par.py --out "$OUT" "$@" 2>&1 | grep -v "^INFO\|^(EngineCore" | tee "$OUT/bench.log"
