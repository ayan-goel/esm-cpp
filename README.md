# esm.cpp

CPU-first C++ inference engine for ESM-2 protein language models, with Python bindings.

**Status: Phase 0 complete.** Scalar FP32 forward passes HF parity on ESM-2-8M and 35M. See [SPEC.md](SPEC.md), [tasks/plan.md](tasks/plan.md), and the Phase 0 retro at [notes/phase0.md](notes/phase0.md).

## Reproduce the Phase 0 gate

```bash
# One-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Capture HF goldens (~1 min for 8M, ~2 min for 35M)
python tools/capture_golden.py --model esm2_t6_8M  --num-seqs 100 --out tests/golden/esm2_t6_8M
python tools/capture_golden.py --model esm2_t12_35M --num-seqs 100 --out tests/golden/esm2_t12_35M

# Run parity (default 30 seqs; export ESM_CPP_PARITY_NSEQS=100 for the full set)
pytest tests/python/test_against_hf.py -v -s
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/python -v
```
