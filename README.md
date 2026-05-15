# esm.cpp

CPU-first C++ inference engine for ESM-2 protein language models, with Python bindings.

**Status: Phase 0 in progress.** See [SPEC.md](SPEC.md) and [tasks/plan.md](tasks/plan.md).

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
