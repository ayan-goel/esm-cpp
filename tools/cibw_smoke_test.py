"""Smoke test run by cibuildwheel against each freshly-built wheel.

Exercises the pybind11 boundary without needing any weight artifacts:
importing the package loads the C++ extension, and constructing the
Tokenizer + encoding a short sequence forces the C++ symbol table to
resolve. If this script exits 0, the wheel is at least loadable and
linkable on the target Python + ISA + libc combination.

Anything more substantial (a forward pass) would need weights, which
cibuildwheel's quarantined env doesn't have. The full e2e tests in
tests/python/ run separately under the CI workflow.
"""

import sys

import esm_cpp

print("esm-cpp version:", esm_cpp.__version__)
print("python         :", sys.version.split()[0])
print("platform       :", sys.platform)

tok = esm_cpp.Tokenizer()
ids = tok.encode("MKTGVA")
assert len(ids) > 0, f"tokenizer returned no ids for non-empty input: {ids!r}"
print("tokenizer ids  :", list(ids))

print("OK")
