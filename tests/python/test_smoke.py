"""Smoke test: package imports and reports its version."""

import esm_cpp


def test_version_is_string():
    assert isinstance(esm_cpp.__version__, str)
    assert esm_cpp.__version__ == "0.1.0"
