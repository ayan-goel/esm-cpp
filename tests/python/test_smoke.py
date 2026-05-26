"""Smoke test: package imports and reports a sensible, consistent version."""

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

import esm_cpp

# SemVer-ish: major.minor.patch with optional pre-release / build suffix.
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+([-+][A-Za-z0-9.]+)?$")


def test_version_is_semver_string():
    v = esm_cpp.__version__
    assert isinstance(v, str)
    assert _SEMVER_RE.match(v), f"unexpected version: {v!r}"


def test_version_matches_distribution_metadata():
    """esm_cpp/_version.py and pyproject.toml must not drift apart.

    Requires the package to be installed (editable or otherwise) so
    importlib.metadata can resolve it. Skipped if running from a raw
    source checkout without `pip install -e .`.
    """
    try:
        dist_version = pkg_version("esm-cpp")
    except PackageNotFoundError:
        import pytest

        pytest.skip("esm-cpp not installed via pip; metadata unavailable")
    assert esm_cpp.__version__ == dist_version
