"""Phase 14 T5: shared manifest helper for Apple artifacts.

Writes a small `esm_cpp_artifact.json` next to each .mlmodelc bundle and
inside each AMX artifact root, so the C++ Model::Load* can detect stale
artifacts after a trace change (`tools/esm_traceable.py` / the builder
scripts have a different SHA than the artifact was built against).

Schema (versioned by `manifest_version`):
  {
    "manifest_version": 1,
    "trace_sha": "<hex 64>",          # SHA-256 of the trace + builder files
    "model_id": "facebook/esm2_t33_650M_UR50D",
    "precision": "fp16",              # fp16 | fp32
    "compute_units": "CPU_AND_NE",    # CPU_ONLY | CPU_AND_NE | ALL
    "kind": "amx-fp16" | "whole-graph",
    "shape": [B, L],                  # only for kind=whole-graph
    "created_at": "2026-05-25T16:00:00Z",
    "tool_version": "esm-cpp 0.2.0"
  }

The C++ side reads + checks `trace_sha`; a mismatch logs one warning and
keeps using the artifact (the user explicitly downloaded / built it;
this is informational, not fatal).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "esm_cpp_artifact.json"
MANIFEST_VERSION = 1

# Files contributing to the trace SHA. Order is stable; new files added
# at end so a SHA change is meaningful (= traceable forward graph or
# builder logic changed, not just incidental refactoring).
_TRACE_INPUTS = [
    "tools/esm_traceable.py",
    "tools/build_whole_graph_artifacts.py",
    "tools/build_amx_artifacts.py",
]


def _repo_root() -> Path:
    # Best-effort: the helper lives at <repo>/tools/_artifact_manifest.py.
    return Path(__file__).resolve().parent.parent


def compute_trace_sha() -> str:
    """SHA-256 of the trace + builder source files.

    Algorithm (chosen for trivial portability to CMake `file(SHA256)`):
      For each present input file, compute file's hex SHA-256.
      Build a string "<rel>=<hex_sha>|<rel>=<hex_sha>|...".
      Take SHA-256 of that string.

    CMake's `file(SHA256 path var)` reads bytes and returns the hex
    string, matching `hashlib.sha256(p.read_bytes()).hexdigest()` exactly.
    The composition string then matches between the two implementations.
    """
    parts = []
    root = _repo_root()
    for rel in _TRACE_INPUTS:
        p = root / rel
        if not p.is_file():
            # Skip missing files (partial checkout) — the SHA will diverge
            # from a full checkout. That's the right signal.
            continue
        file_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        parts.append(f"{rel}={file_sha}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _now_iso_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_manifest(
    dir_path: Path,
    *,
    kind: str,
    model_id: str,
    precision: str,
    compute_units: str,
    shape: tuple[int, int] | None = None,
    tool_version: str = "esm-cpp",
) -> Path:
    """Write `esm_cpp_artifact.json` into `dir_path`.

    `dir_path` is the artifact root for AMX (the dir containing all the
    *.mlmodelc subdirs) or the directory containing a single
    `whole_graph.mlmodelc` for whole-graph. The manifest lives at the
    same level as the .mlmodelc bundles, NOT inside them.
    """
    if kind not in ("amx-fp16", "whole-graph"):
        raise ValueError(f"kind must be amx-fp16 or whole-graph, got {kind!r}")
    payload: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "trace_sha": compute_trace_sha(),
        "model_id": model_id,
        "precision": precision,
        "compute_units": compute_units,
        "kind": kind,
        "created_at": _now_iso_utc(),
        "tool_version": tool_version,
    }
    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f"shape must be (B, L), got {shape!r}")
        payload["shape"] = [int(shape[0]), int(shape[1])]
    out = dir_path / MANIFEST_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def read_manifest(dir_path: Path) -> dict[str, Any] | None:
    """Read the manifest from `dir_path`, returning None if absent / invalid."""
    p = dir_path / MANIFEST_NAME
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data
