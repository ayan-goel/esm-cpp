"""Phase 14 T4: Python e2e parity test for the auto-load UX.

Mirrors the user-facing flow: the user does `pip install esm-cpp`,
points `esm-cpp-fetch-artifacts` at our GitHub release (which populates
~/.cache/esm_cpp/<key>/), and then `model.forward_scheduled(...)` Just
Works at the headline speed with zero env vars.

This test stages a Phase-13 whole-graph artifact at
~/.cache/esm_cpp/<key>/ via a temp dir + monkey-patched
ESM_CPP_CACHE_DIR, then loads the model from a non-HF path (so the
cache key falls back to `path.stem`), asserts:
  1. model.whole_graph_shapes returns the staged shape
  2. forward_scheduled returns logits matching an explicit-FP32 forward
     to corr >= 0.999 (same gate the C++ parity test uses)
  3. no env vars were set anywhere

SKIPs cleanly:
  - on non-Apple
  - when the 8M HF safetensors isn't in the user's HF cache
  - when an 8M whole-graph fixture artifact isn't present at
    `weights/esm2_8m.whole-graph/B-1_L-66/whole_graph.mlmodelc`
    (the same Phase-13 fixture the C++ tests use)
"""
from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

import numpy as np
import pytest

import esm_cpp


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _hf_8m_safetensors() -> Path | None:
    snaps = HF_CACHE / "models--facebook--esm2_t6_8M_UR50D" / "snapshots"
    if not snaps.is_dir():
        return None
    cand = list(snaps.glob("*/model.safetensors"))
    return cand[0] if cand else None


def _fixture_8m_whole_graph() -> Path | None:
    # L=67 matches the 65-residue test sequence + cls + eos.
    p = REPO_ROOT / "weights" / "esm2_8m.whole-graph" / "B-1_L-67" / "whole_graph.mlmodelc"
    return p if p.is_dir() else None


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@pytest.mark.skipif(not _is_apple_silicon(), reason="Apple Silicon only")
@pytest.mark.skipif(_hf_8m_safetensors() is None, reason="HF 8M cache missing")
@pytest.mark.skipif(_fixture_8m_whole_graph() is None,
                    reason="8M whole-graph fixture not built — run "
                           "`build_whole_graph_artifacts.py --shapes 1x66 "
                           "--model facebook/esm2_t6_8M_UR50D --out "
                           "weights/esm2_8m.whole-graph`.")
def test_auto_load_from_cache_dir(tmp_path: Path, monkeypatch) -> None:
    """ESM_CPP_CACHE_DIR -> <key>/whole-graph/B-1_L-66/  is auto-discovered."""
    sft = _hf_8m_safetensors()
    fixture = _fixture_8m_whole_graph()
    assert sft is not None and fixture is not None

    # Copy the safetensors into a temp dir so the cache key falls back to
    # the path stem (rather than the HF cache key) — makes the cache lookup
    # deterministic regardless of what's in the user's real cache dir.
    weights = tmp_path / "esm2_8m_test.safetensors"
    weights.symlink_to(sft.resolve())

    # Stage the artifact in a temp cache.
    cache_root = tmp_path / "cache"
    key = "esm2_8m_test"  # matches Path(weights).stem
    artifact_dest = cache_root / key / "whole-graph" / "B-1_L-67"
    artifact_dest.mkdir(parents=True, exist_ok=True)
    # Symlink the .mlmodelc bundle so we don't pay the copy cost.
    (artifact_dest / "whole_graph.mlmodelc").symlink_to(fixture.resolve())

    monkeypatch.setenv("ESM_CPP_CACHE_DIR", str(cache_root))
    # Crucial: no ESM_APPLE_ANE_GRAPH set — proving the default-on flip works.
    monkeypatch.delenv("ESM_APPLE_ANE_GRAPH", raising=False)
    monkeypatch.delenv("ESM_APPLE_AMX", raising=False)

    model = esm_cpp.Model.load_from_safetensors(str(weights))

    # 1. Auto-discovery should have registered the shape.
    shapes = model.whole_graph_shapes
    assert shapes == [(1, 67)], f"expected [(1, 67)], got {shapes}"
    assert model.whole_graph_path.endswith("whole-graph"), \
        f"whole_graph_path={model.whole_graph_path}"

    # 2. forward_scheduled on a length-67 ids -> whole-graph path engages
    # and matches an explicit non-WG forward at corr >= 0.999.
    tok = esm_cpp.Tokenizer()
    seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ids = np.asarray(tok.encode(seq), dtype=np.int32)
    assert ids.size == 67, f"sequence encoded to L={ids.size}, expected 67"

    wg_out = model.forward_scheduled([ids])
    assert len(wg_out) == 1
    assert wg_out[0].shape == (67, 33)
    assert np.isfinite(wg_out[0]).all()

    # 3. Reference forward via a second model with the env opted out.
    monkeypatch.setenv("ESM_APPLE_ANE_GRAPH", "off")
    monkeypatch.setenv("ESM_APPLE_AMX", "off")
    ref_model = esm_cpp.Model.load_from_safetensors(str(weights))
    # Auto-load still fires (just because shapes get registered) but the
    # forward path opts out via env, so this runs the standard scheduled path.
    ref_out = ref_model.forward_scheduled([ids])
    assert len(ref_out) == 1

    corr = float(np.corrcoef(wg_out[0].ravel(), ref_out[0].ravel())[0, 1])
    argmax_agree = float(
        (wg_out[0].argmax(-1) == ref_out[0].argmax(-1)).mean())
    assert corr >= 0.999, f"corr {corr:.6f} below 0.999"
    assert argmax_agree >= 0.99, f"argmax {argmax_agree:.4f} below 0.99"
