"""Smoke tests for the exact pseudo-perplexity (PPPL) harness.

Phase 2 Slice 1 requires:
  - PPPL is finite and positive on real protein sequences.
  - The harness exposes a callable that takes a Model + list of seqs.
  - PPPL of natural sequences is far below PPPL of uniformly-random
    "sequences" (sanity: the model is more surprised by garbage).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import esm_cpp
from esm_cpp.eval.pppl import pppl

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _safetensors_path(hf_id: str) -> Path:
    snapshots = HF_CACHE / f"models--{hf_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        pytest.skip(f"HF cache missing: {snapshots}")
    candidates = list(snapshots.glob("*/model.safetensors"))
    if not candidates:
        pytest.skip(f"no model.safetensors under {snapshots}")
    return candidates[0]


@pytest.fixture(scope="module")
def model_8m() -> esm_cpp.Model:
    path = _safetensors_path("facebook/esm2_t6_8M_UR50D")
    return esm_cpp.Model.load_from_safetensors(str(path))


REAL_SEQS = [
    # Short real protein fragments (no `<` or special chars; canonical aa).
    "MKTGVAQRLELDSPMVLQKRSGE",
    "MAGAASPCANGCGPSAPSDAEVVHLCRSLEVGTVMTLF",
    "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMG",
]


def test_pppl_returns_finite_positive_value(model_8m: esm_cpp.Model) -> None:
    tok = esm_cpp.Tokenizer()
    value = pppl(model_8m, tok, REAL_SEQS[:1])
    assert np.isfinite(value)
    assert value > 1.0
    assert value < 100.0, f"PPPL = {value} suspiciously high"


def test_pppl_random_sequence_is_worse_than_real(model_8m: esm_cpp.Model) -> None:
    tok = esm_cpp.Tokenizer()
    real = pppl(model_8m, tok, REAL_SEQS)
    # Random amino-acid permutation (deterministic for reproducibility).
    # The model should be more surprised by it than by real proteins,
    # since real sequences have evolutionary constraints the model has
    # learned. (A repeated-single-residue sequence does NOT qualify:
    # masking one X among Xs is trivially predictable. Random does.)
    rng = np.random.default_rng(0xc0de)
    canonical = "LAGVSERTIDPKQNFYMHWC"  # 20 canonical aa
    random_seq = "".join(rng.choice(list(canonical), size=40))
    garbage = pppl(model_8m, tok, [random_seq])
    assert garbage > real, (
        f"Sanity check failed: random PPPL {garbage:.2f} <= real {real:.2f}; "
        "model should be more surprised by a random sequence than by real proteins."
    )


def test_pppl_returns_same_value_across_runs(model_8m: esm_cpp.Model) -> None:
    tok = esm_cpp.Tokenizer()
    a = pppl(model_8m, tok, REAL_SEQS)
    b = pppl(model_8m, tok, REAL_SEQS)
    assert a == b, f"PPPL is non-deterministic: {a} vs {b}"
