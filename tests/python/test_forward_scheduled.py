"""Phase 3 Slice 2: scheduled batch parity + order-preservation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import esm_cpp

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


def _encode(tok: esm_cpp.Tokenizer, seq: str) -> np.ndarray:
    return np.asarray(tok.encode(seq), dtype=np.int32)


def test_forward_scheduled_matches_serial(model_8m: esm_cpp.Model) -> None:
    tok = esm_cpp.Tokenizer()
    seqs = [
        "MKTGVAQRLELDSPMVLQ",
        "MAGAASPCANGCGPSAPSDAEVVHLCRSL",
        "MSEEKRGGQATKLP",
        "MKTGVAQRLELDSP",
    ]
    ids_list = [_encode(tok, s) for s in seqs]
    scheduled = model_8m.forward_scheduled(ids_list)
    serial = [model_8m.forward(ids) for ids in ids_list]
    assert len(scheduled) == len(serial)
    for i, (s, r) in enumerate(zip(scheduled, serial)):
        np.testing.assert_allclose(
            s, r, rtol=1e-4, atol=1e-4, err_msg=f"seq {i}")


def test_forward_scheduled_preserves_input_order(
    model_8m: esm_cpp.Model,
) -> None:
    """Outputs must come back in input order, even when bucket-split
    fires on length-imbalanced inputs and reorders internally."""
    tok = esm_cpp.Tokenizer()
    seqs_long = ["M" + ("AGAASPCANGCGPSAPSDAEVVHLCRSL" * 4)] * 4  # ~113 each
    seqs_short = ["MKTGV"] * 4  # 5 each
    # Interleave: short, long, short, long, ... to force a non-trivial reorder.
    interleaved = []
    for s, l in zip(seqs_short, seqs_long):
        interleaved += [s, l]
    ids_list = [_encode(tok, s) for s in interleaved]
    # max/mean is dramatic here -> bucket-split should fire.
    scheduled = model_8m.forward_scheduled(ids_list)
    # Verify each output corresponds to its input position by length match.
    assert len(scheduled) == len(ids_list)
    for i, (logits, ids) in enumerate(zip(scheduled, ids_list)):
        assert logits.shape[0] == len(ids), (
            f"position {i}: expected length {len(ids)}, got {logits.shape[0]}")


def test_forward_scheduled_single_sequence(model_8m: esm_cpp.Model) -> None:
    tok = esm_cpp.Tokenizer()
    ids = _encode(tok, "MKTGVAQRLELDSPMVLQ")
    scheduled = model_8m.forward_scheduled([ids])
    serial = model_8m.forward(ids)
    np.testing.assert_allclose(scheduled[0], serial, rtol=1e-5, atol=1e-5)


def test_forward_scheduled_handles_empty_input(model_8m: esm_cpp.Model) -> None:
    assert model_8m.forward_scheduled([]) == []


def test_forward_scheduled_respects_max_batch_size(
    model_8m: esm_cpp.Model,
) -> None:
    """max_batch_size chunks the dispatch but doesn't affect numerics."""
    tok = esm_cpp.Tokenizer()
    ids_list = [_encode(tok, "MKTGV" * 5) for _ in range(8)]
    a = model_8m.forward_scheduled(ids_list, max_batch_size=8)
    b = model_8m.forward_scheduled(ids_list, max_batch_size=2)
    for i, (x, y) in enumerate(zip(a, b)):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-4,
                                    err_msg=f"chunking changed seq {i}")
