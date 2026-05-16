"""Phase 3 Slice 1: packed cu_seqlens forward parity tests.

The packed forward must be bit-equivalent (to FP32 round-off) to running
each sequence through serial Forward. The load-bearing correctness risk
is the per-sequence token_dropout rescale: the embed_layer computes one
mask_ratio per sequence span, not a global one — get this wrong and
every PPPL measurement silently drifts.
"""

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


def test_packed_forward_matches_serial(model_8m: esm_cpp.Model) -> None:
    """Three sequences of varying length; packed[i] == serial(seq_i)."""
    tok = esm_cpp.Tokenizer()
    seqs = [
        "MKTGVAQRLELDSPMVLQ",
        "MAGAASPCANGCGPSAPSDAEVVHLCRSL",
        "MSEEKRGGQATKLP",
    ]
    ids_list = [_encode(tok, s) for s in seqs]

    # Serial reference: one forward per sequence.
    serial = [model_8m.forward(ids) for ids in ids_list]

    # Packed: concatenate ids, build cu_seqlens, one packed forward.
    packed_ids = np.concatenate(ids_list).astype(np.int32)
    cu = np.zeros(len(ids_list) + 1, dtype=np.int32)
    for i, ids in enumerate(ids_list):
        cu[i + 1] = cu[i] + len(ids)
    packed = model_8m.forward_packed(packed_ids, cu)

    assert len(packed) == len(serial)
    for i, (p, s) in enumerate(zip(packed, serial)):
        assert p.shape == s.shape, f"seq {i}: {p.shape} vs {s.shape}"
        # FP32 reordering tolerance: the packed path runs one big GEMM per
        # Linear instead of N small ones, so summation order differs.
        np.testing.assert_allclose(
            p, s, rtol=1e-4, atol=1e-4,
            err_msg=f"packed[{i}] != serial[{i}]")


def test_packed_forward_single_sequence_matches_forward(
    model_8m: esm_cpp.Model,
) -> None:
    """B=1 case: packed forward must collapse to the same numbers as forward()."""
    tok = esm_cpp.Tokenizer()
    ids = _encode(tok, "MKTGVAQRLELDSPMVLQ")
    serial = model_8m.forward(ids)
    cu = np.array([0, len(ids)], dtype=np.int32)
    packed = model_8m.forward_packed(ids, cu)
    assert len(packed) == 1
    np.testing.assert_allclose(packed[0], serial, rtol=1e-5, atol=1e-5)


def test_packed_forward_per_sequence_mask_ratio(
    model_8m: esm_cpp.Model,
) -> None:
    """Per-sequence token_dropout rescale is load-bearing.

    Pack two sequences: one with no masks, one with a mask token. Each
    sequence's logits must match its own serial forward — which means
    each used its own mask_ratio for the 0.88 rescale, not a global avg.
    """
    tok = esm_cpp.Tokenizer()
    mask_id = esm_cpp.Tokenizer.mask_id

    ids_unmasked = _encode(tok, "MKTGVAQRLELDSPMVLQ")
    ids_with_mask = _encode(tok, "MAGAASPCANGCGPSAPSDAEVVHLCRSL").copy()
    # Replace one inner residue with <mask> to introduce a non-zero mask
    # ratio in just this sequence.
    ids_with_mask[5] = mask_id

    # Serial: each sequence gets its own mask_ratio in Embed.
    serial_a = model_8m.forward(ids_unmasked)
    serial_b = model_8m.forward(ids_with_mask)

    # Packed: if the per-sequence rescale is wrong, sequence A would
    # silently see sequence B's mask ratio (or an average), producing
    # a different first-layer hidden state.
    packed_ids = np.concatenate([ids_unmasked, ids_with_mask]).astype(np.int32)
    cu = np.array(
        [0, len(ids_unmasked), len(ids_unmasked) + len(ids_with_mask)],
        dtype=np.int32,
    )
    packed = model_8m.forward_packed(packed_ids, cu)
    np.testing.assert_allclose(packed[0], serial_a, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(packed[1], serial_b, rtol=1e-4, atol=1e-4)


def test_packed_forward_validates_cu_seqlens(model_8m: esm_cpp.Model) -> None:
    tok = esm_cpp.Tokenizer()
    ids = _encode(tok, "MKTGVA")
    with pytest.raises(Exception):
        # cu_seqlens last entry doesn't match packed_ids size
        model_8m.forward_packed(ids, np.array([0, 4], dtype=np.int32))
    with pytest.raises(Exception):
        # cu_seqlens too short
        model_8m.forward_packed(ids, np.array([0], dtype=np.int32))


def test_packed_forward_workspace_stays_zero_alloc(
    model_8m: esm_cpp.Model,
) -> None:
    """After the first packed forward at a given total-tokens shape, the
    workspace capacity should not grow on subsequent identical calls."""
    tok = esm_cpp.Tokenizer()
    seqs = ["MKTGVAQRLELDSPMVLQ", "MAGAASPCANGCGPSAPS"]
    ids_list = [_encode(tok, s) for s in seqs]
    packed_ids = np.concatenate(ids_list).astype(np.int32)
    cu = np.zeros(len(ids_list) + 1, dtype=np.int32)
    for i, ids in enumerate(ids_list):
        cu[i + 1] = cu[i] + len(ids)
    # Warmup.
    model_8m.forward_packed(packed_ids, cu)
    cap_after_first = model_8m.workspace_capacity_bytes
    # Repeat — no growth allowed.
    for _ in range(3):
        model_8m.forward_packed(packed_ids, cu)
        assert model_8m.workspace_capacity_bytes == cap_after_first
