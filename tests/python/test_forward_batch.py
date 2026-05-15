"""ForwardBatch parity + threading determinism tests.

Phase 1 Slice 5 acceptance: ForwardBatch runs N independent sequences
through the process-global thread pool, each on its own thread-local
Workspace. Output must be bitwise identical to calling forward() once
per sequence — batch parallelism doesn't change per-sequence math.
"""

from __future__ import annotations

import os
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


def _make_ids(length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    payload = rng.integers(low=4, high=29, size=length - 2, dtype=np.int32)
    ids = np.concatenate([[esm_cpp.Tokenizer.cls_id], payload, [esm_cpp.Tokenizer.eos_id]]).astype(np.int32)
    mask = np.ones_like(ids, dtype=np.int32)
    return ids, mask


def test_forward_batch_matches_forward_per_sequence(model_8m: esm_cpp.Model) -> None:
    # 8 sequences of mixed length — covers batch > pool size if pool is small.
    lengths = [16, 32, 48, 24, 40, 56, 30, 64]
    ids_list = []
    masks_list = []
    for i, L in enumerate(lengths):
        ids, mask = _make_ids(L, seed=10 + i)
        ids_list.append(ids)
        masks_list.append(mask)

    expected = [model_8m.forward(ids, mask) for ids, mask in zip(ids_list, masks_list)]
    actual = model_8m.forward_batch(ids_list, masks_list)
    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected)):
        np.testing.assert_array_equal(
            a, e, err_msg=f"seq {i}: batch output diverged from single-seq forward"
        )


def test_forward_batch_handles_none_masks(model_8m: esm_cpp.Model) -> None:
    ids_list = [_make_ids(L, seed=L)[0] for L in (16, 24, 32)]
    out = model_8m.forward_batch(ids_list, None)
    assert len(out) == 3
    for arr, ids in zip(out, ids_list):
        assert arr.shape == (len(ids), esm_cpp.Tokenizer.vocab_size)


def test_pool_size_is_positive(model_8m: esm_cpp.Model) -> None:
    n = esm_cpp.Model.num_threads
    assert n >= 1


def test_forward_batch_deterministic_across_thread_counts(model_8m: esm_cpp.Model) -> None:
    # ESM_NUM_THREADS is read at first Model::load and cached, so we can't
    # re-init the pool from inside the same process. This test instead
    # validates that repeated calls within the same pool produce
    # bit-identical results — the partition is deterministic for a given
    # batch size and pool size, so re-running produces the same output.
    ids_list = [_make_ids(L, seed=L * 7)[0] for L in (20, 28, 36, 44)]
    masks_list = [np.ones_like(ids, dtype=np.int32) for ids in ids_list]
    out1 = model_8m.forward_batch(ids_list, masks_list)
    out2 = model_8m.forward_batch(ids_list, masks_list)
    for a, b in zip(out1, out2):
        np.testing.assert_array_equal(a, b)
