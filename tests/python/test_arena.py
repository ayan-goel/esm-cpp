"""Arena / Workspace zero-growth regression test.

Phase 1 Slice 2 requirement: after the first forward at a given seq_len,
the per-Model scratch arena should not grow on subsequent calls at the
same length. Confirms the bump allocator is reused, not reallocated.
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


def _make_ids(length: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    payload = rng.integers(low=4, high=29, size=length - 2, dtype=np.int32)
    ids = np.concatenate([[esm_cpp.Tokenizer.cls_id], payload, [esm_cpp.Tokenizer.eos_id]]).astype(np.int32)
    mask = np.ones_like(ids, dtype=np.int32)
    return ids, mask


def test_arena_does_not_grow_on_repeat_forwards(model_8m: esm_cpp.Model) -> None:
    ids, mask = _make_ids(64)
    # First forward establishes capacity.
    _ = model_8m.forward(ids, mask)
    cap_after_first = model_8m.workspace_capacity_bytes
    assert cap_after_first > 0, "arena should have grown on first forward"
    # Subsequent forwards at the same length must not grow the arena.
    for _ in range(5):
        _ = model_8m.forward(ids, mask)
        assert model_8m.workspace_capacity_bytes == cap_after_first


def test_arena_growth_is_monotonic_when_seq_len_grows(model_8m: esm_cpp.Model) -> None:
    short_ids, short_mask = _make_ids(64)
    _ = model_8m.forward(short_ids, short_mask)
    cap_short = model_8m.workspace_capacity_bytes

    long_ids, long_mask = _make_ids(128)
    _ = model_8m.forward(long_ids, long_mask)
    cap_long = model_8m.workspace_capacity_bytes

    assert cap_long >= cap_short, "longer sequence must not shrink the arena"
    # And rerunning the short forward must not change the (now larger) capacity.
    _ = model_8m.forward(short_ids, short_mask)
    assert model_8m.workspace_capacity_bytes == cap_long
