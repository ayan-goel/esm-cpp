"""Phase 3 Slice 4: GGUF round-trip parity for ESM-2 weights."""

from __future__ import annotations

import tempfile
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
    return esm_cpp.Model.load_from_safetensors(
        str(_safetensors_path("facebook/esm2_t6_8M_UR50D")))


def test_gguf_round_trip_preserves_forward(model_8m: esm_cpp.Model) -> None:
    tok = esm_cpp.Tokenizer()
    ids = np.asarray(tok.encode("MKTGVAQRLELDSPMVLQ"), dtype=np.int32)
    expected = model_8m.forward(ids)

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        out_path = Path(f.name)
    try:
        model_8m.save_to_gguf(str(out_path))
        loaded = esm_cpp.Model.load_from_gguf(str(out_path))
        got = loaded.forward(ids)
        np.testing.assert_allclose(got, expected, rtol=0, atol=0,
                                    err_msg="GGUF round-trip changed logits")
    finally:
        out_path.unlink(missing_ok=True)


def test_gguf_round_trip_preserves_config(model_8m: esm_cpp.Model) -> None:
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        out_path = Path(f.name)
    try:
        model_8m.save_to_gguf(str(out_path))
        loaded = esm_cpp.Model.load_from_gguf(str(out_path))
        cfg_a = model_8m.config
        cfg_b = loaded.config
        assert cfg_a.num_hidden_layers == cfg_b.num_hidden_layers
        assert cfg_a.hidden_size == cfg_b.hidden_size
        assert cfg_a.num_attention_heads == cfg_b.num_attention_heads
        assert cfg_a.head_dim == cfg_b.head_dim
        assert cfg_a.intermediate_size == cfg_b.intermediate_size
        assert cfg_a.vocab_size == cfg_b.vocab_size
        assert cfg_a.token_dropout == cfg_b.token_dropout
        assert cfg_a.mask_token_id == cfg_b.mask_token_id
        assert abs(cfg_a.layer_norm_eps - cfg_b.layer_norm_eps) < 1e-9
    finally:
        out_path.unlink(missing_ok=True)


def test_gguf_auto_detect_via_load(model_8m: esm_cpp.Model) -> None:
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        out_path = Path(f.name)
    try:
        model_8m.save_to_gguf(str(out_path))
        # Load via the auto-detect entry point — should pick GGUF.
        loaded = esm_cpp.Model.load(str(out_path))
        assert loaded.config.num_hidden_layers == model_8m.config.num_hidden_layers
    finally:
        out_path.unlink(missing_ok=True)


def test_gguf_file_size_is_reasonable(model_8m: esm_cpp.Model) -> None:
    """Sanity bound: GGUF FP32 ≈ 4 bytes/param + small metadata overhead."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        out_path = Path(f.name)
    try:
        model_8m.save_to_gguf(str(out_path))
        size = out_path.stat().st_size
        # 8M has ~7.5M parameters → ~30 MB; bound the spread loosely so
        # the test doesn't break on small metadata changes.
        assert 25 * 1024 * 1024 < size < 50 * 1024 * 1024, (
            f"unexpected GGUF size: {size / 1024 / 1024:.1f} MB")
    finally:
        out_path.unlink(missing_ok=True)
