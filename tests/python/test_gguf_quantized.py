"""Phase 3 Slice 5: quantized GGUF round-trip parity."""

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


def test_quantized_gguf_round_trip_preserves_forward() -> None:
    tok = esm_cpp.Tokenizer()
    ids = np.asarray(tok.encode("MKTGVAQRLELDSPMVLQ"), dtype=np.int32)

    # In-memory: load + quantize.
    in_mem = esm_cpp.Model.load_from_safetensors(
        str(_safetensors_path("facebook/esm2_t6_8M_UR50D")))
    in_mem.quantize_weights()
    expected = in_mem.forward(ids)

    # Round-trip: write quantized GGUF, load it back, forward.
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        out_path = Path(f.name)
    try:
        in_mem.save_to_gguf(str(out_path))
        loaded = esm_cpp.Model.load_from_gguf(str(out_path))
        assert loaded.config.weights_quantized is True
        got = loaded.forward(ids)
        np.testing.assert_allclose(
            got, expected, rtol=0, atol=0,
            err_msg="quantized GGUF round-trip changed logits")
    finally:
        out_path.unlink(missing_ok=True)


def test_quantized_gguf_loads_standalone() -> None:
    """Quantized GGUF should be self-contained: no FP32 source needed."""
    in_mem = esm_cpp.Model.load_from_safetensors(
        str(_safetensors_path("facebook/esm2_t6_8M_UR50D")))
    in_mem.quantize_weights()

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        out_path = Path(f.name)
    try:
        in_mem.save_to_gguf(str(out_path))
        # Free the in-memory model; reload from GGUF only.
        del in_mem
        loaded = esm_cpp.Model.load_from_gguf(str(out_path))
        assert loaded.config.weights_quantized is True
        tok = esm_cpp.Tokenizer()
        ids = np.asarray(tok.encode("MKTGV"), dtype=np.int32)
        # Just verify a forward runs without error.
        logits = loaded.forward(ids)
        assert logits.shape[0] == len(ids)
        assert logits.shape[1] == esm_cpp.Tokenizer.vocab_size
    finally:
        out_path.unlink(missing_ok=True)


def test_quantized_gguf_file_size_smaller_than_fp32() -> None:
    """INT8 weights should shrink the GGUF file substantially."""
    fp32_model = esm_cpp.Model.load_from_safetensors(
        str(_safetensors_path("facebook/esm2_t6_8M_UR50D")))
    q_model = esm_cpp.Model.load_from_safetensors(
        str(_safetensors_path("facebook/esm2_t6_8M_UR50D")))
    q_model.quantize_weights()

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f_fp:
        fp_path = Path(f_fp.name)
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f_q:
        q_path = Path(f_q.name)
    try:
        fp32_model.save_to_gguf(str(fp_path))
        q_model.save_to_gguf(str(q_path))
        fp32_size = fp_path.stat().st_size
        q_size = q_path.stat().st_size
        # INT8 weights take ~1/4 the bytes of FP32. lm_head + embed
        # stay FP32, so the ratio isn't a full 4x. Bound: quantized
        # GGUF < 75% of FP32 GGUF.
        assert q_size < 0.75 * fp32_size, (
            f"quantized GGUF not appreciably smaller: "
            f"fp32={fp32_size}, q={q_size}")
    finally:
        fp_path.unlink(missing_ok=True)
        q_path.unlink(missing_ok=True)


def test_quantized_gguf_records_first_block_fp16_flag() -> None:
    in_mem = esm_cpp.Model.load_from_safetensors(
        str(_safetensors_path("facebook/esm2_t6_8M_UR50D")))
    in_mem.quantize_weights()
    in_mem.set_first_block_fc1_fp16(True)

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        out_path = Path(f.name)
    try:
        in_mem.save_to_gguf(str(out_path))
        loaded = esm_cpp.Model.load_from_gguf(str(out_path))
        assert loaded.config.weights_quantized is True
        assert loaded.config.first_block_fc1_fp16 is True
    finally:
        out_path.unlink(missing_ok=True)
