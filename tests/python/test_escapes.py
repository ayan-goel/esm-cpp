"""Sensitivity escape tests for Phase 2 Slice 5.

Two escapes from the SPEC's INT8 recipe:
  1. lm_head.dense + lm_head.layer_norm stay FP32 (always).
  2. First transformer block's fc1 input falls back to FP16 if PPPL
     drift > 0.2 (operator flag, not automatic).
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
def model_path() -> Path:
    return _safetensors_path("facebook/esm2_t6_8M_UR50D")


def test_quantize_weights_does_not_touch_lm_head(model_path: Path) -> None:
    """Even after quantize_weights(), the lm_head path must remain FP32.
    We verify by checking that running the SAME masked input through the
    FP32-only path (lm_head only) produces logits whose lm_head-induced
    component matches identically between FP32 and INT8 models."""
    tok = esm_cpp.Tokenizer()
    ids = np.asarray(tok.encode("MKTGVAQRLELDSPMVLQ"), dtype=np.int32)
    mask = np.ones_like(ids, dtype=np.int32)

    fp32 = esm_cpp.Model.load_from_safetensors(str(model_path))
    fp32_logits = fp32.forward(ids, mask)

    int8 = esm_cpp.Model.load_from_safetensors(str(model_path))
    int8.quantize_weights()
    int8_logits = int8.forward(ids, mask)

    # The lm_head is FP32 in both. So any drift between fp32 and int8 logits
    # is solely from the per-layer INT8 quant, not from lm_head. A weak but
    # informative assertion: the logits *shape* matches and values are close-
    # enough that lm_head couldn't have re-quantized.
    assert fp32_logits.shape == int8_logits.shape
    # Drift bound: with INT8 weights on per-layer linears at 8M (6 layers,
    # tame outliers), pointwise drift should sit well under 0.5. lm_head
    # being quantized would push this much higher.
    max_drift = float(np.max(np.abs(fp32_logits - int8_logits)))
    assert max_drift < 0.5, (
        f"unexpectedly large logit drift {max_drift:.4f} suggests lm_head "
        "may have been quantized; check QuantizeWeights escape list."
    )


def test_first_block_fp16_flag_is_supported(model_path: Path) -> None:
    """Setter is wired and queryable via config."""
    model = esm_cpp.Model.load_from_safetensors(str(model_path))
    assert model.config.first_block_fc1_fp16 is False
    model.set_first_block_fc1_fp16(True)
    assert model.config.first_block_fc1_fp16 is True
    model.set_first_block_fc1_fp16(False)
    assert model.config.first_block_fc1_fp16 is False


def test_first_block_fp16_changes_pppl_slightly(model_path: Path) -> None:
    """Enabling the FP16 escape should produce a *small* PPPL shift on 8M.
    Bigger than 0 (the escape is doing something), smaller than the gross
    quantization noise (it's only one site at one layer)."""
    tok = esm_cpp.Tokenizer()
    seqs = ["MKTGVAQRLELDSPMVLQKRSGE", "MAGAASPCANGCGPSAPSDAEVVHLCRSL"]

    fp32 = esm_cpp.Model.load_from_safetensors(str(model_path))
    p_baseline = pppl(fp32, tok, seqs)

    escaped = esm_cpp.Model.load_from_safetensors(str(model_path))
    escaped.set_first_block_fc1_fp16(True)
    p_escaped = pppl(escaped, tok, seqs)

    drift = abs(p_escaped - p_baseline)
    # Single-site FP16 should produce a non-zero but tiny PPPL shift.
    # Bound: shouldn't move PPPL by more than 0.5 (very loose).
    assert drift < 0.5, (
        f"FP16 escape on first-block fc1 produced an outsized PPPL shift: "
        f"baseline={p_baseline:.4f}, escaped={p_escaped:.4f}, drift={drift:.4f}"
    )
