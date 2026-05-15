"""W8A16 PPPL drift smoke test.

After Model.quantize_weights(), the per-layer Linear projections route
through LinearInt8 (per-channel symmetric INT8 weights, FP32 activations).
At 8M with W8A16 the PPPL drift vs FP32 should be tiny (~0.05) because
the model has only 6 layers and weight-only quant doesn't compound
activation outlier issues.

8M is not a shipping INT8 target (SPEC: below 150M ships FP32 only).
This test is the "is the INT8 forward producing sane logits?" smoke
check, not a quality claim. The actual gate measurement is 150M / 650M
on the full 1000-seq UniRef50 holdout (Slice 7 hand-off).
"""

from __future__ import annotations

from pathlib import Path

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


REAL_SEQS = [
    "MKTGVAQRLELDSPMVLQKRSGE",
    "MAGAASPCANGCGPSAPSDAEVVHLCRSLEVGTVMTLF",
    "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMG",
]


def test_w8a16_pppl_drift_within_smoke_budget() -> None:
    path = _safetensors_path("facebook/esm2_t6_8M_UR50D")
    tok = esm_cpp.Tokenizer()

    fp32 = esm_cpp.Model.load_from_safetensors(str(path))
    pppl_fp32 = pppl(fp32, tok, REAL_SEQS)

    quant = esm_cpp.Model.load_from_safetensors(str(path))
    assert quant.config.weights_quantized is False
    quant.quantize_weights()
    assert quant.config.weights_quantized is True
    pppl_int8 = pppl(quant, tok, REAL_SEQS)

    drift = abs(pppl_int8 - pppl_fp32)
    # 8M W8A16 on 3 short sequences: drift should be << 1.0. Loose budget
    # (0.5) for the smoke; production gate (Slice 7) is < 0.1 at 150M/650M
    # on the 1000-seq UniRef50 holdout.
    assert drift < 0.5, (
        f"FP32 PPPL = {pppl_fp32:.4f}, W8A16 PPPL = {pppl_int8:.4f}, "
        f"drift = {drift:.4f}; expected < 0.5 on the 3-seq smoke."
    )


def test_quantize_weights_idempotent_at_config_level() -> None:
    path = _safetensors_path("facebook/esm2_t6_8M_UR50D")
    model = esm_cpp.Model.load_from_safetensors(str(path))
    model.quantize_weights()
    assert model.config.weights_quantized
    # Re-quantizing is a no-op at the config level (already flagged).
    # We don't currently dedup the work but the flag protects against
    # silent state regressions.
    model.quantize_weights()
    assert model.config.weights_quantized
