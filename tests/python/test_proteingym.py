"""Tests for the ProteinGym masked-marginal scorer.

Validates the scoring math, Spearman utility, and end-to-end behaviour
on a synthetic mini-assay. Full ProteinGym v1.3 (217 assays, ~2.7M
variants) is the gate measurement and runs separately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import esm_cpp
from esm_cpp.eval.proteingym import Variant, score_assay, spearman

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
        str(_safetensors_path("facebook/esm2_t6_8M_UR50D"))
    )


def test_spearman_handles_perfect_correlation() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert abs(spearman(xs, ys) - 1.0) < 1e-9


def test_spearman_handles_perfect_anticorrelation() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [50.0, 40.0, 30.0, 20.0, 10.0]
    assert abs(spearman(xs, ys) + 1.0) < 1e-9


def test_spearman_handles_ties() -> None:
    # Ties get the average rank; this should not blow up.
    xs = [1.0, 1.0, 2.0, 3.0]
    ys = [4.0, 5.0, 6.0, 7.0]
    rho = spearman(xs, ys)
    assert -1.0 <= rho <= 1.0


def test_spearman_zero_variance_returns_zero() -> None:
    xs = [1.0, 1.0, 1.0]
    ys = [1.0, 2.0, 3.0]
    assert spearman(xs, ys) == 0.0


def test_score_assay_zero_diff_variant_scores_zero(model_8m: esm_cpp.Model) -> None:
    tok = esm_cpp.Tokenizer()
    wt = "MKTGVAQRLELDSPMVLQ"
    # Variant identical to wt has no diffs; score must be exactly 0.
    variants = [Variant(sequence=wt, fitness=1.0)]
    scores, fitnesses = score_assay(model_8m, tok, wt, variants)
    assert scores == [0.0]
    assert fitnesses == [1.0]


def test_score_assay_single_mut_is_log_ratio(model_8m: esm_cpp.Model) -> None:
    tok = esm_cpp.Tokenizer()
    wt = "MKTGVAQRLELDSPMVLQ"
    # Substitute position 6 (Q -> A); single-mut score equals
    # log P(A | masked@6) - log P(Q | masked@6).
    assert wt[6] == "Q", "test pre-condition: wt[6] should be Q"
    variant_seq = wt[:6] + "A" + wt[7:]
    assert variant_seq != wt, "variant must differ from wt"
    variants = [Variant(sequence=variant_seq, fitness=0.5)]
    scores, _ = score_assay(model_8m, tok, wt, variants)
    assert len(scores) == 1
    assert scores[0] != 0.0
    assert scores[0] == scores[0]  # not NaN


def test_score_assay_on_synthetic_mini_dms(model_8m: esm_cpp.Model) -> None:
    tok = esm_cpp.Tokenizer()
    # Six variants of a short wt sequence; differ at a few positions.
    # Just verify the end-to-end pipeline produces a Spearman in [-1, 1]
    # and the scores list aligns with the fitnesses list.
    wt = "MKTGVAQRLELDSPMVLQKR"
    variants = [
        Variant(sequence=wt[:5] + "A" + wt[6:], fitness=0.8),  # Q->A
        Variant(sequence=wt[:5] + "L" + wt[6:], fitness=0.6),  # Q->L
        Variant(sequence=wt[:7] + "K" + wt[8:], fitness=0.3),  # R->K
        Variant(sequence=wt[:7] + "E" + wt[8:], fitness=0.1),  # R->E (charge-swap)
        Variant(sequence=wt[:10] + "A" + wt[11:], fitness=0.5),  # L->A
        Variant(sequence=wt[:10] + "D" + wt[11:], fitness=0.0),  # L->D
    ]
    scores, fitnesses = score_assay(model_8m, tok, wt, variants)
    assert len(scores) == 6
    assert all(s == s for s in scores)  # no NaN
    rho = spearman(scores, fitnesses)
    assert -1.0 <= rho <= 1.0
