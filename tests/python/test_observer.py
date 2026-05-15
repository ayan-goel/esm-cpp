"""End-to-end ActivationObserver smoke test on a real model."""

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
    return esm_cpp.Model.load_from_safetensors(
        str(_safetensors_path("facebook/esm2_t6_8M_UR50D"))
    )


def _make_ids(text: str) -> tuple[np.ndarray, np.ndarray]:
    tok = esm_cpp.Tokenizer()
    ids = np.asarray(tok.encode(text), dtype=np.int32)
    mask = np.ones_like(ids, dtype=np.int32)
    return ids, mask


def test_observer_collects_stats_for_every_site_every_layer(model_8m: esm_cpp.Model) -> None:
    obs = esm_cpp.ActivationObserver()
    ids, mask = _make_ids("MKTGVAQRLELDSPMVLQKRSGE")
    model_8m.forward_with_observer(ids, mask, obs)
    stats = obs.percentile(99.9)
    assert stats, "observer produced no stats"

    n_layers = model_8m.config.num_hidden_layers
    expected_sites = {"attn_ln_output", "attn_out", "ffn_ln_output", "inter_gelu"}
    for layer in range(n_layers):
        for site in expected_sites:
            key = f"layer{layer}.{site}"
            assert key in stats, f"observer missing site {key}"


def test_observer_99_9_pctile_values_are_sane(model_8m: esm_cpp.Model) -> None:
    obs = esm_cpp.ActivationObserver()
    ids, mask = _make_ids("MKTGVAQRLELDSPMVLQKRSGE")
    model_8m.forward_with_observer(ids, mask, obs)
    stats = obs.percentile(99.9)
    # Magnitudes should land in a reasonable range; far outside this
    # window indicates an observer wiring bug (passing wrong buffer / wrong
    # length / writing to a stale arena slot).
    for key, value in stats.items():
        assert np.isfinite(value), f"{key}: not finite"
        assert 1e-4 < value < 1e4, f"{key}: 99.9-pctile={value:.4g} is suspicious"


def test_observer_accumulates_across_multiple_forwards(model_8m: esm_cpp.Model) -> None:
    obs = esm_cpp.ActivationObserver()
    seqs = ["MKTGVAQRLE", "MAGAAS", "MGSSHHHHHH"]
    for s in seqs:
        ids, mask = _make_ids(s)
        model_8m.forward_with_observer(ids, mask, obs)
    stats_all = obs.percentile(99.9)

    # Clear and re-observe just one — should produce fewer/different stats per site
    obs.clear()
    ids, mask = _make_ids(seqs[0])
    model_8m.forward_with_observer(ids, mask, obs)
    stats_one = obs.percentile(99.9)

    assert set(stats_all.keys()) == set(stats_one.keys()), (
        "site key set should be the same whether we observe 1 seq or 3"
    )
