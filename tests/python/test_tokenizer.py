"""Byte-exact parity tests for esm.cpp's Tokenizer against HF EsmTokenizer.

Phase 0, Slice 2 acceptance criterion: 10,000 random sequences tokenize
byte-exact to HuggingFace `facebook/esm2_t6_8M_UR50D`.
"""

from __future__ import annotations

import random

import pytest

import esm_cpp

# HF is optional at test time; skip the parity test if it isn't installed.
try:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    HF_AVAILABLE = False


CANONICAL_AA = "LAGVSERTIDPKQNFYMHWC"  # 20 amino acids, UR50 order
RARE_AA = "XBUZO.-"
NOISE_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^&*"


@pytest.fixture(scope="module")
def hf_tokenizer():
    if not HF_AVAILABLE:
        pytest.skip("transformers not installed")
    return AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


@pytest.fixture
def ours():
    return esm_cpp.Tokenizer()


def _random_sequence(rng: random.Random, alphabet: str, lo: int, hi: int) -> str:
    length = rng.randint(lo, hi)
    return "".join(rng.choices(alphabet, k=length))


def test_vocab_constants(ours):
    assert esm_cpp.Tokenizer.vocab_size == 33
    assert esm_cpp.Tokenizer.cls_id == 0
    assert esm_cpp.Tokenizer.pad_id == 1
    assert esm_cpp.Tokenizer.eos_id == 2
    assert esm_cpp.Tokenizer.unk_id == 3
    assert esm_cpp.Tokenizer.mask_id == 32
    assert ours.id_to_token(20) == "M"
    assert ours.token_to_id("M") == 20


@pytest.mark.parametrize("seq,expected", [
    ("MKTGV", [0, 20, 15, 11, 6, 7, 2]),
    ("XBUZ", [0, 24, 25, 26, 27, 2]),
    ("MKTGVBUZO.X-", [0, 20, 15, 11, 6, 7, 25, 26, 27, 28, 29, 24, 30, 2]),
    ("M K T G V", [0, 20, 15, 11, 6, 7, 2]),
    ("mktgv", [0, 3, 2]),
    ("<mask>", [0, 32, 2]),
    ("M<mask>K", [0, 20, 32, 15, 2]),
    ("<null_1>", [0, 31, 2]),
])
def test_known_cases(ours, seq, expected):
    assert ours.encode(seq) == expected


def test_decode_round_trip(ours):
    seq = "MKTGVAQERSILDPQNFYMHWC"
    ids = ours.encode(seq, add_special=False)
    assert ours.decode(ids) == seq


def test_truncation_keeps_special_tokens(ours):
    seq = "M" * 2000
    ids = ours.encode(seq, add_special=True, truncate=True)
    assert len(ids) == esm_cpp.Tokenizer.model_max_length
    assert ids[0] == 0
    assert ids[-1] == 2


@pytest.mark.parametrize("num_seqs", [10_000])
def test_byte_exact_parity_against_hf(ours, hf_tokenizer, num_seqs):
    """The slice-level acceptance criterion."""
    rng = random.Random(0)
    alphabets = [
        CANONICAL_AA,                              # majority: canonical only
        CANONICAL_AA + RARE_AA,                    # canonical + rare
        CANONICAL_AA + RARE_AA + NOISE_CHARS,      # everything: stress unk/whitespace
    ]
    weights = [0.6, 0.3, 0.1]
    mismatches: list[tuple[str, list[int], list[int]]] = []
    for _ in range(num_seqs):
        alphabet = rng.choices(alphabets, weights=weights, k=1)[0]
        seq = _random_sequence(rng, alphabet, 1, 200)
        # truncate=False to match HF's default (no truncation).
        ours_ids = ours.encode(seq, add_special=True, truncate=False)
        hf_ids = hf_tokenizer(seq, add_special_tokens=True)["input_ids"]
        if ours_ids != hf_ids:
            mismatches.append((seq, ours_ids, hf_ids))
            if len(mismatches) > 5:
                break
    if mismatches:
        details = "\n".join(
            f"  seq={s!r}\n    ours={o}\n    hf  ={h}" for s, o, h in mismatches[:5]
        )
        pytest.fail(f"{len(mismatches)} mismatches in {num_seqs} sequences:\n{details}")
