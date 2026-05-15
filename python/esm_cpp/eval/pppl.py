"""Exact pseudo-perplexity (PPPL) evaluation for ESM-2 via esm.cpp.

Definition (Lin et al. 2023):
  For each protein sequence of length L (excluding <cls> and <eos>):
    for each position p in [1, L-1):
      mask position p, run a forward pass on the masked sequence,
      NLL_p = -log softmax(logits[p])[true_token_at_p]
    NLL_seq = sum_p NLL_p / (L - 2)
  PPPL = exp(mean_over_dataset(NLL_seq))

Each sequence's L masked variants are batched via `Model.forward_batch`
so the L-fold inflation runs concurrently across the thread pool. This
is the "killer demo" for the Phase 3 cu_seqlens scheduler (research-
report §"Continuous-batching scheduler design"); even without that
scheduler, ForwardBatch's per-thread Workspace handles it correctly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

import esm_cpp


def _softmax_logp_at(logits_row: np.ndarray, token: int) -> float:
    """Log-softmax probability of `token` given a [V] logits row."""
    m = float(logits_row.max())
    return float(logits_row[token] - m - np.log(np.exp(logits_row - m).sum()))


def _per_sequence_nll(
    model: esm_cpp.Model,
    tokenizer: esm_cpp.Tokenizer,
    sequence: str,
    mask_token_id: int,
) -> tuple[float, int]:
    """Returns (sum_NLL_over_unmasked_positions, num_unmasked_positions)."""
    ids = np.asarray(tokenizer.encode(sequence), dtype=np.int32)
    L = int(ids.shape[0])
    # Skip <cls> at position 0 and <eos> at position L-1.
    masked_positions = list(range(1, L - 1))
    if not masked_positions:
        return 0.0, 0
    variants: list[np.ndarray] = []
    for p in masked_positions:
        variant = ids.copy()
        variant[p] = mask_token_id
        variants.append(variant)
    masks = [np.ones_like(v) for v in variants]
    logits_list = model.forward_batch(variants, masks)
    total_nll = 0.0
    for p, logits in zip(masked_positions, logits_list):
        # logits has shape [L, vocab_size]; row p is the prediction at
        # the masked position.
        true_token = int(ids[p])
        log_p = _softmax_logp_at(logits[p], true_token)
        total_nll += -log_p
    return total_nll, len(masked_positions)


def pppl(
    model: esm_cpp.Model,
    tokenizer: esm_cpp.Tokenizer,
    sequences: Iterable[str],
) -> float:
    """Compute exact PPPL across a dataset of amino-acid sequences."""
    mask_id = esm_cpp.Tokenizer.mask_id
    total_nll = 0.0
    total_positions = 0
    for s in sequences:
        nll, n = _per_sequence_nll(model, tokenizer, s, mask_id)
        total_nll += nll
        total_positions += n
    if total_positions == 0:
        raise ValueError("dataset produced zero unmasked positions")
    mean_nll = total_nll / total_positions
    return float(np.exp(mean_nll))


def _load_fasta(path: Path) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    with path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            sequences.append("".join(current))
    return sequences


def _safetensors_path_for(hf_id: str) -> Path:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = cache / f"models--{hf_id.replace('/', '--')}" / "snapshots"
    candidates = list(snapshots.glob("*/model.safetensors"))
    if not candidates:
        raise SystemExit(
            f"HF cache miss for {hf_id}; run `huggingface-cli download {hf_id}`"
        )
    return candidates[0]


_HF_ID_FOR_SHORTHAND = {
    "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
    "esm2_t36_3B": "facebook/esm2_t36_3B_UR50D",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="esm2_t6_8M",
                        choices=list(_HF_ID_FOR_SHORTHAND.keys()))
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to a FASTA file with the PPPL holdout.")
    parser.add_argument("--num-seqs", type=int, default=0,
                        help="Limit to the first N sequences (0 = all).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the JSON summary to this path.")
    args = parser.parse_args()

    hf_id = _HF_ID_FOR_SHORTHAND[args.model]
    path = _safetensors_path_for(hf_id)
    model = esm_cpp.Model.load_from_safetensors(str(path))
    tokenizer = esm_cpp.Tokenizer()
    sequences = _load_fasta(args.data)
    if args.num_seqs > 0:
        sequences = sequences[: args.num_seqs]
    value = pppl(model, tokenizer, sequences)
    summary = {
        "model": args.model,
        "data": str(args.data),
        "num_sequences": len(sequences),
        "isa": esm_cpp.current_isa(),
        "pppl": value,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
