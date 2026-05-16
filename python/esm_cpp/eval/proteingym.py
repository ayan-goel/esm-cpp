"""ProteinGym v1.3 zero-shot masked-marginal scoring.

For each (wild-type, variant) pair in a DMS assay:
  for each position p where variant[p] != wt[p]:
    mask position p, forward the masked WT, read logits[p]
    log_p_wt   += log softmax(logits[p])[wt[p]]
    log_p_mut  += log softmax(logits[p])[variant[p]]
  variant_score = log_p_mut - log_p_wt
Spearman(variant_score, experimental_fitness) is the assay's metric;
mean across the 217 substitution assays is the gate.

A wt forward at every distinct masked position is cached so single-,
double-, and triple-mutants on the same wt share masked passes.

Full data: clone OATML-Markslab/ProteinGym at tag PG_v1.3 via
tools/fetch_proteingym.py. This module accepts an arbitrary directory
of assay CSVs so any subset can be evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import esm_cpp


@dataclass(frozen=True)
class Variant:
    """One DMS variant of the wt sequence and its experimental fitness."""
    sequence: str
    fitness: float

    def diffs_from(self, wt: str) -> list[tuple[int, str, str]]:
        if len(self.sequence) != len(wt):
            raise ValueError("variant and wt must have equal length")
        return [(i, wt[i], self.sequence[i])
                for i in range(len(wt)) if self.sequence[i] != wt[i]]


def _log_softmax_at(logits_row: np.ndarray, token: int) -> float:
    m = float(logits_row.max())
    return float(logits_row[token] - m - np.log(np.exp(logits_row - m).sum()))


def score_assay(
    model: esm_cpp.Model,
    tokenizer: esm_cpp.Tokenizer,
    wt_sequence: str,
    variants: Iterable[Variant],
) -> tuple[list[float], list[float]]:
    """Returns (scores, fitnesses) aligned by index for Spearman input."""
    variants = list(variants)
    if not variants:
        return [], []
    # Discover all mutation positions across the assay so we can run each
    # masked WT forward exactly once and reuse logits across variants.
    masked_positions: set[int] = set()
    for v in variants:
        for p, _, _ in v.diffs_from(wt_sequence):
            masked_positions.add(p)
    pos_list = sorted(masked_positions)

    wt_ids = np.asarray(tokenizer.encode(wt_sequence), dtype=np.int32)
    L = int(wt_ids.shape[0])
    mask_id = esm_cpp.Tokenizer.mask_id

    # Build one masked variant per (zero-indexed) protein position; the
    # tokenizer prepends <cls> so the model index is p + 1.
    masked_inputs: list[np.ndarray] = []
    for p in pos_list:
        variant_ids = wt_ids.copy()
        variant_ids[p + 1] = mask_id
        masked_inputs.append(variant_ids)
    # All variants share the WT length, so cu_seqlens packs them densely
    # in a single forward.
    logits_list = model.forward_scheduled(masked_inputs)
    # Map: protein position -> [V] row at that position.
    row_at: dict[int, np.ndarray] = {}
    for p, logits in zip(pos_list, logits_list):
        row_at[p] = logits[p + 1]

    scores: list[float] = []
    fitnesses: list[float] = []
    for v in variants:
        s = 0.0
        for p, wt_aa, mut_aa in v.diffs_from(wt_sequence):
            wt_tok = tokenizer.token_to_id(wt_aa)
            mut_tok = tokenizer.token_to_id(mut_aa)
            row = row_at[p]
            s += _log_softmax_at(row, mut_tok) - _log_softmax_at(row, wt_tok)
        scores.append(s)
        fitnesses.append(v.fitness)
    return scores, fitnesses


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank-order correlation."""
    if len(xs) < 2:
        return 0.0
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    n = len(xs)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    denom_x = math.sqrt(sum((r - mean_rx) ** 2 for r in rx))
    denom_y = math.sqrt(sum((r - mean_ry) ** 2 for r in ry))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return num / (denom_x * denom_y)


def _rankdata(xs: list[float]) -> list[float]:
    """Average-rank fractional ranking (matches scipy.stats.rankdata)."""
    indexed = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and xs[indexed[j + 1]] == xs[indexed[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-indexed average
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _load_assay_csv(path: Path) -> tuple[str, list[Variant]]:
    """Reads a ProteinGym-style assay CSV. Expected columns:
       mutated_sequence (str), DMS_score (float)
    plus a `target_seq` column on at least one row (or a sibling .json
    manifest). For the synthetic tests we lay rows out the same way.
    """
    rows = list(csv.DictReader(path.open("r")))
    if not rows:
        raise SystemExit(f"empty assay file {path}")
    # The wt is whatever row matches "mutant" == "WT" (PG convention),
    # falling back to the longest common subsequence approach if absent.
    wt_row = next((r for r in rows if r.get("mutant", "").upper() == "WT"), None)
    if wt_row is None:
        wt_row = rows[0]
    wt = wt_row["mutated_sequence"]
    variants: list[Variant] = []
    for r in rows:
        seq = r["mutated_sequence"]
        if r.get("mutant", "").upper() == "WT":
            continue
        try:
            fitness = float(r["DMS_score"])
        except (KeyError, ValueError):
            continue
        variants.append(Variant(sequence=seq, fitness=fitness))
    return wt, variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="esm2_t6_8M")
    parser.add_argument("--assays", type=Path, required=True,
                        help="Directory containing ProteinGym-format assay CSVs.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to the first N assays alphabetically (0 = all).")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from esm_cpp.eval.pppl import _HF_ID_FOR_SHORTHAND, _safetensors_path_for
    hf_id = _HF_ID_FOR_SHORTHAND[args.model]
    path = _safetensors_path_for(hf_id)
    model = esm_cpp.Model.load_from_safetensors(str(path))
    tokenizer = esm_cpp.Tokenizer()

    assay_paths = sorted(args.assays.glob("*.csv"))
    if args.limit > 0:
        assay_paths = assay_paths[: args.limit]
    per_assay: list[dict[str, object]] = []
    for ap in assay_paths:
        wt, variants = _load_assay_csv(ap)
        scores, fitnesses = score_assay(model, tokenizer, wt, variants)
        rho = spearman(scores, fitnesses)
        per_assay.append({"assay": ap.stem, "n": len(variants), "spearman": rho})
    mean_rho = (sum(a["spearman"] for a in per_assay) / len(per_assay)
                if per_assay else 0.0)
    summary = {
        "model": args.model,
        "isa": esm_cpp.current_isa(),
        "num_assays": len(per_assay),
        "mean_spearman": mean_rho,
        "assays": per_assay,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
