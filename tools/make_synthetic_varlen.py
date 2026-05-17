"""Generate an OAS-length-distribution-matched synthetic FASTA.

The variable-length benchmark measures the cu_seqlens-packed-batch
path's advantage over HuggingFace's padded-batch path: HF pads every
sequence in the batch to max(len), wasting attention compute on pad
tokens, while esm-cpp processes the actual lengths back-to-back. That
advantage scales with the length distribution — narrow distributions
give little win, OAS-like (mean ~120, max ~250) gives a 2-3× compute
saving on top of the architecture's baseline speedup.

For benchmarking the perf characteristic is fully determined by the
length distribution; the amino acid content doesn't affect ms/forward.
This script emits a deterministic FASTA with lengths sampled from a
mixture that matches published OAS heavy-chain statistics:
  - 95 %: ~Gaussian(μ=120, σ=12), clipped to [80, 180] — the canonical
    IGHV+IGHD+IGHJ recombined heavy chain.
  - 5 %: uniform [200, 250] — IgE / extreme-CDR3 / IgM tail.

Content is uniform random over the canonical 20 amino acids. For
benchmarks using REAL OAS sequences (recommended for any external
quality claim), see `tools/fetch_oas_antibodies.py` which consumes a
locally-downloaded OAS CSV.

Output is a FASTA plus a `.manifest.json` recording the generation
parameters and SHA-256, matching `fetch_oas_antibodies.py`'s format.

Usage:
  python tools/make_synthetic_varlen.py \\
      --num-seqs 256 \\
      --out benchmarks/data/synthetic_varlen_v1.fasta
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def sample_length(rng: random.Random) -> int:
    """OAS-heavy-chain-matched length mixture.

    Body: 95% Gaussian(μ=120, σ=12), clipped to [80, 180].
    Tail: 5% Uniform[200, 250] for IgE / extreme-CDR3 variants.
    """
    if rng.random() < 0.05:
        return rng.randint(200, 250)
    raw = round(rng.gauss(120, 12))
    return max(80, min(180, raw))


def sample_seq(rng: random.Random, length: int) -> str:
    # Uniform over canonical 20 — content doesn't matter for the perf
    # characteristic, only length does. ESM-2 tokenization handles any
    # valid amino acid identically (one token per residue).
    return "".join(rng.choice(_ALPHABET) for _ in range(length))


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-seqs", type=int, default=256,
                        help="Number of sequences to emit (default: 256).")
    parser.add_argument("--seed", type=int, default=20260516,
                        help="RNG seed for reproducibility.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output FASTA path.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Manifest JSON path; defaults to <out>.manifest.json.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    lengths: list[int] = []
    seqs: list[str] = []
    for _ in range(args.num_seqs):
        length = sample_length(rng)
        lengths.append(length)
        seqs.append(sample_seq(rng, length))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i, s in enumerate(seqs):
            f.write(f">seq_{i:05d} len={len(s)}\n{s}\n")

    manifest_path = args.manifest or args.out.with_suffix(
        args.out.suffix + ".manifest.json")
    manifest = {
        "source": "synthetic (OAS-length-distribution-matched)",
        "generator": "tools/make_synthetic_varlen.py",
        "snapshot_date": datetime.now(timezone.utc).date().isoformat(),
        "num_sequences": args.num_seqs,
        "seed": args.seed,
        "fasta_sha256": _sha256_of(args.out),
        "length_distribution": {
            "model": "0.95 * Gaussian(120, 12) clip[80, 180] + "
                     "0.05 * Uniform[200, 250]",
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
            "median": sorted(lengths)[len(lengths) // 2],
        },
        "alphabet": _ALPHABET,
        "note": (
            "Synthetic content for perf benchmarking only — measures "
            "cu_seqlens-packed-batch vs HF-padded-batch. For external "
            "quality claims, use real OAS sequences via "
            "tools/fetch_oas_antibodies.py."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(seqs)} sequences to {args.out} "
          f"(mean={manifest['length_distribution']['mean']}, "
          f"min={min(lengths)}, max={max(lengths)})")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
