"""Fetch a fixed OAS (Observed Antibody Space) sample for benchmarking.

The Observed Antibody Space (https://opig.stats.ox.ac.uk/webapps/oas/)
publishes paired heavy+light antibody sequences from multiple studies.
This script downloads a deterministic sample, deduplicates, filters by
length, and writes a FASTA + manifest pinning the source URLs, snapshot
date, and SHA-256 of the output. The harness is OAS-shaped but accepts
any FASTA — operators can substitute their own corpus.

Usage:
  python tools/fetch_oas_antibodies.py \\
      --num-seqs 1000 \\
      --out data/oas_sample_v1.fasta

The script is intentionally conservative: it expects an OAS study URL
(or a local CSV with `cdr3 / sequence_aa` columns) and does not auto-
discover URLs. For the public benchmark sample, the manifest in
docs/benchmarks.md records the exact OAS study/version used.

Hand-off note (Phase 3 Slice 6.1): the gate-machine measurement is
expected to use an OAS sample from a specific 2024–2025 OAS study;
this script provides the deterministic-filter side, while the URL is
operator-specified at run time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable


def _load_csv_column(path: Path, column: str) -> list[str]:
    seqs: list[str] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise SystemExit(
                f"--input {path} must be CSV with a `{column}` column. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            value = (row.get(column) or "").strip()
            if value:
                seqs.append(value)
    return seqs


def _is_valid_protein(seq: str) -> bool:
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    return bool(seq) and all(c in valid for c in seq)


def _filter_and_sample(seqs: Iterable[str], min_len: int, max_len: int,
                        num_seqs: int, seed: int) -> list[str]:
    pool: list[str] = []
    seen: set[str] = set()
    for s in seqs:
        s = s.upper()
        if not _is_valid_protein(s):
            continue
        if not (min_len <= len(s) <= max_len):
            continue
        if s in seen:
            continue
        seen.add(s)
        pool.append(s)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:num_seqs]


def _write_fasta(seqs: list[str], out: Path) -> None:
    with out.open("w") as f:
        for i, s in enumerate(seqs):
            f.write(f">seq_{i:05d}\n{s}\n")


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV containing antibody sequences "
                              "(typically downloaded from OAS).")
    parser.add_argument("--column", default="sequence_aa",
                        help="CSV column containing the amino-acid sequence.")
    parser.add_argument("--num-seqs", type=int, default=1000,
                        help="Number of sequences to retain in the sample.")
    parser.add_argument("--min-len", type=int, default=80)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the deterministic shuffle.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output FASTA path.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Where to write the manifest JSON. Defaults to "
                              "<out>.manifest.json next to the FASTA.")
    args = parser.parse_args()

    seqs = _load_csv_column(args.input, args.column)
    sample = _filter_and_sample(seqs, args.min_len, args.max_len,
                                 args.num_seqs, args.seed)
    if not sample:
        raise SystemExit("no sequences passed filters")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_fasta(sample, args.out)
    manifest_path = args.manifest or args.out.with_suffix(args.out.suffix +
                                                            ".manifest.json")
    manifest = {
        "source_csv": str(args.input),
        "source_csv_sha256": _sha256_of(args.input),
        "snapshot_date": datetime.utcnow().date().isoformat(),
        "num_sequences": len(sample),
        "min_len": args.min_len,
        "max_len": args.max_len,
        "seed": args.seed,
        "fasta_sha256": _sha256_of(args.out),
        "citation": (
            "Olsen et al., 'Observed Antibody Space: A diverse database of "
            "cleaned, annotated and translated unpaired and paired antibody "
            "sequences', Protein Sci. 2022; doi:10.1002/pro.4205"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(sample)} sequences to {args.out}; "
           f"manifest at {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
