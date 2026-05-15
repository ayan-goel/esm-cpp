"""Calibration + quantization driver.

Reads a FASTA calibration corpus, runs forward_with_observer on each
sequence, dumps the resulting 99.9-percentile stats to JSON. Slice 4
consumes this JSON in the SmoothQuant migration pass.

Usage:
  python -m esm_cpp.quantize --calibrate \\
      --model esm2_t12_35M \\
      --calib data/uniref50_calib_v1.fasta \\
      --out weights/esm2_t12_35M_calib.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import esm_cpp
from esm_cpp.eval.pppl import _HF_ID_FOR_SHORTHAND, _load_fasta, _safetensors_path_for


def calibrate(
    model: esm_cpp.Model,
    tokenizer: esm_cpp.Tokenizer,
    sequences: list[str],
    percentile: float = 99.9,
) -> dict[str, float]:
    """Run forward_with_observer over every sequence; return per-site percentile."""
    observer = esm_cpp.ActivationObserver()
    for s in sequences:
        ids = np.asarray(tokenizer.encode(s), dtype=np.int32)
        mask = np.ones_like(ids, dtype=np.int32)
        model.forward_with_observer(ids, mask, observer)
    return observer.percentile(percentile)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration and dump activation stats.")
    parser.add_argument("--model", default="esm2_t6_8M",
                        choices=list(_HF_ID_FOR_SHORTHAND.keys()))
    parser.add_argument("--calib", type=Path, required=True,
                        help="FASTA file with the calibration corpus.")
    parser.add_argument("--num-calib", type=int, default=0,
                        help="Limit to the first N sequences (0 = all).")
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not args.calibrate:
        raise SystemExit("only --calibrate is implemented in Slice 3; "
                         "--apply-smoothquant arrives in Slice 4.")

    hf_id = _HF_ID_FOR_SHORTHAND[args.model]
    path = _safetensors_path_for(hf_id)
    model = esm_cpp.Model.load_from_safetensors(str(path))
    tokenizer = esm_cpp.Tokenizer()
    sequences = _load_fasta(args.calib)
    if args.num_calib > 0:
        sequences = sequences[: args.num_calib]
    stats = calibrate(model, tokenizer, sequences, percentile=args.percentile)
    summary = {
        "model": args.model,
        "calib": str(args.calib),
        "num_sequences": len(sequences),
        "percentile": args.percentile,
        "isa": esm_cpp.current_isa(),
        "stats": stats,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(stats)} site stats to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
