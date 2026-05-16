"""Calibration + quantization driver.

Two modes:
  --calibrate         Run forward_with_observer on a FASTA corpus and
                      dump 99.9-percentile activation stats to JSON.
  --apply-smoothquant Load FP32 weights, apply SmoothQuant migration,
                      quantize per-layer Linears to INT8, write a
                      quantized GGUF artifact.

Examples:
  python -m esm_cpp.quantize --calibrate \\
      --model esm2_t12_35M \\
      --calib data/uniref50_calib_v1.fasta \\
      --out weights/esm2_t12_35M_calib.json

  python -m esm_cpp.quantize --apply-smoothquant \\
      --model esm2_t12_35M \\
      --stats weights/esm2_t12_35M_calib.json \\
      --alpha 0.5 \\
      --output weights/esm2_t12_35M_q8.gguf
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


def _run_calibrate(args: argparse.Namespace) -> int:
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


def _run_apply_smoothquant(args: argparse.Namespace) -> int:
    hf_id = _HF_ID_FOR_SHORTHAND[args.model]
    src = _safetensors_path_for(hf_id)
    model = esm_cpp.Model.load_from_safetensors(str(src))
    stats_json = json.loads(args.stats.read_text())
    stats = stats_json["stats"]
    if not isinstance(stats, dict):
        raise SystemExit(f"--stats file missing 'stats' dict: {args.stats}")
    model.apply_smoothquant(stats, args.alpha)
    model.quantize_weights()
    if args.first_block_fp16:
        model.set_first_block_fc1_fp16(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_to_gguf(str(args.output))
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(
        f"Wrote {args.output} ({size_mb:.1f} MB) — alpha={args.alpha} "
        f"first_block_fp16={args.first_block_fp16}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    # --calibrate / --apply-smoothquant kept as legacy top-level flags
    # so existing CI driver scripts don't need to change.
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration and dump activation stats.")
    parser.add_argument("--apply-smoothquant", action="store_true",
                        help="Apply SmoothQuant + quantize + write GGUF.")
    parser.add_argument("--model", default="esm2_t6_8M",
                        choices=list(_HF_ID_FOR_SHORTHAND.keys()))

    # Calibration args
    parser.add_argument("--calib", type=Path,
                        help="FASTA file with the calibration corpus.")
    parser.add_argument("--num-calib", type=int, default=0,
                        help="Limit to the first N sequences (0 = all).")
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--out", type=Path,
                        help="Output JSON path for --calibrate.")

    # SmoothQuant args
    parser.add_argument("--stats", type=Path,
                        help="JSON stats file from --calibrate.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SmoothQuant alpha (default 0.5).")
    parser.add_argument("--first-block-fp16", action="store_true",
                        help="Enable the layer-0 fc1 FP16 escape.")
    parser.add_argument("--output", type=Path,
                        help="Output GGUF path for --apply-smoothquant.")

    args = parser.parse_args()
    if args.calibrate and args.apply_smoothquant:
        raise SystemExit("--calibrate and --apply-smoothquant are mutually exclusive")
    if args.calibrate:
        if not args.calib or not args.out:
            raise SystemExit("--calibrate requires --calib and --out")
        return _run_calibrate(args)
    if args.apply_smoothquant:
        if not args.stats or not args.output:
            raise SystemExit("--apply-smoothquant requires --stats and --output")
        return _run_apply_smoothquant(args)
    raise SystemExit("pick --calibrate or --apply-smoothquant")


if __name__ == "__main__":
    raise SystemExit(main())
