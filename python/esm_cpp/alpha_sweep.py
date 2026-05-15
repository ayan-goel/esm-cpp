"""SmoothQuant α-sweep + PPPL drift driver.

For each α in the sweep:
  1. Load fresh FP32 weights.
  2. Apply SmoothQuant at this α using the calibration stats.
  3. Quantize per-layer Linear weights to INT8.
  4. Run PPPL on a small subset and report drift vs the FP32 baseline.

Reports a per-α table and picks the α with the lowest |PPPL drift|. If
the best α still exceeds the PPPL gate, the operator can flip on the
first-block FP16 escape via Model.set_first_block_fc1_fp16(True) and
re-sweep.

Usage:
  python -m esm_cpp.alpha_sweep \\
      --model esm2_t12_35M \\
      --calib data/uniref50_calib.fasta --num-calib 256 \\
      --pppl-data data/uniref50_pppl_small.fasta --num-pppl 100 \\
      --alphas 0.3,0.4,0.5,0.6,0.7 \\
      --out notes/phase2-alpha-sweep-35m.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import esm_cpp
from esm_cpp.eval.pppl import (_HF_ID_FOR_SHORTHAND, _load_fasta,
                               _safetensors_path_for, pppl)
from esm_cpp.quantize import calibrate


def _baseline_pppl(model_path: Path, sequences: list[str]) -> float:
    model = esm_cpp.Model.load_from_safetensors(str(model_path))
    return pppl(model, esm_cpp.Tokenizer(), sequences)


def _pppl_at_alpha(model_path: Path, alpha: float,
                    calib_stats: dict[str, float],
                    sequences: list[str],
                    first_block_fp16: bool) -> float:
    model = esm_cpp.Model.load_from_safetensors(str(model_path))
    model.apply_smoothquant(calib_stats, alpha)
    model.quantize_weights()
    if first_block_fp16:
        model.set_first_block_fc1_fp16(True)
    return pppl(model, esm_cpp.Tokenizer(), sequences)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="esm2_t6_8M",
                        choices=list(_HF_ID_FOR_SHORTHAND.keys()))
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--num-calib", type=int, default=0)
    parser.add_argument("--pppl-data", type=Path, required=True)
    parser.add_argument("--num-pppl", type=int, default=0)
    parser.add_argument("--alphas", type=str, default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--first-block-fp16", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    hf_id = _HF_ID_FOR_SHORTHAND[args.model]
    model_path = _safetensors_path_for(hf_id)

    calib_seqs = _load_fasta(args.calib)
    if args.num_calib > 0:
        calib_seqs = calib_seqs[: args.num_calib]
    pppl_seqs = _load_fasta(args.pppl_data)
    if args.num_pppl > 0:
        pppl_seqs = pppl_seqs[: args.num_pppl]

    # Calibrate once; reuse stats across α values (they don't depend on α).
    calib_model = esm_cpp.Model.load_from_safetensors(str(model_path))
    calib_stats = calibrate(calib_model, esm_cpp.Tokenizer(), calib_seqs)

    baseline = _baseline_pppl(model_path, pppl_seqs)

    alphas = [float(s) for s in args.alphas.split(",")]
    rows: list[dict[str, float]] = []
    for alpha in alphas:
        p_int8 = _pppl_at_alpha(model_path, alpha, calib_stats, pppl_seqs,
                                 args.first_block_fp16)
        rows.append({
            "alpha": alpha,
            "pppl_fp32": baseline,
            "pppl_int8": p_int8,
            "drift": abs(p_int8 - baseline),
        })

    best = min(rows, key=lambda r: r["drift"])
    summary = {
        "model": args.model,
        "first_block_fp16": args.first_block_fp16,
        "num_calib": len(calib_seqs),
        "num_pppl": len(pppl_seqs),
        "isa": esm_cpp.current_isa(),
        "rows": rows,
        "best_alpha": best["alpha"],
        "best_drift": best["drift"],
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
