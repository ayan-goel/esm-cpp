"""Phase 13 T2: time coremltools.predict on a compiled whole-graph mlmodelc.

This is one of the three numbers needed for the GO/NO-GO gate. It's an upper
bound — coremltools predict carries Python overhead the C++ bridge avoids.
The point is to establish "how fast can the same artifact go in *any* runtime,"
so the C++ bridge has a target.

Run with the convert-time env:
    /tmp/ct312/bin/python tools/bench_whole_graph_coreml_predict.py \\
        --mlpackage weights/esm2_8m.whole-graph/B-8_L-256/whole_graph.mlpackage \\
        --batch 8 --seq-len 256
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import coremltools as ct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlpackage", type=Path, required=True,
                    help="path to .mlpackage (built by build_whole_graph_artifacts.py)")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument(
        "--compute-units", choices=("CPU_ONLY", "CPU_AND_NE", "ALL"),
        default="CPU_AND_NE")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    compute_units = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }[args.compute_units]

    print(f"[load] {args.mlpackage}")
    t = time.perf_counter()
    m = ct.models.MLModel(str(args.mlpackage), compute_units=compute_units)
    print(f"  loaded in {time.perf_counter()-t:.2f}s")

    rng = np.random.default_rng(args.seed)
    # Tokens 4..23 are amino-acid IDs in ESM-2 tokenization. Doesn't matter
    # which ones we pick for a bench.
    ids = rng.integers(4, 24, size=(args.batch, args.seq_len), dtype=np.int32)
    mask = np.ones((args.batch, args.seq_len), dtype=np.int32)

    print(f"[predict] B={args.batch} L={args.seq_len} compute_units={args.compute_units}")
    # Warmup
    for _ in range(args.warmup):
        m.predict({"input_ids": ids, "attention_mask": mask})

    ts = []
    for _ in range(args.iters):
        t = time.perf_counter()
        m.predict({"input_ids": ids, "attention_mask": mask})
        ts.append(time.perf_counter() - t)

    ts = sorted(ts)
    p50 = ts[len(ts) // 2] * 1000
    p10 = ts[max(0, len(ts) // 10)] * 1000
    p90 = ts[min(len(ts) - 1, (9 * len(ts)) // 10)] * 1000
    print(f"  p10={p10:.1f}  p50={p50:.1f}  p90={p90:.1f}  ms  (over {args.iters} iters)")
    print(f"BENCH: coremltools_predict p50_ms={p50:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
