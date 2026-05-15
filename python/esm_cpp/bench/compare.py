"""Throughput + latency comparison: esm.cpp vs HuggingFace EsmModel eager FP32.

Phase 1 gate (from SPEC §2): >=2x HuggingFace PyTorch eager FP32 throughput
on ESM-2-650M, batch 16, 300aa, single-socket x86 with AVX-512+VNNI.

Usage:
  python -m esm_cpp.bench.compare \\
      --model esm2_t33_650M --batch 16 --len 300 \\
      --warmup 5 --iters 30 --out results.json

The script:
  1. Loads HF EsmForMaskedLM in eager FP32 mode.
  2. Loads esm.cpp Model from the same safetensors file.
  3. Generates a deterministic random batch of sequences of the requested
     length and runs both engines on the same input.
  4. Times warmup-then-iters runs of each, reports p50/p99 latency and
     throughput. Pins MKL / OMP / esm.cpp thread counts so the comparison
     is apples-to-apples.

Designed to be re-runnable on the x86 gate machine; default arguments
match the gate config.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

import esm_cpp


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _safetensors_path(hf_id: str) -> Path:
    snapshots = HF_CACHE / f"models--{hf_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        raise SystemExit(
            f"HF cache missing: {snapshots}. Run "
            f"`huggingface-cli download {hf_id}` first."
        )
    candidates = list(snapshots.glob("*/model.safetensors"))
    if not candidates:
        raise SystemExit(f"no model.safetensors under {snapshots}")
    return candidates[0]


def _make_batch(batch: int, length: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    cls_id = esm_cpp.Tokenizer.cls_id
    eos_id = esm_cpp.Tokenizer.eos_id
    for _ in range(batch):
        payload = rng.integers(low=4, high=29, size=length - 2, dtype=np.int32)
        ids = np.concatenate([[cls_id], payload, [eos_id]]).astype(np.int32)
        out.append(ids)
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def _bench_esm_cpp(model: esm_cpp.Model, ids_list: list[np.ndarray],
                    masks_list: list[np.ndarray], warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        model.forward_batch(ids_list, masks_list)
    timings: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.forward_batch(ids_list, masks_list)
        timings.append(time.perf_counter() - t0)
    batch = len(ids_list)
    return {
        "engine": "esm.cpp",
        "iters": iters,
        "p50_ms": _percentile(timings, 50) * 1000.0,
        "p99_ms": _percentile(timings, 99) * 1000.0,
        "mean_ms": (statistics.mean(timings) * 1000.0) if timings else 0.0,
        "throughput_seqs_per_s": batch / statistics.mean(timings) if timings else 0.0,
        "isa": esm_cpp.current_isa(),
        "host_isa": esm_cpp.host_isa(),
        "threads": esm_cpp.Model.num_threads,
    }


def _bench_hf(hf_id: str, ids_list: list[np.ndarray], warmup: int, iters: int) -> dict[str, Any]:
    import torch
    from transformers import EsmForMaskedLM

    model = EsmForMaskedLM.from_pretrained(hf_id, torch_dtype=torch.float32,
                                            attn_implementation="eager")
    model.eval()
    # Pad to max length so HF can run as a single batched tensor — this is
    # the apples-to-apples comparison the gate measures.
    max_len = max(len(ids) for ids in ids_list)
    pad_id = esm_cpp.Tokenizer.pad_id
    padded = np.full((len(ids_list), max_len), pad_id, dtype=np.int64)
    mask = np.zeros((len(ids_list), max_len), dtype=np.int64)
    for i, ids in enumerate(ids_list):
        padded[i, : len(ids)] = ids
        mask[i, : len(ids)] = 1
    input_ids = torch.from_numpy(padded)
    attention_mask = torch.from_numpy(mask)
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids=input_ids, attention_mask=attention_mask)
        timings: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(input_ids=input_ids, attention_mask=attention_mask)
            timings.append(time.perf_counter() - t0)
    return {
        "engine": "huggingface_eager_fp32",
        "iters": iters,
        "p50_ms": _percentile(timings, 50) * 1000.0,
        "p99_ms": _percentile(timings, 99) * 1000.0,
        "mean_ms": (statistics.mean(timings) * 1000.0) if timings else 0.0,
        "throughput_seqs_per_s": len(ids_list) / statistics.mean(timings) if timings else 0.0,
        "threads_intra_op": int(torch.get_num_threads()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="esm2_t33_650M",
                        choices=["esm2_t6_8M", "esm2_t12_35M", "esm2_t30_150M",
                                  "esm2_t33_650M", "esm2_t36_3B"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--len", dest="length", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the JSON result here (in addition to stdout).")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Run only esm.cpp (useful when PyTorch/HF unavailable).")
    args = parser.parse_args()

    hf_id_map = {
        "esm2_t6_8M":   "facebook/esm2_t6_8M_UR50D",
        "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
        "esm2_t30_150M":"facebook/esm2_t30_150M_UR50D",
        "esm2_t33_650M":"facebook/esm2_t33_650M_UR50D",
        "esm2_t36_3B":  "facebook/esm2_t36_3B_UR50D",
    }
    hf_id = hf_id_map[args.model]
    path = _safetensors_path(hf_id)

    ids_list = _make_batch(args.batch, args.length, args.seed)
    masks_list = [np.ones_like(ids, dtype=np.int32) for ids in ids_list]
    cpp_model = esm_cpp.Model.load_from_safetensors(str(path))

    cpp_result = _bench_esm_cpp(cpp_model, ids_list, masks_list, args.warmup,
                                  args.iters)

    hf_result: dict[str, Any] | None = None
    if not args.skip_hf:
        hf_result = _bench_hf(hf_id, ids_list, args.warmup, args.iters)

    summary: dict[str, Any] = {
        "config": {
            "model": args.model,
            "batch": args.batch,
            "length": args.length,
            "warmup": args.warmup,
            "iters": args.iters,
            "esm_num_threads_env": os.environ.get("ESM_NUM_THREADS"),
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
        },
        "esm_cpp": cpp_result,
        "huggingface": hf_result,
    }
    if hf_result is not None:
        summary["speedup"] = (
            hf_result["mean_ms"] / cpp_result["mean_ms"]
            if cpp_result["mean_ms"] > 0 else None
        )
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
