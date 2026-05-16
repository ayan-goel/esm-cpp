"""Public benchmark: esm.cpp vs HuggingFace EsmForMaskedLM eager FP32.

Reports throughput (seqs/sec), p50/p99 latency, and a hardware block
(CPU model, core count, ISA, RAM). Per the Phase 3 ship gate, the
headline measurement is x86 AVX-512+VNNI on real OAS antibody data;
this harness runs anywhere the wheel installs.

Usage:
  python -m esm_cpp.bench.compare \\
      --model weights/esm2_8m.gguf \\
      --dataset data/oas_sample_v1.fasta \\
      --modes esm-cpp-fp32,hf-eager-fp32 \\
      --output results/dev_host_8m.json

Modes:
  esm-cpp-fp32   FP32 weights via Model.load_from_safetensors / load
  esm-cpp-int8   Quantized weights via Model.load_from_gguf
                 (the GGUF must have been written with quantized state)
  hf-eager-fp32  HuggingFace EsmForMaskedLM, attn_implementation="eager"

Without --dataset, synthesizes a deterministic random batch matching
--batch / --len. With --dataset, reads FASTA and uses the actual
sequences (real OAS sequences are the headline workload).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import esm_cpp


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

_HF_ID_FOR_SHORTHAND = {
    "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
    "esm2_t36_3B": "facebook/esm2_t36_3B_UR50D",
}


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


def _resolve_model_source(model: str) -> dict[str, str]:
    """Returns {'kind': 'gguf'|'safetensors', 'path': ..., 'hf_id': ...}."""
    p = Path(model)
    if p.is_file() and p.suffix == ".gguf":
        return {"kind": "gguf", "path": str(p), "hf_id": ""}
    if p.is_file():
        return {"kind": "safetensors", "path": str(p), "hf_id": ""}
    if model in _HF_ID_FOR_SHORTHAND:
        hf_id = _HF_ID_FOR_SHORTHAND[model]
    else:
        hf_id = model
    return {"kind": "safetensors", "path": str(_safetensors_path(hf_id)),
            "hf_id": hf_id}


def _load_fasta(path: Path) -> list[str]:
    out, cur = [], []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    out.append("".join(cur))
                    cur = []
            else:
                cur.append(line)
        if cur:
            out.append("".join(cur))
    return out


def _encode_batch(tokenizer: esm_cpp.Tokenizer,
                   sequences: Iterable[str]) -> list[np.ndarray]:
    return [np.asarray(tokenizer.encode(s), dtype=np.int32) for s in sequences]


def _synthetic_batch(batch: int, length: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    cls_id = esm_cpp.Tokenizer.cls_id
    eos_id = esm_cpp.Tokenizer.eos_id
    for _ in range(batch):
        payload = rng.integers(low=4, high=29, size=length - 2, dtype=np.int32)
        out.append(np.concatenate([[cls_id], payload, [eos_id]]).astype(np.int32))
    return out


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _hardware_block() -> dict[str, Any]:
    """CPU model, core count, ISA flags, RAM. Best-effort across OSes."""
    out: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "esm_cpp_host_isa": esm_cpp.host_isa(),
        "esm_cpp_current_isa": esm_cpp.current_isa(),
        "esm_cpp_num_threads": esm_cpp.Model.num_threads,
    }
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        out["cpu_model"] = line.split(":", 1)[1].strip()
                        break
                f.seek(0)
                cores = sum(1 for line in f if line.startswith("processor"))
                if cores:
                    out["cpu_threads"] = cores
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kib = int(line.split()[1])
                            out["ram_gb"] = round(kib / (1024 * 1024), 1)
                            break
            except Exception:
                pass
        elif platform.system() == "Darwin":
            try:
                out["cpu_model"] = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True).strip()
            except Exception:
                pass
            try:
                out["cpu_threads"] = int(subprocess.check_output(
                    ["sysctl", "-n", "hw.logicalcpu"], text=True).strip())
            except Exception:
                pass
            try:
                ram_bytes = int(subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True).strip())
                out["ram_gb"] = round(ram_bytes / (1024 ** 3), 1)
            except Exception:
                pass
    except Exception:
        pass
    return out


def _bench_esm_cpp(model: esm_cpp.Model,
                    ids_list: list[np.ndarray],
                    warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        model.forward_scheduled(ids_list)
    timings: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.forward_scheduled(ids_list)
        timings.append(time.perf_counter() - t0)
    batch = len(ids_list)
    mean = statistics.mean(timings) if timings else 0.0
    return {
        "iters": iters,
        "p50_ms": _percentile(timings, 50) * 1000.0,
        "p99_ms": _percentile(timings, 99) * 1000.0,
        "mean_ms": mean * 1000.0,
        "throughput_seqs_per_s": (batch / mean) if mean > 0 else 0.0,
        "isa": esm_cpp.current_isa(),
    }


def _bench_hf(hf_id: str, ids_list: list[np.ndarray],
               warmup: int, iters: int) -> dict[str, Any]:
    import torch
    from transformers import EsmForMaskedLM
    model = EsmForMaskedLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
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
    mean = statistics.mean(timings) if timings else 0.0
    return {
        "iters": iters,
        "p50_ms": _percentile(timings, 50) * 1000.0,
        "p99_ms": _percentile(timings, 99) * 1000.0,
        "mean_ms": mean * 1000.0,
        "throughput_seqs_per_s": (len(ids_list) / mean) if mean > 0 else 0.0,
        "torch_intra_threads": int(torch.get_num_threads()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="esm2_t33_650M",
                        help="HF id, shorthand (esm2_t6_8M, ...), or path "
                              "to a .safetensors / .gguf file.")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="FASTA file with the benchmark sequences. "
                              "If omitted, generates a synthetic uniform-"
                              "length batch (--batch x --len).")
    parser.add_argument("--modes", default="esm-cpp-fp32,hf-eager-fp32",
                        help="Comma-separated subset of "
                              "{esm-cpp-fp32, esm-cpp-int8, hf-eager-fp32}.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--len", dest="length", type=int, default=300)
    parser.add_argument("--num-seqs", type=int, default=0,
                        help="Limit FASTA to first N sequences (0 = all).")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the results.json here (in addition "
                              "to stdout).")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid = {"esm-cpp-fp32", "esm-cpp-int8", "hf-eager-fp32"}
    for m in modes:
        if m not in valid:
            raise SystemExit(f"unknown mode: {m}; valid = {sorted(valid)}")

    tokenizer = esm_cpp.Tokenizer()
    if args.dataset:
        seqs = _load_fasta(args.dataset)
        if args.num_seqs > 0:
            seqs = seqs[: args.num_seqs]
        ids_list = _encode_batch(tokenizer, seqs)
        dataset_label = str(args.dataset)
        n_seqs = len(ids_list)
    else:
        ids_list = _synthetic_batch(args.batch, args.length, args.seed)
        dataset_label = (f"synthetic(seed={args.seed}, batch={args.batch}, "
                          f"len={args.length})")
        n_seqs = args.batch

    src = _resolve_model_source(args.model)

    results: dict[str, dict[str, Any]] = {}
    if "esm-cpp-fp32" in modes or "esm-cpp-int8" in modes:
        if src["kind"] == "gguf":
            model = esm_cpp.Model.load_from_gguf(src["path"])
        else:
            model = esm_cpp.Model.load_from_safetensors(src["path"])
        if "esm-cpp-fp32" in modes:
            if model.config.weights_quantized:
                # Re-load FP32 for the FP32 mode.
                fp32 = esm_cpp.Model.load_from_safetensors(src["path"]) \
                    if src["kind"] == "safetensors" else None
                if fp32 is None:
                    raise SystemExit(
                        "esm-cpp-fp32 requested but model is a quantized "
                        "GGUF; pass --model <fp32-source>")
                results["esm-cpp-fp32"] = _bench_esm_cpp(
                    fp32, ids_list, args.warmup, args.iters)
            else:
                results["esm-cpp-fp32"] = _bench_esm_cpp(
                    model, ids_list, args.warmup, args.iters)
        if "esm-cpp-int8" in modes:
            if not model.config.weights_quantized:
                # In-memory quantize for the comparison row.
                model.quantize_weights()
            results["esm-cpp-int8"] = _bench_esm_cpp(
                model, ids_list, args.warmup, args.iters)

    if "hf-eager-fp32" in modes:
        if not src["hf_id"]:
            raise SystemExit(
                "hf-eager-fp32 needs an HF model id. Pass --model "
                "facebook/esm2_t6_8M_UR50D or one of the shorthands.")
        results["hf-eager-fp32"] = _bench_hf(
            src["hf_id"], ids_list, args.warmup, args.iters)

    speedup = None
    if "esm-cpp-fp32" in results and "hf-eager-fp32" in results:
        a = results["esm-cpp-fp32"]["throughput_seqs_per_s"]
        b = results["hf-eager-fp32"]["throughput_seqs_per_s"]
        speedup = a / b if b > 0 else None
    if "esm-cpp-int8" in results and "hf-eager-fp32" in results:
        a = results["esm-cpp-int8"]["throughput_seqs_per_s"]
        b = results["hf-eager-fp32"]["throughput_seqs_per_s"]
        speedup_int8 = a / b if b > 0 else None
    else:
        speedup_int8 = None

    summary: dict[str, Any] = {
        "config": {
            "model": args.model,
            "model_source": src,
            "dataset": dataset_label,
            "num_sequences": n_seqs,
            "warmup": args.warmup,
            "iters": args.iters,
            "modes": modes,
        },
        "hardware": _hardware_block(),
        "results": results,
        "speedup_fp32_vs_hf": speedup,
        "speedup_int8_vs_hf": speedup_int8,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
