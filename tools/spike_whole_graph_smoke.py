"""Phase 13 T1: smoke that coremltools.convert(traceable ESM) actually runs.

We trace our own clean ESM-2 forward (tools/esm_traceable.py, loaded with HF
weights) on a tiny fixed shape, convert to a CoreML mlpackage with compute_units
CPU_AND_NE, then compare:

  - HF eager logits  vs.  our traceable forward logits  (must be ~bit-equal)
  - our traceable forward logits  vs.  coremltools predict (must be corr ≥ 0.99
    in fp16; tighter in fp32)

Pass criterion: both deltas pass. If they don't, the convert path is broken or
our reimplementation drifted from HF — fix the bug, don't widen the bound.

Run with the convert-time env:
    /tmp/ct312/bin/python tools/spike_whole_graph_smoke.py
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

import coremltools as ct

# tools/ sibling import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from esm_traceable import EsmCfg, EsmMaskedLMTraceable, load_from_hf


def corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    af -= af.mean()
    bf -= bf.mean()
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom == 0.0:
        return 0.0
    return float(np.dot(af, bf) / denom)


def argmax_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float((a.argmax(-1) == b.argmax(-1)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument(
        "--compute-units",
        choices=("CPU_ONLY", "CPU_AND_NE", "ALL"),
        default="CPU_AND_NE")
    ap.add_argument("--out", default="/tmp/esm2_8m_smoke.mlpackage")
    ap.add_argument("--corr-threshold", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        import shutil

        shutil.rmtree(out)

    print(f"[load] {args.model}  B={args.batch}  L={args.seq_len}")
    t = time.perf_counter()
    hf = EsmForMaskedLM.from_pretrained(args.model).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"  loaded in {time.perf_counter()-t:.1f}s")

    cfg = EsmCfg.from_hf(hf.config)
    trace_model = EsmMaskedLMTraceable(cfg, batch=args.batch, seq_len=args.seq_len)
    load_from_hf(trace_model, hf)
    print(f"  cfg: H={cfg.hidden_size} L={cfg.num_hidden_layers} heads={cfg.num_attention_heads} vocab={cfg.vocab_size}")

    rng = np.random.default_rng(args.seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    seqs = ["".join(rng.choice(list(aa), size=args.seq_len - 2)) for _ in range(args.batch)]
    enc = tok(seqs, return_tensors="pt", padding="max_length", max_length=args.seq_len, truncation=True)
    input_ids = enc["input_ids"].to(torch.int32)
    attention_mask = enc["attention_mask"].to(torch.int32)
    print(f"  input_ids: {tuple(input_ids.shape)} {input_ids.dtype}")

    # HF eager reference
    with torch.no_grad():
        hf_logits = hf(input_ids=input_ids.to(torch.long),
                       attention_mask=attention_mask.to(torch.long)).logits.numpy()
    print(f"  HF eager logits: {hf_logits.shape}  finite={np.isfinite(hf_logits).all()}")

    # Our traceable forward
    with torch.no_grad():
        my_logits = trace_model(input_ids, attention_mask).numpy()
    print(f"  ours (eager) logits: {my_logits.shape}  finite={np.isfinite(my_logits).all()}")

    c_hf = corr(hf_logits, my_logits)
    a_hf = argmax_agreement(hf_logits, my_logits)
    print(f"[parity] HF eager vs ours-eager: corr={c_hf:.6f} argmax_agree={a_hf:.4f}")
    if c_hf < 0.9999:
        print(f"FAIL: ours-eager drifted from HF (corr {c_hf:.6f} < 0.9999) — fix the model, don't widen the gate.")
        return 3

    print(f"[trace] precision={args.precision} compute_units={args.compute_units}")
    t = time.perf_counter()
    with torch.no_grad():
        traced = torch.jit.trace(
            trace_model, (input_ids, attention_mask), check_trace=False, strict=False)
    print(f"  traced in {time.perf_counter()-t:.1f}s")

    print(f"[convert] -> {out}")
    t = time.perf_counter()
    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    compute_units = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }[args.compute_units]
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=tuple(input_ids.shape), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=tuple(attention_mask.shape), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits")],
        compute_precision=precision,
        compute_units=compute_units,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    print(f"  converted in {time.perf_counter()-t:.1f}s")

    mlmodel.save(str(out))
    bytes_total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    print(f"  saved {out} ({bytes_total/1e6:.1f} MB)")

    print("[predict] reloading + running coremltools predict")
    m = ct.models.MLModel(str(out), compute_units=compute_units)
    ids_np = input_ids.numpy().astype(np.int32)
    mask_np = attention_mask.numpy().astype(np.int32)
    # warmup
    m.predict({"input_ids": ids_np, "attention_mask": mask_np})
    t = time.perf_counter()
    for _ in range(3):
        cm_out = m.predict({"input_ids": ids_np, "attention_mask": mask_np})
    t_predict = (time.perf_counter() - t) / 3.0
    pred = list(cm_out.values())[0]
    print(f"  predict in {t_predict*1000:.1f} ms (avg of 3)  out={pred.shape} {pred.dtype}")

    c_cm = corr(my_logits, pred)
    a_cm = argmax_agreement(my_logits, pred)
    c_hf_cm = corr(hf_logits, pred)
    a_hf_cm = argmax_agreement(hf_logits, pred)
    finite = bool(np.isfinite(pred).all())
    print(f"[parity] ours-eager vs CoreML predict: corr={c_cm:.6f} argmax_agree={a_cm:.4f}")
    print(f"[parity] HF eager   vs CoreML predict: corr={c_hf_cm:.6f} argmax_agree={a_hf_cm:.4f}  finite={finite}")

    if not finite:
        print("FAIL: non-finite logits in CoreML output")
        return 1
    if c_hf_cm < args.corr_threshold:
        print(f"FAIL: corr {c_hf_cm:.6f} < threshold {args.corr_threshold}")
        return 2
    print(f"PASS: HF-vs-CoreML corr {c_hf_cm:.6f} >= {args.corr_threshold}, argmax {a_hf_cm:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
