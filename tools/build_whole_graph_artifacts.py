"""Phase 13 T3/T4: convert-time artifact generator for the whole-graph ANE path.

For a given HF ESM-2 checkpoint + (B, L) shape pair, traces our clean
EsmMaskedLMTraceable (tools/esm_traceable.py), converts to a CoreML mlprogram
at fp16 with compute_units=CPU_AND_NE, and emits a compiled .mlmodelc bundle
the C++ WholeGraphContext can mmap at Model load.

Each (model, B, L) yields exactly one .mlmodelc — distinct from Phase-12's
per-Linear-per-bucket layout. Whole-graph ANE wants ONE compiled graph hot,
not 198 tiny ones (Phase-12 retro).

Layout produced:
    <out>/B-<B>_L-<L>/whole_graph.mlmodelc

Idempotent: skips rebuilding if the output bundle already exists, unless
--force is passed.

Run with the convert-time env:
    /tmp/ct312/bin/python tools/build_whole_graph_artifacts.py \\
        --model facebook/esm2_t6_8M_UR50D \\
        --shapes 1x256,8x256 \\
        --out weights/esm2_8m.whole-graph
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

import coremltools as ct

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esm_traceable import EsmCfg, EsmMaskedLMTraceable, load_from_hf
from _artifact_manifest import write_manifest as _write_manifest


def parse_shapes(s: str) -> list[tuple[int, int]]:
    shapes = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        b, l = tok.lower().split("x")
        shapes.append((int(b), int(l)))
    return shapes


def corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    af -= af.mean()
    bf -= bf.mean()
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom == 0.0:
        return 0.0
    return float(np.dot(af, bf) / denom)


def build_one(hf, tok, cfg: EsmCfg, batch: int, seq_len: int,
              out_dir: Path, precision: str, compute_units: str,
              corr_threshold: float, force: bool, seed: int,
              model_id: str = "") -> bool:
    tag = f"B-{batch}_L-{seq_len}"
    target_dir = out_dir / tag
    bundle = target_dir / "whole_graph.mlmodelc"
    if bundle.exists() and not force:
        print(f"  [skip] {bundle} exists (use --force to rebuild)")
        return True
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] {tag} precision={precision} compute_units={compute_units}")
    trace_model = EsmMaskedLMTraceable(cfg, batch=batch, seq_len=seq_len)
    load_from_hf(trace_model, hf)

    rng = np.random.default_rng(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    seqs = ["".join(rng.choice(list(aa), size=seq_len - 2)) for _ in range(batch)]
    enc = tok(seqs, return_tensors="pt", padding="max_length", max_length=seq_len, truncation=True)
    input_ids = enc["input_ids"].to(torch.int32)
    attention_mask = enc["attention_mask"].to(torch.int32)

    with torch.no_grad():
        hf_logits = hf(input_ids=input_ids.to(torch.long),
                       attention_mask=attention_mask.to(torch.long)).logits.numpy()
        ours_logits = trace_model(input_ids, attention_mask).numpy()
    c_ours = corr(hf_logits, ours_logits)
    if c_ours < 0.9999:
        print(f"  FAIL: ours-eager drift from HF (corr {c_ours:.6f} < 0.9999)")
        return False

    with torch.no_grad():
        traced = torch.jit.trace(
            trace_model, (input_ids, attention_mask), check_trace=False, strict=False)

    t = time.perf_counter()
    precision_ct = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    compute_units_ct = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }[compute_units]
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=tuple(input_ids.shape), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=tuple(attention_mask.shape), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits")],
        compute_precision=precision_ct,
        compute_units=compute_units_ct,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    print(f"  converted in {time.perf_counter()-t:.1f}s")

    # Save as mlpackage first, then compile to mlmodelc via CoreML's compileModel.
    pkg = target_dir / "whole_graph.mlpackage"
    if pkg.exists():
        shutil.rmtree(pkg)
    mlmodel.save(str(pkg))
    pkg_bytes = sum(p.stat().st_size for p in pkg.rglob('*') if p.is_file())
    print(f"  saved mlpackage  ({pkg_bytes/1e6:.1f} MB)")

    # Compile to mlmodelc using coremltools' compileModel utility, then save.
    # get_compiled_model_path() returns a temp dir that the MLModel cleans up
    # when it goes out of scope — keep a reference live while copying AND for
    # the spot-check predict (.mlmodelc has no Manifest.json so we can't re-open
    # it as an MLModel here in Python; the C++ bridge can though, via
    # MLModel.modelWithContentsOfURL).
    print("  compiling -> mlmodelc")
    t = time.perf_counter()
    holder = ct.models.MLModel(str(pkg), compute_units=compute_units_ct)
    compiled_url = holder.get_compiled_model_path()
    print(f"  compiled in {time.perf_counter()-t:.1f}s")
    if bundle.exists():
        shutil.rmtree(bundle)
    shutil.copytree(compiled_url, str(bundle))
    mlmodelc_bytes = sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    print(f"  saved mlmodelc ({mlmodelc_bytes/1e6:.1f} MB) -> {bundle}")

    # Spot-check via the holder (still wraps the in-memory compiled MLModel)
    print("  spot-check predict (coremltools)")
    ids_np = input_ids.numpy().astype(np.int32)
    mask_np = attention_mask.numpy().astype(np.int32)
    cm_out = holder.predict({"input_ids": ids_np, "attention_mask": mask_np})
    pred = list(cm_out.values())[0]
    c_cm = corr(hf_logits, pred)
    finite = bool(np.isfinite(pred).all())
    print(f"  corr(HF, CoreML): {c_cm:.6f}  finite={finite}")
    if not finite or c_cm < corr_threshold:
        print(f"  FAIL: spot-check corr {c_cm:.6f} < {corr_threshold} (or non-finite)")
        return False

    # Keep the .mlpackage alongside the .mlmodelc so coremltools can re-open
    # the model for predict/quality runs (the .mlmodelc has no Manifest.json
    # and can't be re-loaded from Python).
    _write_manifest(
        target_dir,
        kind="whole-graph",
        model_id=model_id,
        precision=precision,
        compute_units=compute_units,
        shape=(batch, seq_len),
    )
    print(f"  [ok] {tag}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--shapes", required=True, help="comma-separated BxL pairs, e.g. 1x256,8x256")
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument(
        "--compute-units", choices=("CPU_ONLY", "CPU_AND_NE", "ALL"),
        default="CPU_AND_NE")
    ap.add_argument("--corr-threshold", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(f"[load] {args.model}")
    t = time.perf_counter()
    hf = EsmForMaskedLM.from_pretrained(args.model).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = EsmCfg.from_hf(hf.config)
    print(f"  loaded in {time.perf_counter()-t:.1f}s  H={cfg.hidden_size} L={cfg.num_hidden_layers} vocab={cfg.vocab_size}")

    shapes = parse_shapes(args.shapes)
    n_ok = 0
    for batch, seq_len in shapes:
        ok = build_one(
            hf, tok, cfg, batch, seq_len, args.out, args.precision,
            args.compute_units, args.corr_threshold, args.force, args.seed,
            model_id=args.model)
        if ok:
            n_ok += 1
    print(f"[done] built {n_ok}/{len(shapes)} shapes -> {args.out}")
    return 0 if n_ok == len(shapes) else 1


if __name__ == "__main__":
    sys.exit(main())
