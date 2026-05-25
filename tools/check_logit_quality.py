"""Fast INT8-vs-FP32 logit-quality proxy on Apple Silicon (the P9/P10 quality
check). Confirms a kernel change preserves quality without the slow full PPPL:
loads FP32 and INT8 650M, compares final logits (Pearson correlation + argmax
agreement) over a few sequences, and checks for NaN. The formal PPPL (<0.1) /
ProteinGym (<0.01) gates use the unchanged W8A8 recipe and remain the carry-forward.

  .venv-arm/bin/python tools/check_logit_quality.py --model esm2_t33_650M
"""
import argparse, json, sys
import numpy as np
import esm_cpp

_HF = {"esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
       "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D"}


def _path(hf_id):
    from pathlib import Path
    snaps = Path.home() / ".cache/huggingface/hub" / f"models--{hf_id.replace('/','--')}" / "snapshots"
    c = list(snaps.glob("*/model.safetensors"))
    if not c:
        sys.exit(f"missing cache for {hf_id}")
    return str(c[0])


def _stats(ref, got):
    corrs, agrees, any_nan = [], [], False
    for a, b in zip(ref, got):
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            any_nan = True
        corrs.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
        agrees.append(float((a.argmax(-1) == b.argmax(-1)).mean()))
    return {"min_logit_correlation": min(corrs) if corrs else 0.0,
            "argmax_agreement": float(np.mean(agrees)) if agrees else 0.0,
            "any_nan": any_nan}


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t33_650M", choices=list(_HF))
    ap.add_argument("--amx-dir", default=None,
                    help="Optional path to the AMX fp16 artifact directory built "
                         "by tools/build_amx_artifacts.py. When set, also runs the "
                         "AMX-fp16 forward and reports its drift vs FP32.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tok = esm_cpp.Tokenizer()
    rng = np.random.default_rng(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    seqs = ["".join(rng.choice(list(aa), size=int(n))) for n in rng.integers(60, 200, size=5)]
    ids = [np.asarray(tok.encode(s), dtype=np.int32) for s in seqs]

    p = _path(_HF[args.model])
    # FP32 reference.
    os.environ.pop("ESM_APPLE_AMX", None)
    m = esm_cpp.Model.load_from_safetensors(p)
    fp32 = [m.forward(i) for i in ids]
    del m
    # INT8 SDOT.
    m = esm_cpp.Model.load_from_safetensors(p)
    m.quantize_weights()
    int8 = [m.forward(i) for i in ids]
    res = {"model": args.model, "isa": esm_cpp.current_isa(),
           "num_seqs": len(seqs),
           "int8_vs_fp32": _stats(fp32, int8)}
    # AMX fp16 (optional).
    if args.amx_dir:
        del m
        m = esm_cpp.Model.load_from_safetensors(p)
        loaded = m.load_amx_artifacts(args.amx_dir)
        os.environ["ESM_APPLE_AMX"] = "on"
        amx = [m.forward(i) for i in ids]
        res["amx_loaded"] = loaded
        res["amx_fp16_vs_fp32"] = _stats(fp32, amx)
        res["amx_fp16_vs_int8"] = _stats(int8, amx)
    print(json.dumps(res, indent=2))
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
