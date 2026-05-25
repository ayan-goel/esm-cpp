"""Phase 11 T2: convert-time artifact generator for the fp16-AMX backend.

For each Linear in an ESM-2 checkpoint, emit a per-Linear `.mlmodelc` containing
a single `mb.linear` op with the fp32 weight + bias *baked in as constants* and
`M` as a dynamic dimension (`RangeDim(1, 65536)`). The C++ BNNSGraph runtime
loads these at `Model::load`, caches a context per Linear, and executes them in
the forward path under `ESM_APPLE_AMX=on`. This is the heavy convert-time step;
the user-facing 3.14 runtime needs zero coremltools at runtime — BNNSGraph reads
the precompiled mlmodelc directly.

The P10 spike proved: weights baked as constants → BNNS pre-packs them into AMX
layout → ~2× the hand-written NEON SDOT path (10.4 ms vs ~20.5 ms on fc1 at
M=2048). Weights-as-runtime-input loses this pre-pack and falls back to ~17 ms.

Run with the Python 3.12 + coremltools 9.0 env (coremltools' native MIL blob
libs are not packaged in the Python 3.14 wheel as of v9.0):

  /tmp/ct312/bin/python tools/build_amx_artifacts.py \\
      --safetensors ~/.cache/huggingface/.../model.safetensors \\
      --precision fp16 \\
      --out weights/esm2_650m.amx-fp16
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Symbol, types
from safetensors.numpy import load_file


# The Linear-weight key patterns we materialize into mlmodelc. Order matches
# the model.cpp forward graph. `lm_head.decoder.weight` is the tied embedding
# matrix (N = vocab_size = 33) — small N, low AMX ROI, skipped here; the C++
# loader treats absent artifacts as a per-Linear fall-back to the default path.
_LAYER_KINDS = (
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
)
_TOP_KINDS = ("lm_head.dense",)

_LAYER_RE = re.compile(r"^esm\.encoder\.layer\.(\d+)\.(.+)\.weight$")


def _discover_linears(state: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return [(artifact_name, W[N,K] fp32, bias[N] fp32 or None), ...].

    Artifact name mirrors the safetensors weight key minus the `.weight` suffix,
    so the C++ loader maps tensor names → artifact paths trivially.
    """
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    layers: dict[int, dict[str, np.ndarray]] = {}
    for name, arr in state.items():
        m = _LAYER_RE.match(name)
        if m and m.group(2) in _LAYER_KINDS:
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = arr
    for layer_idx in sorted(layers):
        for kind in _LAYER_KINDS:
            w = layers[layer_idx].get(kind)
            if w is None:
                continue
            base = f"esm.encoder.layer.{layer_idx}.{kind}"
            b = state.get(f"{base}.bias")
            out.append((base, w.astype(np.float32, copy=False),
                        None if b is None else b.astype(np.float32, copy=False)))
    for kind in _TOP_KINDS:
        w = state.get(f"{kind}.weight")
        if w is None:
            continue
        b = state.get(f"{kind}.bias")
        out.append((kind, w.astype(np.float32, copy=False),
                    None if b is None else b.astype(np.float32, copy=False)))
    return out


def _build_one(name: str, W: np.ndarray, bias: np.ndarray | None,
               precision: ct.precision) -> ct.models.MLModel:
    """One Linear → single-op mlprogram, M as a Symbol (dynamic), weight+bias baked.

    The Symbol leaves M unbounded in the MIL program; the C++ runtime selects the
    actual M per forward via BNNSGraphContextSetDynamicShapes. The `inputs=` arg
    to `ct.convert` carries the bounds (RangeDim) into the compiled package.
    """
    N, K = int(W.shape[0]), int(W.shape[1])
    b = bias if bias is not None else np.zeros(N, dtype=np.float32)
    m_sym = Symbol("M")

    @mb.program(input_specs=[mb.TensorSpec(shape=(m_sym, K), dtype=types.fp32)])
    def prog(x):  # pragma: no cover (passed as a builder)
        return mb.linear(x=x, weight=W, bias=b, name="out")

    return ct.convert(
        prog, convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
        compute_precision=precision,
        inputs=[ct.TensorType(name="x",
                              shape=ct.Shape(shape=(ct.RangeDim(1, 65536), K)),
                              dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS15,
    )


def _save_compiled(model: ct.models.MLModel, dst: Path) -> None:
    """coremltools compiles to a temp .mlmodelc; copy it to `dst`."""
    src = Path(model.get_compiled_model_path())
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _spot_check(model: ct.models.MLModel, W: np.ndarray,
                bias: np.ndarray | None, M_probe: int = 8) -> float:
    """Predict on a random input; compare to a numpy reference.

    Returns `max|Y - ref| / max(|ref|, 1e-3)`: a single max-magnitude-normalized
    error that doesn't explode on near-zero entries (which is what a per-element
    relative metric does — see the dev log on this). fp16 typically lands ~5e-4.
    """
    rng = np.random.default_rng(0)
    K = int(W.shape[1])
    X = (rng.standard_normal((M_probe, K)).astype(np.float32) * 0.5)
    Y = model.predict({"x": X})["out"].astype(np.float32)
    ref = X @ W.T
    if bias is not None:
        ref = ref + bias
    scale = max(float(np.abs(ref).max()), 1e-3)
    return float(np.max(np.abs(Y - ref)) / scale)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--safetensors", required=True, type=Path)
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16",
                    help="fp16 hits the AMX-fp16 path (~2x SDOT in the P10 spike). "
                         "fp32 is the safety baseline (≈ cblas_sgemm).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output dir: <linear_name>.mlmodelc/ inside.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only build the first N Linears (debug / smoke).")
    ap.add_argument("--rel-err-threshold", type=float, default=3e-2,
                    help="Spot-check max-magnitude-normalized err vs numpy "
                         "ref. fp16 accumulation noise grows with K — for K=5120 "
                         "the typical worst-case is ~1.5%. 3% gives margin without "
                         "masking a real correctness bug (which would be O(1.0)).")
    args = ap.parse_args()

    if not args.safetensors.exists():
        sys.exit(f"missing: {args.safetensors}")
    args.out.mkdir(parents=True, exist_ok=True)

    state = load_file(str(args.safetensors))
    linears = _discover_linears(state)
    if args.limit > 0:
        linears = linears[: args.limit]
    print(f"discovered {len(linears)} Linears -> {args.out}", flush=True)
    if not linears:
        sys.exit("no Linears matched the discovery patterns")

    prec = (ct.precision.FLOAT16 if args.precision == "fp16"
            else ct.precision.FLOAT32)
    skipped = built = 0
    worst_err = 0.0
    t0 = time.perf_counter()
    for i, (name, W, bias) in enumerate(linears, 1):
        dst = args.out / f"{name}.mlmodelc"
        if dst.exists() and (dst / "model.mil").exists():
            skipped += 1
            continue
        N, K = int(W.shape[0]), int(W.shape[1])
        t = time.perf_counter()
        model = _build_one(name, W, bias, prec)
        _save_compiled(model, dst)
        err = _spot_check(model, W, bias)
        worst_err = max(worst_err, err)
        if err > args.rel_err_threshold:
            sys.exit(f"FAIL: {name} max rel err {err:.4f} > threshold "
                     f"{args.rel_err_threshold} (precision={args.precision})")
        built += 1
        print(f"  [{i:3d}/{len(linears)}] {name}  [N={N},K={K}]  "
              f"err={err:.4f}  ({time.perf_counter() - t:.1f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    total_mb = sum(
        sum(p.stat().st_size for p in (args.out / f"{n}.mlmodelc").rglob("*")
            if p.is_file()) for n, *_ in linears) / 1e6
    print(f"\nDone: built {built} new, skipped {skipped} existing, "
          f"max rel err {worst_err:.4f}, {elapsed:.1f}s, total size {total_mb:.0f} MB",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
