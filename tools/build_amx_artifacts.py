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
               precision: ct.precision, units: ct.ComputeUnit,
               fixed_m: int | None) -> ct.models.MLModel:
    """One Linear → single-op mlprogram, weight+bias baked as constants.

    If `fixed_m` is None: M is a Symbol+RangeDim (the Phase-11 AMX/BNNSGraph path
    — varlen-friendly, runs on CPU only). If `fixed_m` is an int: M is that fixed
    integer — required for ANE dispatch (ANE rejects dynamic shapes; verified in
    P12 T1 characterization). `units` selects the target compute unit.
    """
    N, K = int(W.shape[0]), int(W.shape[1])
    b = bias if bias is not None else np.zeros(N, dtype=np.float32)
    if fixed_m is None:
        m_sym = Symbol("M")

        @mb.program(input_specs=[mb.TensorSpec(shape=(m_sym, K), dtype=types.fp32)])
        def prog(x):  # pragma: no cover
            return mb.linear(x=x, weight=W, bias=b, name="out")

        return ct.convert(
            prog, convert_to="mlprogram",
            compute_units=units, compute_precision=precision,
            inputs=[ct.TensorType(name="x",
                                  shape=ct.Shape(shape=(ct.RangeDim(1, 65536), K)),
                                  dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS15,
        )

    @mb.program(input_specs=[mb.TensorSpec(shape=(fixed_m, K), dtype=types.fp32)])
    def prog_fixed(x):  # pragma: no cover
        return mb.linear(x=x, weight=W, bias=b, name="out")

    return ct.convert(
        prog_fixed, convert_to="mlprogram",
        compute_units=units, compute_precision=precision,
        minimum_deployment_target=ct.target.macOS15,
    )


def _save_compiled(model: ct.models.MLModel, dst: Path) -> None:
    """coremltools compiles to a temp .mlmodelc; copy it to `dst`."""
    src = Path(model.get_compiled_model_path())
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _spot_check(model: ct.models.MLModel, W: np.ndarray,
                bias: np.ndarray | None, M_probe: int) -> float:
    """Predict on a random input of shape [M_probe, K]; compare to a numpy
    reference. Returns max-magnitude-normalized error (fp16 typically ~5e-3)."""
    rng = np.random.default_rng(0)
    K = int(W.shape[1])
    X = (rng.standard_normal((M_probe, K)).astype(np.float32) * 0.5)
    Y = model.predict({"x": X})["out"].astype(np.float32)
    ref = X @ W.T
    if bias is not None:
        ref = ref + bias
    scale = max(float(np.abs(ref).max()), 1e-3)
    return float(np.max(np.abs(Y - ref)) / scale)


_UNITS = {
    "CPU_ONLY":    ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_NE":  ct.ComputeUnit.CPU_AND_NE,
    "ALL":         ct.ComputeUnit.ALL,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--safetensors", required=True, type=Path)
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--compute-units", choices=tuple(_UNITS), default="CPU_ONLY",
                    help="CPU_ONLY = AMX/BNNSGraph (Phase 11, RangeDim M). "
                         "CPU_AND_NE = ANE-targeted (Phase 12); requires --buckets.")
    ap.add_argument("--buckets", default="",
                    help="Comma-separated fixed M values (e.g. '2048,8192,16384'). "
                         "Required for ANE/static-shape artifacts. Empty -> RangeDim.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output dir. Static-shape mode lays out per-bucket "
                         "subdirs <out>/M-<m>/<linear>.mlmodelc; RangeDim mode "
                         "puts artifacts directly under <out>.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rel-err-threshold", type=float, default=3e-2)
    args = ap.parse_args()

    if not args.safetensors.exists():
        sys.exit(f"missing: {args.safetensors}")
    units = _UNITS[args.compute_units]
    buckets: list[int] = []
    if args.buckets:
        buckets = sorted({int(x) for x in args.buckets.split(",") if x})
        for m in buckets:
            if m <= 0 or m > 1_000_000:
                sys.exit(f"bad bucket {m}")
    if units == ct.ComputeUnit.CPU_AND_NE and not buckets:
        sys.exit("CPU_AND_NE requires --buckets (ANE rejects dynamic shapes)")
    if units == ct.ComputeUnit.CPU_ONLY and buckets:
        print("note: --buckets with CPU_ONLY → static-shape AMX artifacts "
              "(BNNSGraph still loads them; LinearProj must use the static path)")
    args.out.mkdir(parents=True, exist_ok=True)

    state = load_file(str(args.safetensors))
    linears = _discover_linears(state)
    if args.limit > 0:
        linears = linears[: args.limit]
    prec = (ct.precision.FLOAT16 if args.precision == "fp16"
            else ct.precision.FLOAT32)
    mode = f"buckets={buckets}" if buckets else "RangeDim(M)"
    print(f"discovered {len(linears)} Linears -> {args.out}  "
          f"units={args.compute_units}  mode={mode}", flush=True)
    if not linears:
        sys.exit("no Linears matched the discovery patterns")

    # Plan: (subdir, fixed_m, m_probe) per artifact group.
    if buckets:
        groups = [(f"M-{m}", m, m) for m in buckets]
    else:
        groups = [("", None, 8)]  # RangeDim — probe with a small M

    skipped = built = 0
    worst_err = 0.0
    t0 = time.perf_counter()
    total_n = len(linears) * len(groups)
    for sub, fixed_m, m_probe in groups:
        sub_out = args.out / sub if sub else args.out
        sub_out.mkdir(parents=True, exist_ok=True)
        for i, (name, W, bias) in enumerate(linears, 1):
            dst = sub_out / f"{name}.mlmodelc"
            if dst.exists() and (dst / "model.mil").exists():
                skipped += 1
                continue
            N, K = int(W.shape[0]), int(W.shape[1])
            t = time.perf_counter()
            model = _build_one(name, W, bias, prec, units, fixed_m)
            _save_compiled(model, dst)
            err = _spot_check(model, W, bias, m_probe)
            worst_err = max(worst_err, err)
            if err > args.rel_err_threshold:
                sys.exit(f"FAIL: {name} (M={fixed_m or 'dyn'}) max rel err "
                         f"{err:.4f} > {args.rel_err_threshold}")
            built += 1
            tag = f"M={fixed_m}" if fixed_m else "M=dyn"
            print(f"  [{built+skipped:4d}/{total_n}] {name} ({tag}) "
                  f"[N={N},K={K}]  err={err:.4f}  "
                  f"({time.perf_counter()-t:.1f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    total_bytes = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"\nDone: built {built} new, skipped {skipped} existing, "
          f"max rel err {worst_err:.4f}, {elapsed:.1f}s, "
          f"total size {total_bytes / 1e6:.0f} MB", flush=True)

    # Phase 14: stamp a manifest so the C++ auto-load can detect stale
    # artifacts (trace_sha mismatch). One manifest per artifact root —
    # for the bucket variant we write one manifest per M-<m>/ subdir.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _artifact_manifest import write_manifest as _write_manifest
        for sub, _fixed_m, _m_probe in groups:
            sub_out = args.out / sub if sub else args.out
            _write_manifest(
                sub_out,
                kind="amx-fp16",
                model_id=str(args.safetensors),  # path-as-id; the publish tool overrides
                precision=args.precision,
                compute_units=args.compute_units,
            )
    except Exception as e:
        print(f"warning: manifest write failed: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
