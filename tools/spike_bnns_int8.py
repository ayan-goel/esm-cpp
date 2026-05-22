"""Phase 10 T2 spike: is Apple-AMX INT8 (via CoreML/BNNS) faster than our SDOT?

The make-or-break question for the BNNSGraph headline: does a CoreML int8 model,
run on the CPU compute unit (which routes through BNNS -> the AMX coprocessor),
actually do *int8* matmul on AMX (the projected ~2x), or does it just dequantize
weights and run fp32 AMX (no win over the cblas FP32 path we already have)?

We compare, on the fc1 650M shape (M=512, K=1280, N=5120), CoreML predict latency
(CPU_ONLY) for: fp32 / weight-only-int8 / full-W8A8. Reference points from the P9
spike (same shape, M=512): SDOT int8 = 6.41 ms, cblas AMX fp32 = 4.95 ms. If the
best int8 CoreML model is meaningfully below ~4.95 ms, int8-on-AMX is real and the
pipeline is worth building. If it's >= the fp32 number, the lever is dead.

Run with the py3.12 coremltools venv (native MIL libs):
  /tmp/ct312/bin/python tools/spike_bnns_int8.py
"""
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

M, K, N = 2048, 1280, 5120
RNG = np.random.default_rng(0)
W_FP32 = (RNG.standard_normal((N, K)).astype(np.float32) * 0.05)
BIAS = (RNG.standard_normal(N).astype(np.float32) * 0.01)


def build_fp32(precision):
    @mb.program(input_specs=[mb.TensorSpec(shape=(M, K), dtype=types.fp32)])
    def prog(x):
        return mb.linear(x=x, weight=W_FP32, bias=BIAS, name="out")
    return ct.convert(prog, convert_to="mlprogram",
                      compute_units=ct.ComputeUnit.CPU_ONLY,
                      compute_precision=precision,
                      minimum_deployment_target=ct.target.macOS15)


def time_predict(model, label, iters=50, warmup=10):
    # Fresh random input every call so CoreML cannot cache a prediction.
    feeds = [{"x": (RNG.standard_normal((M, K)).astype(np.float32) * 0.5)}
             for _ in range(warmup + iters)]
    for i in range(warmup):
        model.predict(feeds[i])
    ts = []
    for i in range(iters):
        t0 = time.perf_counter()
        model.predict(feeds[warmup + i])
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    print(f"  {label:30s} p50={ts[len(ts)//2]:6.2f} ms   min={ts[0]:6.2f} ms")
    return ts[len(ts) // 2]


def main():
    print(f"fc1 650M shape: M={M} K={K} N={N}  (M=2048, fresh input/iter)")
    fp32 = build_fp32(ct.precision.FLOAT32)
    base = time_predict(fp32, "fp32 (precision=FLOAT32)")
    fp16 = build_fp32(ct.precision.FLOAT16)
    time_predict(fp16, "fp16 (precision=FLOAT16)")

    # Weight-only int8, per-channel symmetric (our weight recipe).
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights)
    wcfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel",
        weight_threshold=0))
    w8 = linear_quantize_weights(fp16, wcfg)
    w8.compute_units = ct.ComputeUnit.CPU_ONLY
    time_predict(w8, "weight-only int8")

    # Full W8A8: also quantize activations (needs calibration samples).
    try:
        from coremltools.optimize.coreml import linear_quantize_activations
        acfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", weight_threshold=0))
        samples = [{"x": (RNG.standard_normal((M, K)).astype(np.float32) * 0.5)}
                   for _ in range(4)]
        a8 = linear_quantize_activations(w8, acfg, sample_data=samples)
        a8.compute_units = ct.ComputeUnit.CPU_ONLY
        time_predict(a8, "W8A8 (act+weight int8)")
    except Exception as e:
        print(f"  W8A8 path failed: {type(e).__name__}: {e}")

    print(f"\nVerdict input: compare the int8 p50s to the fp32 control ({base:.2f} ms) "
          f"and to AMX-fp32 4.95ms / SDOT 6.41ms.")


if __name__ == "__main__":
    main()
