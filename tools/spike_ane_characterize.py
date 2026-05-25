"""Phase 12 T1: characterize ANE-vs-AMX speedup before committing to the
integration. Three measurements:

  (a) per-shape: ANE-vs-AMX ratio on all five 650M Linear shapes at fixed M=2048
      — does the spike's ~5.5× hold beyond fc1?
  (b) per-M:    ANE-vs-AMX ratio on fc1 across M ∈ {64..32768} — where does ANE
      stop winning (too-small M has launch overhead; too-large M may not fit ANE).
      Picks the bucket set for T2.
  (c) drift:    ANE-fp16 vs CPU-fp16 max-magnitude-normalized output diff on
      one Linear — gates the quality risk.

Run with the Python 3.12 + coremltools 9 env:
  /tmp/ct312/bin/python tools/spike_ane_characterize.py --out <path>
"""
import argparse, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

# Four critical 650M shapes (d=1280, ffn=5120) + lm_dense (which is [d,d]).
SHAPES = {
    "qkv_d_d":    (1280, 1280),
    "out_d_d":    (1280, 1280),
    "fc1_4d_d":   (5120, 1280),
    "fc2_d_4d":   (1280, 5120),
    "lm_dense":   (1280, 1280),
}


def build_model(N, K, units, M):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)).astype(np.float32) * 0.05)
    b = (rng.standard_normal(N).astype(np.float32) * 0.01)

    @mb.program(input_specs=[mb.TensorSpec(shape=(M, K), dtype=types.fp32)])
    def prog(x):
        return mb.linear(x=x, weight=W, bias=b, name="out")

    return ct.convert(prog, convert_to="mlprogram",
                      compute_units=units,
                      compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.macOS15)


def time_p50(model, M, K, iters=40, warmup=10):
    rng = np.random.default_rng(0)
    feeds = [{"x": (rng.standard_normal((M, K)).astype(np.float32) * 0.5)}
             for _ in range(warmup + iters)]
    for i in range(warmup):
        model.predict(feeds[i])
    ts = []
    for i in range(iters):
        t = time.perf_counter()
        model.predict(feeds[warmup + i])
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def drift_metric(M, K, N):
    """Max-magnitude-normalized diff between ANE-fp16 and CPU-fp16 on the same
    Linear (random weight). >> 1% would be a real quality concern."""
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)).astype(np.float32) * 0.05)
    b = (rng.standard_normal(N).astype(np.float32) * 0.01)
    X = (rng.standard_normal((M, K)).astype(np.float32) * 0.5)

    @mb.program(input_specs=[mb.TensorSpec(shape=(M, K), dtype=types.fp32)])
    def prog(x):
        return mb.linear(x=x, weight=W, bias=b, name="out")

    m_cpu = ct.convert(prog, convert_to="mlprogram",
                       compute_units=ct.ComputeUnit.CPU_ONLY,
                       compute_precision=ct.precision.FLOAT16,
                       minimum_deployment_target=ct.target.macOS15)
    m_ane = ct.convert(prog, convert_to="mlprogram",
                       compute_units=ct.ComputeUnit.CPU_AND_NE,
                       compute_precision=ct.precision.FLOAT16,
                       minimum_deployment_target=ct.target.macOS15)
    y_cpu = m_cpu.predict({"x": X})["out"].astype(np.float32)
    y_ane = m_ane.predict({"x": X})["out"].astype(np.float32)
    scale = max(float(np.abs(y_cpu).max()), 1e-3)
    return float(np.max(np.abs(y_cpu - y_ane)) / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results = {"phase": "12 T1", "hardware": "Apple M3 Pro",
               "per_shape_M2048_ms": {}, "per_M_fc1_ms": {},
               "ane_vs_cpu_fp16_drift": {}}

    # (a) per-shape at M=2048
    M = 2048
    print(f"\n[a] per-shape at M={M}:")
    print(f"  {'shape':18s} {'CPU(AMX)':>10s} {'ANE':>8s} {'speedup':>10s}")
    for label, (N, K) in SHAPES.items():
        m_cpu = build_model(N, K, ct.ComputeUnit.CPU_ONLY, M)
        m_ane = build_model(N, K, ct.ComputeUnit.CPU_AND_NE, M)
        cpu_p50, cpu_min = time_p50(m_cpu, M, K)
        ane_p50, ane_min = time_p50(m_ane, M, K)
        ratio = cpu_p50 / max(ane_p50, 1e-6)
        results["per_shape_M2048_ms"][label] = {
            "N": N, "K": K, "cpu_p50": cpu_p50, "cpu_min": cpu_min,
            "ane_p50": ane_p50, "ane_min": ane_min, "ane_speedup": ratio,
        }
        print(f"  {label:18s} {cpu_p50:8.2f} {ane_p50:8.2f}  {ratio:8.2f}x")

    # (b) per-M on fc1
    print(f"\n[b] per-M on fc1 (N=5120 K=1280):")
    print(f"  {'M':>6s} {'CPU(AMX)':>10s} {'ANE':>8s} {'speedup':>10s}")
    N, K = 5120, 1280
    for m in (64, 256, 1024, 2048, 4096, 8192, 16384, 32768):
        try:
            m_cpu = build_model(N, K, ct.ComputeUnit.CPU_ONLY, m)
            m_ane = build_model(N, K, ct.ComputeUnit.CPU_AND_NE, m)
            cpu_p50, _ = time_p50(m_cpu, m, K)
            ane_p50, _ = time_p50(m_ane, m, K)
            ratio = cpu_p50 / max(ane_p50, 1e-6)
            results["per_M_fc1_ms"][str(m)] = {
                "cpu_p50": cpu_p50, "ane_p50": ane_p50, "ane_speedup": ratio,
            }
            print(f"  {m:6d} {cpu_p50:8.2f} {ane_p50:8.2f}  {ratio:8.2f}x")
        except Exception as e:
            results["per_M_fc1_ms"][str(m)] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {m:6d}  ERROR: {type(e).__name__}: {e}")

    # (c) drift on fc1 at M=2048
    print("\n[c] ANE-vs-CPU drift on fc1 (M=2048):")
    drift = drift_metric(2048, 1280, 5120)
    results["ane_vs_cpu_fp16_drift"]["fc1_M2048"] = drift
    print(f"  max-magnitude-normalized = {drift:.4f}  (gate ~ 0.01)")

    # Bucket recommendation
    ratios = {int(m): r["ane_speedup"] for m, r in results["per_M_fc1_ms"].items()
              if "ane_speedup" in r}
    winning_Ms = sorted([m for m, r in ratios.items() if r >= 2.0])
    results["bucket_recommendation"] = {
        "winning_M_threshold": "ANE speedup >= 2x over CPU/AMX",
        "winning_Ms": winning_Ms,
        "suggested_buckets": (
            "Pick 3 covering uniform (~2048), medium-varlen (~8192), "
            "large-varlen (~32768) IF all three win; otherwise drop the losers."
        ),
    }
    print(f"\nWinning M (ANE >= 2x AMX): {winning_Ms}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
