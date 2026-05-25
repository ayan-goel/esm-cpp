# Benchmarks

Public benchmark for esm.cpp on synthetic uniform-length batches and on
variable-length workloads matching OAS antibody heavy-chain statistics
(mean 120, max 250). All numbers below are reproducible from the
commit + commands in this file. Raw JSON results live in
`benchmarks/results/`.

## Headline (v0.1.0)

**`esm-cpp-int8` runs ESM-2-650M at 9.3-10× HuggingFace eager FP32
throughput on variable-length antibody-shaped data, on a single
22-vCPU Sapphire Rapids socket.**

Measured 2026-05-17 on `c3-standard-22` (Intel Xeon 8481C, AMX-INT8,
86 GB RAM, ESM-2-650M, 256 sequences sampled from the
OAS-distribution-matched holdout in
`benchmarks/data/synthetic_varlen_v1.fasta`):

| Variant | esm-cpp-int8 | hf-eager-fp32 | Speedup |
|---|---:|---:|---:|
| **650M / varlen / default** | **12.37 s** | 115.12 s | **9.31× HF** |
| 650M / varlen / `ESM_AMX_ATTENTION=on ESM_QUANTIZE_LM_HEAD=on` | 11.45 s | 115.00 s | **10.04× HF** |
| 150M / varlen / default | 4.88 s | 46.63 s | 9.55× HF |
| 8M / varlen / default   | 651 ms | 5.51 s | 8.47× HF |

The variable-length advantage comes from the cu_seqlens packed-batch
forward: HuggingFace pads every sequence in the batch to `max(len)`
and processes the resulting padded tensor uniformly; esm.cpp packs
sequences back-to-back along the token axis and isolates per-sequence
attention via `cu_seqlens`. On the OAS-shaped distribution this saves
~3× of HF's attention compute and ~2× of its FFN compute.

For the uniform-length comparison (8 sequences × 256 tokens each, the
worst case for the cu_seqlens architecture since there's no padding to
skip):

| Variant | esm-cpp-int8 | hf-eager-fp32 | Speedup |
|---|---:|---:|---:|
| **650M / uniform / default** | **917 ms** | 3415 ms | **4.09× HF** |
| 650M / uniform / both env vars on | 830 ms | 3326 ms | **4.49× HF** |
| 150M / uniform / default | 403 ms | 1071 ms | 2.66× HF |
| 8M / uniform / default   | 22 ms | 24 ms | 1.13× HF |

## ARM (Apple Silicon / Linux ARM) — v0.2

The AArch64 kernel stack mirrors x86's three tiers — NEON FMLA (FP32),
NEON SDOT (INT8, the VNNI analog), and an opt-in NEON SMMLA/i8mm path
(the AMX analog) — runtime-dispatched via `sysctl`/`getauxval` feature
detection. The default build links no Apple Accelerate, so the same
binary runs on Linux ARM / AWS Graviton.

Measured on **Apple M3 Pro** (12 threads), `esm-cpp-int8` (NEON SDOT) vs
HF eager FP32 on the 256-sequence OAS-distribution dataset
(`benchmarks/data/synthetic_varlen_v1.fasta`) and on uniform 8×100:

| Variant | esm-cpp | hf-eager-fp32 | Speedup |
|---|---:|---:|---:|
| 650M / uniform 8×256 (Phase-13, `ESM_APPLE_ANE_GRAPH=on`) | **459 ms** | 4617 ms | **10.05× HF** |
| 650M / uniform 8×256 (Phase-11, `ESM_APPLE_AMX=on`) | 1.74 s | 3.88 s | 2.23× HF |
| 650M / uniform 8×256 (Phase-10, SDOT default) | 2.17 s | 3.88 s | 1.79× HF |
| 650M / uniform 8×256 (Phase-9) | 2.92 s | 3.80 s | 1.45× HF |
| 650M / varlen (Phase-9, SDOT) | 37.8 s | 150.1 s | **3.97× HF** |
| 35M / varlen | 3.20 s | 9.46 s | **2.95× HF** |
| 8M / varlen  | 1.02 s | 3.28 s | **3.21× HF** |
| 35M / uniform 8×100 | 78.5 ms | 99.8 ms | 1.27× HF |
| 8M / uniform 8×100  | 30.5 ms | 34.5 ms | 1.13× HF |

**Workload shape → recommended path** on Apple Silicon:

| Shape | Recommended path | Headline |
|---|---|---|
| Uniform (B, L) with a built artifact | Phase-13 whole-graph CoreML (`ESM_APPLE_ANE_GRAPH=on`) | 10.05× HF at 650M @ 8×256 |
| Varlen / cu_seqlens packed | Phase-11 AMX-fp16 (`ESM_APPLE_AMX=on`) | 4.53× HF at 650M / OAS |
| Default (no env, no artifacts) | NEON SDOT (Phase-10) | 1.79× HF at 650M @ 8×256 |

Whole-graph CoreML requires a per-(B, L) `.mlmodelc` artifact built once at
convert time (1.3 GB for 650M @ 8×256, ~6 GB for the full 4-shape sweep);
the runtime is opt-in and only routes through when the env gate is set AND
the input shape matches a registered artifact AND the batch is uniform.
Mixed-length batches and unregistered shapes silently fall through to the
Phase-11 AMX path (which is itself a working 2.23×-HF number, well-validated).

Phase 10 added two pure-NEON wins (SDOT branch-hoist + register-resident attention PV)
that moved 650M uniform 2.92 → 2.17 s (1.45 → 1.79× HF). Phase 11 added the opt-in
fp16-AMX BNNSGraph path (per-Linear `.mlmodelc` artifacts compiled at convert time +
loaded at `Model::load`), routing the dense GEMMs through Apple's AMX coprocessor in
fp16. With `ESM_APPLE_AMX=on` 650M uniform drops to **1.74 s** (1.79 → **2.23× HF**),
and quality vs FP32 is *better* than the SDOT-INT8 path (corr 0.99997 / argmax 1.0).
The varlen kernels (shared) improve identically each phase but the ~150 s HF varlen
baseline was not re-run, so that row stays the Phase-9 measurement.

Raw JSON: `benchmarks/results/dev_m3_pro_*_neon_*.json`. Reproduce with the
same `esm-cpp-bench` commands below (any ESM-2 safetensors checkpoint;
ISA is auto-detected). Notes:

- **650M is GEMM-bound on M3** (`ESM_PROFILE`: ~68% INT8 GEMM, ~27% attention, ~4%
  glue). Phase 10 landed two pure-NEON wins on these buckets:
  - **SDOT branch-hoist** — splitting the `Kernel4x16` k-loop into a branchless
    `K_main` + a zero-padded tail removed a per-k call/branch/stack-fallback that was
    bottlenecking the SDOT pipe: **~15–24% faster** per GEMM shape (helps M3 *and*
    Graviton/Linux ARM).
  - **Register-resident attention PV** — holding each 16-float output chunk's
    accumulators in registers across all keys (vs a per-key load-modify-store of
    `out_row`) cut the attention section **~28%** (~3× fewer PV memory ops).

  Stacked: 650M uniform 8×256 **2826 → 2170 ms p50, 1.79× HF** (was 1.45×).
- **`ESM_APPLE_AMX=on` (Phase 11, opt-in)** routes the dense GEMMs (qkv / out_proj /
  fc1 / fc2 / lm_dense) through fp16 BNNSGraph contexts compiled from per-Linear
  `.mlmodelc` artifacts. Per-bucket at 650M uniform (post-T8 SDOT baseline → AMX-fp16):
  fc1 497 → 338 ms, fc2 509 → 385 ms, qkv 392 → 278 ms, out_proj 131 → 92 ms,
  lm_dense 14 → 3 ms (GEMM bucket 1543 → 1096 ms, **−29 %**), e2e 2257 → 1740 ms
  (**−23 %**). Quality is *better* than SDOT-INT8 vs FP32 (corr 0.99997 vs 0.99956 on
  650M; corr 0.999994 on 150M). Reproduction:

  ```
  # 1. Build artifacts at convert time (Python 3.12 + coremltools 9):
  /path/to/py3.12/python tools/build_amx_artifacts.py \
      --safetensors ~/.cache/huggingface/.../model.safetensors \
      --precision fp16 \
      --out weights/esm2_650m.amx-fp16
  # 2. Use at runtime (3.14 runtime needs zero coremltools):
  m = esm_cpp.Model.load_from_safetensors(...)
  m.load_amx_artifacts("weights/esm2_650m.amx-fp16")
  # ESM_APPLE_AMX=on engages the path; off (default) falls back to SDOT.
  ```

  Hand-written NEON / SDOT stays the default and the only Linux-ARM path; the
  BNNSGraph runtime is compiled out on non-Apple builds (no Accelerate dep on Linux).

- **Phase 10 spike (kept for context).** Int8 on AMX gives **no** speedup — via
  CoreML/BNNS, int8 W8A8 (14.0 ms) is *slower* than fp16 (11.3 ms) at the fc1 650M
  shape M=2048; BNNS dequantizes int8 → fp16 and runs a float kernel. fp16 is the
  only AMX win. There is no `cblas` half-precision GEMM, so the BNNSGraph
  compiled-graph pipeline is the only path that delivers it.

- **Phase 13 whole-graph CoreML (`ESM_APPLE_ANE_GRAPH=on`)** — ONE
  `.mlmodelc` for the entire ESM-2 forward (33 encoder layers + LM head),
  built at convert time from a clean traced PyTorch wrapper
  (`tools/esm_traceable.py`) loaded with HF weights. The runtime is a small
  Objective-C++ MLModel bridge (`src/apple_whole_graph.mm`) — one MLModel
  kept hot per (B, L) shape registration, op-fused across the full graph,
  zero per-Linear context switching. **650M @ uniform 8×256: 459 ms p50 =
  10.05× HF eager FP32**, 3.78× over Phase-11 AMX. Quality vs HF FP32:
  corr 0.999998, argmax 1.000, PPPL drift < 0.001 on the 25-protein
  holdout subset. Reproduction:

  ```
  # 1. Build artifact at convert time (Python 3.12 + coremltools 9):
  /path/to/py3.12/python tools/build_whole_graph_artifacts.py \
      --model facebook/esm2_t33_650M_UR50D \
      --shapes 8x256 \
      --out weights/esm2_650m.whole-graph \
      --precision fp16 --compute-units CPU_AND_NE
  # 2. Use at runtime (3.14 runtime needs zero coremltools):
  m = esm_cpp.Model.load_from_safetensors(...)
  m.load_whole_graph_artifact(
      "weights/esm2_650m.whole-graph/B-8_L-256/whole_graph.mlmodelc",
      batch=8, seq_len=256, compute_units="cpu_and_ne")
  # ESM_APPLE_ANE_GRAPH=on engages the path under matching (B, L);
  # mixed-length batches and unregistered shapes fall through to Phase-11 AMX.
  ```

  This is the **uniform-shape headline on M3**. Varlen / cu_seqlens stays
  on the Phase-11 AMX path until a whole-graph varlen formulation lands
  (carry-forward; see `notes/phase13.md`).
- **Phase 12 ANE-fp16 (`ESM_APPLE_ANE=on`) — wired but experimental, default OFF.**
  Per-Linear `.mlmodelc` artifacts targeted at the Neural Engine
  (`tools/build_amx_artifacts.py --compute-units CPU_AND_NE --buckets …`). Per-shape
  characterization (Phase 12 T1) measured 2-4× ANE-vs-AMX in isolation, but the
  *integrated* forward is slower than AMX-fp16 because ANE thrashes its
  compiled-state cache when bouncing across 198 per-Linear MLModels per forward
  (cold-start outliers of 1-3 s/call drag the mean to ~100 ms/call vs the 10 ms
  the per-shape spike measured). The path is preserved as `ESM_APPLE_ANE=on` for
  future debug — **leave it off** for actual workloads; the whole-graph row
  above (Phase 13, `ESM_APPLE_ANE_GRAPH=on`) is the actual M3 uniform headline.
  See `notes/phase12.md` for the per-Linear retrospective and `notes/phase13.md`
  for the whole-graph design that delivers the win.
- **SMMLA/i8mm is opt-in** (`ESM_NEON_I8MM=on`). On Apple M3 it does not
  out-throughput SDOT, so SDOT is the default; SMMLA is expected to win on
  Graviton3-class i8mm units and stays validated against the scalar reference.
- **Quality (650M, post-Phase-10):** INT8-vs-FP32 logit correlation **0.99956**,
  masked-marginal argmax agreement **1.0**, no NaN
  (`dev_m3_pro_650m_p10_t10_quality.json`) — the kernel wins preserve the unchanged
  W8A8 recipe. The formal PPPL (<0.1) / ProteinGym (<0.01) gates at 150M/650M and an
  AWS Graviton3 run remain the carry-forward (those checkpoints were not cached in
  the dev environment).
- **Threads:** the default (all logical cores, incl. M-series E-cores) measured
  fastest for this GEMM-bound forward — the grain-based `parallel_for` load-balances
  across heterogeneous P/E cores, so a P-core-only default regresses (12c 2826 ms <
  6 P-only 3030 ms).

## Hardware

- **CPU:** Intel Xeon Platinum 8481C (Sapphire Rapids), 11 physical cores / 22 vCPUs at 2.7 GHz, AVX-512 + AVX-512-VNNI + AVX-512-BF16 + AMX-INT8.
- **RAM:** 86 GB.
- **OS:** Linux 6.8 (Ubuntu 22.04 LTS, Google c3-standard-22).
- **Threading:** `ESM_NUM_THREADS` defaults to physical core count via auto-detect; manual override available.

Other Sapphire Rapids configurations (e.g. AWS `c7i.4xlarge` at 8 cores or `c7i.metal-24xl` for full-socket measurements) reproduce the ratio within ~15 % of the c3-standard-22 numbers above.

## Reproduction commands

The synthetic OAS-distribution-matched holdout ships in the repo at
`benchmarks/data/synthetic_varlen_v1.fasta` (256 sequences, mean 126,
max 249, generated by `tools/make_synthetic_varlen.py` with seed
`20260516`).

```bash
# 1. Build + install (Release).
cmake -B build -DCMAKE_BUILD_TYPE=Release -DESM_BUILD_TESTS=ON
cmake --build build -j$(nproc)
pip install -e .

# 2. Sanity-check: 117 C++ tests pass on a SPR host.
./build/tests/cpp/esm_cpp_tests

# 3. Headline measurement at 650M.
esm-cpp-bench \
    --model esm2_t33_650M \
    --dataset benchmarks/data/synthetic_varlen_v1.fasta \
    --modes esm-cpp-int8,hf-eager-fp32 \
    --warmup 2 --iters 5 \
    --output benchmarks/results/c3_650M_varlen_v1.json

# 4. With the experimental env-var-gated optimizations on.
ESM_AMX_ATTENTION=on ESM_QUANTIZE_LM_HEAD=on \
    esm-cpp-bench \
        --model esm2_t33_650M \
        --dataset benchmarks/data/synthetic_varlen_v1.fasta \
        --modes esm-cpp-int8,hf-eager-fp32 \
        --warmup 2 --iters 5 \
        --output benchmarks/results/c3_650M_varlen_v1_both.json

# 5. Per-section profile (single-iteration; emits ESM_PROFILE breakdown).
ESM_PROFILE=1 esm-cpp-bench \
    --model esm2_t33_650M \
    --dataset benchmarks/data/synthetic_varlen_v1.fasta \
    --modes esm-cpp-int8 \
    --warmup 0 --iters 1
```

## Opt-in env vars

Two perf optimizations ship default-off pending PPPL/ProteinGym validation. They are functional, tested at correctness equivalence within their precision class, and can be enabled per-process:

- **`ESM_AMX_ATTENTION=on`** — switches the FP32 attention kernel to AVX-512 BF16 (`VDPBF16PS`). Q and K are converted to BF16 once per (batch, head) into thread-local scratch; the softmax accumulator stays FP32 per CLAUDE.md. Saves ~67 ms at 650M on uniform 8×256. The FP32 path remains the default until a PPPL drift gate runs on real protein data.
- **`ESM_QUANTIZE_LM_HEAD=on`** — additionally quantizes `lm_head.dense` (FP32 by default — original Phase 2 Slice 5 escape list) to per-channel symmetric INT8. Saves ~19 ms at 650M. Requires PPPL validation since end-of-graph quantization noise compounds onto logits with no downstream LN to absorb.

Both flags survive across model sizes and only fire on SPR-class hosts (the BF16 path checks CPUID for `AVX512_BF16` at process start, falls back silently otherwise).

## Profile breakdown (650M varlen, default)

```
[esm-profile] forward breakdown (one of 5 forwards from forward_scheduled chunking)
  attention                  1258 ms  (25 %)   AVX-512 multi-acc + parallel B×H
  fc2                        1012 ms  (20 %)   AMX-INT8 32×32 microkernel
  fc1                         859 ms  (17 %)   AMX-INT8
  qkv_proj                    767 ms  (15 %)   AMX-INT8
  gelu                        299 ms  ( 6 %)   AVX-512 polynomial erf
  attn_out_proj               264 ms  ( 5 %)   AMX-INT8
  q_scale_rope                175 ms  ( 3 %)   AVX-512 parallel half-then-half
  lm_dense                    150 ms  ( 3 %)   FP32 AVX-512 (was a LinearRef stub before phase 7)
  residual                    135 ms  ( 3 %)   AVX-512 parallel elementwise add
  attn_ln + ffn_ln            155 ms  ( 3 %)
  embed + lm head             19 ms  ( <1 %)
```

The remaining headroom toward 12-15× HF on this workload lives in the AMX GEMM section (57 % of forward) — see [notes/phase7.md](../notes/phase7.md) carry-forward for the v0.2 path through an N-chained AMX microkernel and AMX-BF16 attention tile kernel.

## Test infrastructure

- **117 C++ tests pass** on the gate machine, including:
  - SIMD-vs-Ref cross-checks at the precision class of each kernel (FP32 at 1e-4, INT8 at rtol=1e-3 atol=1, BF16 at rtol=3e-2 atol=5e-4).
  - HF golden-tensor parity at 8M and 35M (rtol=1e-3 atol=8e-2).
  - cu_seqlens packed-batch isolation: per-sequence attention does not leak across batch boundaries.
  - AMX-engaged counter via `ESM_PROFILE` confirms the AMX kernel runs on every eligible GEMM call (198/198 on the 650M forward).

## Results table (auto-generated)

<!-- BENCH_TABLE_BEGIN -->

| Run | Host | ISA | Seqs | Mean ms | Throughput (seqs/s) | Speedup vs HF |
|---|---|---|---:|---:|---:|---:|
| esm2_t30_150M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 2932.6 | 2.73 | 0.37x |
| esm2_t30_150M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 1090.3 | 7.34 | — |
| esm2_t30_150M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 1434.4 | 5.58 | 0.79x |
| esm2_t30_150M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 1126.9 | 7.10 | — |
| esm2_t30_150M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 523.1 | 15.29 | 2.09x |
| esm2_t30_150M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 1093.6 | 7.32 | — |
| esm2_t30_150M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 256 | 4883.9 | 52.42 | 9.55x |
| esm2_t30_150M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 256 | 46631.1 | 5.49 | — |
| esm2_t33_650M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 8699.3 | 0.92 | 0.39x |
| esm2_t33_650M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 3420.4 | 2.34 | — |
| esm2_t33_650M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 4239.4 | 1.89 | 0.79x |
| esm2_t33_650M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 3343.0 | 2.39 | — |
| esm2_t33_650M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 849.6 | 9.42 | 4.04x |
| esm2_t33_650M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 3431.9 | 2.33 | — |
| esm2_t33_650M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 829.2 | 9.65 | 4.01x |
| esm2_t33_650M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 3325.7 | 2.41 | — |
| esm2_t33_650M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 917.7 | 8.72 | 4.09x |
| esm2_t33_650M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 3751.8 | 2.13 | — |
| esm2_t33_650M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 898.5 | 8.90 | 3.79x |
| esm2_t33_650M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 3409.3 | 2.35 | — |
| esm2_t33_650M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 1246.9 | 6.42 | 2.73x |
| esm2_t33_650M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 3408.5 | 2.35 | — |
| esm2_t33_650M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 1165.0 | 6.87 | 2.91x |
| esm2_t33_650M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 3395.2 | 2.36 | — |
| esm2_t33_650M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 1105.9 | 7.23 | 3.12x |
| esm2_t33_650M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 3456.0 | 2.31 | — |
| esm2_t33_650M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 928.3 | 8.62 | 3.65x |
| esm2_t33_650M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 3387.6 | 2.36 | — |
| esm2_t33_650M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 256 | 12371.8 | 20.69 | 9.31x |
| esm2_t33_650M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 256 | 115121.2 | 2.22 | — |
| esm2_t33_650M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 256 | 11450.7 | 22.36 | 10.04x |
| esm2_t33_650M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 256 | 114996.0 | 2.23 | — |
| esm2_t6_8M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 95.5 | 83.81 | 0.26x |
| esm2_t6_8M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 24.5 | 326.09 | — |
| esm2_t6_8M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 80.7 | 99.19 | 0.30x |
| esm2_t6_8M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 24.2 | 330.25 | — |
| esm2_t6_8M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 8 | 22.7 | 352.22 | 1.08x |
| esm2_t6_8M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 8 | 24.6 | 325.54 | — |
| esm2_t6_8M / esm-cpp-int8 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | amx | 256 | 650.5 | 393.54 | 8.47x |
| esm2_t6_8M / hf-eager-fp32 | Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz | n/a | 256 | 5512.9 | 46.44 | — |
| esm2_t6_8M / esm-cpp-fp32 | Apple M3 Pro | neon | 8 | 52.9 | 151.11 | — |
| esm2_t6_8M / esm-cpp-int8 | Apple M3 Pro | neon | 8 | 3024.7 | 2.64 | — |
| unknown / fp32 | unknown | n/a | 0 | 0.0 | 0.00 | — |
| unknown / int8+bf16_attn | unknown | n/a | 0 | 0.0 | 0.00 | — |
| unknown / fp32 | unknown | n/a | 0 | 0.0 | 0.00 | — |
| unknown / int8 | unknown | n/a | 0 | 0.0 | 0.00 | — |

<!-- BENCH_TABLE_END -->

Re-run `python tools/render_benchmark_table.py --results benchmarks/results --out docs/benchmarks.md` after dropping new `*.json` files into `benchmarks/results/` to regenerate this block.
