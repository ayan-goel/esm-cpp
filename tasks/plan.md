# Phase 1 — Implementation Plan

**Scope.** Turn the Phase 0 scalar reference into a SIMD baseline good enough to beat HuggingFace PyTorch eager on CPU. Concretely: runtime kernel dispatch, a Goto-packed FP32 GEMM microkernel (AVX-512 primary, NEON dev fallback), a FlashAttention-style packed-varlen attention kernel with a `cu_seqlens` interface, an arena allocator that makes the forward zero-alloc, and a thread pool that parallelizes across batch and the FFN-4d dimension. Microbenchmarks on the four critical shapes; end-to-end gate measurement on a real AVX-512 instance. See [SPEC.md §2](../SPEC.md), [research-report.md §4–§6](../research-report.md), and the Phase 0 hand-off in [notes/phase0.md](../notes/phase0.md).

**Gate (must hold to declare Phase 1 done).**
- ≥2× HuggingFace `EsmModel` (PyTorch eager FP32) throughput on **ESM-2-650M, batch 16, 300aa, single-socket x86 with AVX-512+VNNI**.
- SGEMM microkernels reach ≥80% of MKL (or the best available CPU SGEMM library on the gate machine) on the four critical shapes for `d ∈ {320, 640, 1280, 2560}`: `[B·L, d, 3d]`, `[B·L, d, d]`, `[B·L, d, 4d]`, `[B·L, 4d, d]`.
- All Phase 0 correctness gates still hold under the SIMD path. Layer-by-layer parity holds with tolerances tightened as far as the SIMD summation order allows (target: original SPEC `< 1e-4` on logits; honest fallback documented if FMA + Goto packing can't get there).
- Forward path zero-alloc: a stress run shows no `malloc`/`free` between the first and last block of a sequence's forward pass.

**Time budget.** Four weeks. The Goto SGEMM kernel and the FlashAttention varlen kernel are the two big lifts (~40% and ~25% of slice-time respectively). The rest is plumbing.

**Dev-machine constraint, surfaced.** This project's primary target is x86 AVX-512+VNNI; the developer host is an Apple M3 Pro (ARM64 NEON, no AVX-512, no AMX). The plan treats AVX-512 as the canonical path the perf gate is measured against, and a NEON path (or Accelerate `cblas_sgemm` wrapper) as a *dev fallback* — sufficient to iterate on attention design, arena layout, and threading locally, but not a co-equal supported backend. CLAUDE.md is explicit: "ARM/Apple Silicon as a co-equal target before v1 ships on x86" is a refuse. The Phase 1 gate measurement requires a Linux x86 instance with AVX-512+VNNI; identifying and standing up that machine is Slice 6's first task and a project dependency.

---

## Why slice this way

Phase 0 had a single dominant risk (numerical correctness) and was sliced around test infrastructure. Phase 1's risks are spread across four independent surfaces — kernel dispatch, GEMM microkernel design, attention API/numerics, threading — and each can break on its own. Each slice ends with the scalar reference still serving as ground truth, so any new SIMD path that produces wrong output is caught the moment it lands.

Two non-obvious ordering decisions:

1. **Dispatch (S1) and arena (S2) come before any SIMD code.** Both are pure refactors of existing-and-passing code. Landing them first means every subsequent slice plugs into stable infrastructure instead of inventing it. Get the boring plumbing done first; defer the interesting work.
2. **GEMM (S3) and varlen attention (S4) ship in that order, not in parallel.** Both touch `src/model.cpp`. Doing them sequentially keeps the orchestrator change-set small per slice and keeps parity tests as a clean bisection signal. GEMM is also the bigger speedup — landing it first gives an immediate visible win that confirms the dispatch infra is sound before we change the harder kernel.

---

## Dependency graph

```
S1 Dispatch + cpu_features ──┐
                             │
S2 Arena + Workspace ────────┤
                             │
                             ▼
                  S3 Goto-packed SGEMM (AVX-512 + NEON; libxsmm hookup)
                             │
                             ▼
                  S4 FlashAttention varlen (cu_seqlens interface)
                             │
                             ▼
                  S5 Thread pool + batch/FFN parallelism
                             │
                             ▼
                  S6 Microbench + x86 gate + retro
                             │
                             ▼
                          Phase 2 ──►
```

Parallelizable: S1 and S2 are independent and can be done in either order or concurrently. The remainder is a linear chain.

---

## Vertical slices

### Slice 1 — Kernel dispatch + CPU feature detect

Introduce the runtime-dispatch layer that every subsequent slice plugs into. No SIMD lands here. The deliverable is "every kernel is selected at load time from a function pointer table, and the scalar reference is the only entry in the table". Tests still pass.

**Tasks**
- **1.1** `src/kernels/cpu_features.cpp` + `include/esm_cpp/cpu_features.h`. `enum class Isa { Ref, Neon, Avx2, Avx512, Avx512Vnni, Amx };`. Function `Isa DetectIsa()` reads `__builtin_cpu_supports` on x86 and `__ARM_NEON`/`hwcap` on ARM. Env var `ESM_FORCE_ISA={ref,neon,avx512,...}` to override for testing. Print the selected ISA to stderr once at first model load if `ESM_LOG_ISA=1`.
- **1.2** Kernel signature header `include/esm_cpp/kernels.h` becomes a *dispatch facade*. The existing function names (`LinearRef`, `LayerNormRef`, …) remain as the scalar reference; new public entry points (`Linear`, `LayerNorm`, `Gelu`, `RopeApplyInplace`, `Attention`) call through a dispatch table set up at first use. CLAUDE.md says no SIMD intrinsics outside `src/kernels/`; the facade enforces that.
- **1.3** Reorganize `src/kernels/*.cpp` so the scalar reference lives behind `#ifdef ESM_KERNEL_REFERENCE` *within the same file as the dispatched entry point*, per CLAUDE.md's "every SIMD kernel has two implementations in the same `.cpp`". The reference TU is compiled with `-DESM_KERNEL_REFERENCE` and exposes the `*Ref` symbols; the dispatch facade picks the reference when ISA is `Ref` or no SIMD impl is registered for the host.
- **1.4** CMake: split `esm_cpp_core` into the orchestration TUs (model.cpp, tokenizer.cpp, io/) plus per-ISA `OBJECT` libraries (`esm_cpp_kernels_ref`, `esm_cpp_kernels_avx512`, `esm_cpp_kernels_neon`) with TU-specific compile flags. AVX-512 TUs get `-mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma`; NEON TU gets the platform default (ARMv8 baseline). Each ISA TU compiles only on the matching host (`CMAKE_SYSTEM_PROCESSOR` check). CLAUDE.md is explicit that target-attribute sprinkling is forbidden — flag isolation via separate TUs is the load-bearing decision here.
- **1.5** CI matrix gains an explicit Linux/x86 runner with `march=skylake-avx512` (or AVX-512 emulation via SDE on a non-AVX-512 GitHub runner as a fallback so the AVX-512 TU at least *compiles* on every PR). The new runner runs `ESM_FORCE_ISA=ref ctest` and a Phase 0 parity sample to keep the reference honest, and `ESM_FORCE_ISA=avx512 ctest` once Slice 3 ships.
- **1.6** A new `tests/cpp/test_dispatch.cpp` GoogleTest: with `ESM_FORCE_ISA=ref` every kernel uses the reference path; with the host's best ISA selected the same kernels still match the reference within FP32 tolerance (this becomes the cross-check harness that S3/S4 will populate).

**Acceptance criteria**
- `cmake --build build && ctest` is green on both macOS-arm64 (NEON TU compiles, dispatch returns `Isa::Neon` but selects reference impl until S3 registers one) and Linux-x86 (AVX-512 TU compiles, dispatch returns `Isa::Avx512` but selects reference impl).
- `ESM_FORCE_ISA=ref python -m pytest tests/python/test_against_hf.py -v` still hits the Phase 0 envelope numbers.
- `clang-tidy` clean on the new files; no SIMD intrinsics outside `src/kernels/`.

**Verification**
Push to a branch; CI matrix green; `nm build/libesm_cpp_kernels_avx512.a | c++filt | grep avx512` shows AVX-512 symbols compiled in (but unused) on the Linux runner.

**Out of slice**
Any actual SIMD code. Any thread-pool work. Any arena work.

---

### Slice 2 — Arena allocator + Workspace

Eliminate per-forward allocations. Today `Model::ForwardWithHiddenStates` constructs ~10 `std::vector<float>` per call (see model.cpp lines 301–313). Move them into a `Workspace` owned by `Model` (mutable member), document the non-reentrancy precondition, and reset the arena cursor at the top of each forward. No SIMD; the goal is a clean memory profile and a clean spot for S3/S4 to allocate from.

**Tasks**
- **2.1** New `include/esm_cpp/workspace.h` declares `esm::Workspace`. It owns a single `std::vector<std::byte>` backing buffer (sized at construction time, growable on first overflow) and exposes `T* allocate<T>(std::size_t n, std::size_t align = alignof(T))` (bump-allocator) and `void reset()`.
- **2.2** Model gains `mutable Workspace ws_` and a constructor-time `ws_.reserve(estimate_for(cfg, max_seqlen=1024))`. The estimate is computed from `L_max * (3d + d + 4d + ffn + …)` plus a small slack — keep it as one helper function so the bound is auditable.
- **2.3** Refactor `Model::ForwardWithHiddenStates` to (a) call `ws_.reset()` at entry, (b) pull every scratch buffer from `ws_.allocate<float>(…)` instead of `std::vector<float>(…)`, (c) keep the `hidden` buffer pulled from `ws_` as well. Return-value `logits` is still a `std::vector<float>` (Python boundary needs to own it).
- **2.4** Document `Model::Forward` non-reentrancy: "Not thread-safe; one call per `Model` instance at a time. Phase 3's scheduler will introduce a per-thread Workspace." Add a debug-build `assert(!ws_.in_use_)` flag set/cleared by RAII on the forward call.
- **2.5** A new test `tests/cpp/test_arena.cpp` that allocates a `Model`, runs `Forward` 10 times with different sequence lengths, and asserts (via a custom `new`/`delete` interposer in debug builds, or via `mtrace` on Linux, or simply via an arena `bytes_high_water` getter) that no growth happens after the first forward.
- **2.6** Phase 0 parity tests re-run; tolerances unchanged. The summation order in the existing scalar kernels is untouched, so numerics must be identical.

**Acceptance criteria**
- All Phase 0 tests green with no tolerance changes.
- New `test_arena.cpp` shows zero arena growth after the first forward at the configured max sequence length.
- `Workspace::bytes_allocated()` reported in a `print_workspace_stats()` debug helper.
- ASan build of the parity tests stays clean — arena alignment and bounds tested implicitly.

**Verification**
`ctest --test-dir build-dbg` (ASan) green. `python -m pytest tests/python/test_against_hf.py` matches Phase 0 envelope numbers exactly (bit-identical, since summation order didn't change).

**Out of slice**
Thread-safe arenas (Phase 3). Workspace ownership exposure to Python (Phase 1 keeps it invisible).

---

### Slice 3 — Goto-packed SGEMM (AVX-512 primary, NEON fallback)

The big lift. Replace `LinearRef` (the 3-loop matmul that is currently the bottleneck — Phase 0 retro: "200 × 1920 × 480 each, ~20s per 35M forward") with a real microkernel.

**Tasks**
- **3.1** Pick the register block. SPEC suggests `24×16` or `16×32` for AVX-512. Recommend `16×32` (16 rows × 32 columns of FP32 = 16 zmm registers used, 16 still free for the K-prefetch loop). One M_R × N_R inner kernel emits one 16-row, 32-col panel of C, accumulating 32 FMAs per K step into 16 zmm accumulators (two `vfmadd231ps` per K iteration). This is the salykova / yzhaiustc reference shape and avoids running out of registers.
- **3.2** Goto packing: pack A into `M_C × K_C` panels in L2 (target `M_C ≈ 256`, `K_C ≈ 512`), pack B into `K_C × N_C` panels in L3 (target `N_C ≈ 4096`). Macrokernel loops (j → i → k_c) call the microkernel on packed tiles. Reference: `salykova.github.io/matmul-cpu` and `yzhaiustc/Optimizing-DGEMM-on-Intel-CPUs-with-AVX512F`.
- **3.3** `src/kernels/gemm_fp32_avx512.cpp` (separate TU; `-mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma`):
    - Microkernel `gemm_kernel_16x32(const float* a_packed, const float* b_packed, float* c, int kc, int ldc, float beta)` written in `_mm512_*` intrinsics. Beta=0 path zeros C and accumulates; beta=1 adds (used for bias add via initial `_mm512_loadu_ps(bias_row); vfmadd_…`).
    - Pack routines `pack_a_16(const float* A, int M, int K, int lda, float* A_packed)` and `pack_b_32(const float* B, int K, int N, int ldb, float* B_packed)`. B is stored row-major `[N, K]` in our convention (PyTorch out-features × in-features); the pack routine transposes-and-tiles into the K-fast layout the microkernel expects. The bias is applied as the C initializer in the first K-pass.
    - Top-level `Linear` function fills C = A @ W^T + bias by walking the M, N, K_C tiles.
- **3.4** `src/kernels/gemm_fp32_neon.cpp` (separate TU; ARMv8 baseline, optional `+sve2` later):
    - Two implementation choices; pick **(b)** for Phase 1 and document the tradeoff:
      - (a) hand-written 8×4 or 4×4 NEON microkernel with `vfmaq_f32`
      - (b) Accelerate's `cblas_sgemm` wrapped as a `Linear` implementation
    - Rationale for (b): Accelerate is Apple-tuned and not what we ship on x86, but it gets the orchestrator and parity tests fast enough on the dev host to iterate at human speed. CLAUDE.md permits this as a *dev fallback*. We are not staffing a NEON microkernel — that's Phase 2-or-later if we ever co-elevate ARM.
    - CMake links `"-framework Accelerate"` on Darwin; on Linux ARM (Graviton, not currently a target) this TU is skipped and the dispatch falls back to `Ref`.
- **3.5** Register both with the dispatch facade from S1. `Linear` becomes:
    ```cpp
    // dispatch facade in src/kernels/gemm_fp32.cpp
    void Linear(const float* A, const float* W, const float* bias, float* C,
                int M, int N, int K) {
      switch (current_isa()) {
        case Isa::Avx512: return gemm_fp32_avx512::Linear(A, W, bias, C, M, N, K);
        case Isa::Neon:   return gemm_fp32_neon::Linear  (A, W, bias, C, M, N, K);
        default:          return LinearRef(A, W, bias, C, M, N, K);
      }
    }
    ```
- **3.6** `libxsmm` integration. FetchContent at a pinned tag (`v1.17` or later that has the `LIBXSMM_TARGET_ARCH=spr` target). Build with `-DLIBXSMM_DEFAULT_CONFIG=1`. Use `libxsmm_smmdispatch` at `Model::load` time to JIT a microkernel for each of the four critical shapes pre-bound at the model's `d` value; fall back to the hand-written kernel for shapes that the JIT misses. This is a *secondary* path, not the primary one — the hand-written 16×32 microkernel must pass the gate on its own. libxsmm exists to cover the small-shape tail (lm_head `[L, 33]`, anything where the M dimension is below 32).
- **3.7** Microbenchmarks in `bench/bench_gemm.cpp` (Google Benchmark, vendored via FetchContent). Cover the four critical shapes for `d ∈ {320, 640, 1280, 2560}` at `B·L ∈ {300, 4800, 8192}`. Report GFLOP/s and a `bench_gemm.json` artifact. Add a comparator script that reads MKL's `dgemm`/`sgemm` benchmark from the gate machine and computes the percentage-of-MKL number.
- **3.8** Correctness tests grow: shape-sweep matrix `M ∈ {1, 17, 64, 300, 4800}`, `N ∈ {33, 320, 1280, 5120}`, `K ∈ {320, 1280, 5120}`, random A/W/bias, `allclose(rtol=1e-6, atol=1e-5)` against `LinearRef`. Catches packing-offset bugs and tail-handling errors at non-multiple-of-block dimensions.
- **3.9** Re-run layer-by-layer parity against HF. With FMA + Goto packing the summation order is closer to BLAS than the 3-loop scalar reference. Measure the new logits/hidden envelope on 100 sequences for both 8M and 35M; commit the new (tighter) tolerances to `tests/python/test_against_hf.py` if they improve. If the Phase 0 envelope holds *and* the gate logits hit `< 1e-4`, mark the SPEC §2 footnote about scalar-vs-BLAS deviation as closed.

**Acceptance criteria**
- AVX-512 microbench on the gate machine reaches ≥80% of MKL on all four critical shapes for `d ∈ {640, 1280, 2560}`. The 320 shape is allowed lower (`≥65%`) because AVX-512 register pressure suffers at very narrow K — document the floor if hit.
- ESM-2-650M single-thread forward on AVX-512 instance hits ≥4× the Phase 0 reference (650M was never run at Phase 0; baseline is recomputed on first run and recorded).
- 35M forward on the dev machine (NEON via Accelerate) hits ≥10× the Phase 0 reference. Acceptance is purely "iteration speed is OK", not a perf target.
- All Phase 0 parity tests green. Logits envelope tightened — target `< 1e-4` absolute, fallback gate `< 5e-4` documented if FMA reordering doesn't close the gap.

**Verification**
`ctest -R Linear`; `./build/bench/bench_gemm --benchmark_out=results.json`; `python -m pytest tests/python/test_against_hf.py -v` on the AVX-512 instance. Compare against an MKL run (`ldd` the cblas_sgemm test) on the same instance.

**Out of slice**
INT8 GEMM (Phase 2). AMX path (Phase 2). VNNI (Phase 2). Per-thread parallel GEMM (S5). Auto-tuning of `M_C, K_C, N_C` per CPU (Phase 1.5 if measured to matter).

---

### Slice 4 — FlashAttention-style packed varlen attention

Replace `AttentionRef` and its `[H, L, dh]` interface with a kernel that takes packed Q/K/V plus `cu_seqlens[B+1]`, tiles the softmax, and accumulates in FP32. The scheduler that *uses* `cu_seqlens` for packing arrives in Phase 3; this slice lays down the kernel and the interface so the rest of the codebase is ready.

**Tasks**
- **4.1** Define the interface in `include/esm_cpp/kernels.h`:
    ```cpp
    void AttentionVarlen(const float* q,           // [T, H, dh]
                         const float* k,
                         const float* v,
                         const int* cu_seqlens,    // [B+1]
                         int batch_size,
                         int max_seqlen,
                         int num_heads,
                         int head_dim,
                         float* out);              // [T, H*dh]
    ```
    `T = cu_seqlens[B]`. For `B=1`, `cu_seqlens = {0, L}` and the kernel collapses to the single-sequence case.
- **4.2** `src/kernels/attention_varlen_ref.cpp` — scalar reference, FP32 softmax accumulator (already exists in spirit as `AttentionRef` but with the old layout). Tile size is `1` (no tiling) — this is the oracle for the vectorized paths. Add a layer-0 parity test against the existing HF goldens with the new interface.
- **4.3** `src/kernels/attention_varlen_avx512.cpp` (separate TU; AVX-512 flags):
    - Block-streaming inner loop over `B_C = 64` queries × `B_R = 64` keys (FlashAttention-2 style); within a block accumulate `S = Q · K^T`, take rowwise running max, rescale online (`exp(s - m) * old_sum`), accumulate `O += softmax_block @ V`. `m, l, O` rows kept in FP32. CLAUDE.md is explicit: FP32 accumulator inside the inner block is mandatory.
    - Reuse the GEMM packing primitives from S3 for the per-block `Q · K^T` and `softmax · V` matmuls. Block sizes chosen so each block's working set (`Q[B_C × dh] + K[B_R × dh] + V[B_R × dh] + S[B_C × B_R] + O[B_C × dh]`) fits in L2 for the `dh ∈ {16, 24, 32, 64, 128}` family.
    - Attention mask: integrate as `-inf` adders into the `S` block for any column whose `cu_seqlens` bucket is pad. (For B=1 there is no pad inside the sequence; the entire `T` dimension is real tokens.)
- **4.4** `src/kernels/attention_varlen_neon.cpp` — dev-fallback NEON implementation. Same tiled structure, scalar inner FMA (NEON `vfmaq_f32`); softmax in scalar with FP32 accumulator. Goal: iterate on the orchestrator without crawling.
- **4.5** Refactor `src/model.cpp` to (a) write Q/K/V projections directly in `[L, H, dh]` layout — eliminate `SplitHeads` per the Phase 0 retro carry-forward, (b) build `cu_seqlens = {0, L}` for the B=1 case, (c) call the new `AttentionVarlen` dispatch facade, (d) reuse the new arena for the kernel's per-block scratch.
- **4.6** Tests:
    - `tests/cpp/test_attention_varlen.cpp` — single-sequence and two-sequence (B=2) packed inputs vs the scalar reference. `allclose(rtol=1e-6, atol=1e-5)` per CLAUDE.md FP32 tolerance.
    - Pad-handling: a 2-sequence packed batch with the second sequence padded out via `cu_seqlens` must produce identical output for the first sequence regardless of the second's contents.
    - HF parity tests re-run end-to-end. Layer-by-layer `attn_output` golden (captured at Phase 0) must match within the new envelope.
- **4.7** Microbench `bench/bench_attention.cpp` — single-sequence (B=1) at L ∈ {64, 128, 256, 512, 1024} and varlen (B=16, L ranges 64–512 packed). Report tokens/sec and FLOPs/cycle.

**Acceptance criteria**
- AVX-512 attention varlen at L=300, H=20, dh=64 (650M shape) reaches ≥70% of the SGEMM throughput on `[L, dh] × [dh, L]` and `[L, L] × [L, dh]` baselines — i.e., the softmax/rescale overhead is bounded.
- Per-head, per-block correctness vs scalar reference within FP32 tolerance on the shape sweep.
- HF parity holds with logits envelope ≤ the Slice 3 envelope (varlen attention must not regress numerics).

**Verification**
`ctest -R AttentionVarlen`; `./build/bench/bench_attention`; the end-to-end parity test on the AVX-512 instance under the new attention path.

**Out of slice**
INT8 K/V attention (Phase 2). Causal masking (ESM is encoder-only — bidirectional only). Cross-attention. Sliding-window attention.

---

### Slice 5 — Thread pool + batch / FFN-4d parallelism

Single-threaded SIMD won't hit the 2× HF target at batch 16 — the gate explicitly assumes parallelism. Introduce a thread pool, pick the two parallelism axes the SPEC permits (batch and FFN-4d), and wire them in. CLAUDE.md: "No raw `std::thread` in the forward path. No `std::async`. No locks in the inner loop."

**Tasks**
- **5.1** `src/threading/thread_pool.cpp` + `include/esm_cpp/thread_pool.h`. A small (~200 LOC) hand-rolled pool: N worker threads, a single MPMC task queue (lock-free if it's not too much trouble; condvar-based otherwise — the queue is touched only at task submission, not in the inner loop). Public API: `void parallel_for(int begin, int end, int grain, std::function<void(int, int)> body)` that splits the range into chunks of size ≥ `grain` and dispatches one per worker. The worker function is `[](int chunk_begin, int chunk_end)` so the body decides what to do per-chunk.
- **5.2** `Model` gains a `thread_pool& pool_` reference. Pool is owned by a process-global singleton initialized lazily on first `Model::load`. Size from `ESM_NUM_THREADS` env var; default to physical-core count (from `std::thread::hardware_concurrency()` minus SMT-aware adjustments where detectable — see llama.cpp's `cpu_get_num_physical_cores` for the cross-platform recipe).
- **5.3** FFN-4d parallelism. Both fc1 (`[L, d] → [L, 4d]`) and fc2 (`[L, 4d] → [L, d]`) split along the `4d` axis. Each thread computes one slab of the output columns. Determinism: fix the partition (round-robin by row-strip of N_C) so the same model + thread count always sums in the same order.
- **5.4** Batch parallelism. For the gate B=16, each of the 16 sequences is an independent transformer pass through the per-block QKV/attn/out_proj/FFN until they meet at the `cu_seqlens`-packed attention call. Parallelize at the *outer* batch loop in `Model::ForwardBatch` (introduce this new entry point; B=1 forward stays for back-compat). Each worker takes one or two sequences; FFN-4d parallelism does NOT compose with batch parallelism naively (would oversubscribe). Pick one axis per call based on `B * grain >= num_threads` → batch axis; else FFN axis.
- **5.5** Per-thread `Workspace`. Each worker thread holds a thread-local arena, sized at pool init. Phase 1 Model still has a single `ws_` (B=1 path), but `ForwardBatch` walks per-thread workspaces. Documented in `Workspace` header.
- **5.6** Parity tests run at thread counts {1, 2, 4, physical_cores}. Single-threaded result is the reference; multi-thread must match within the FFN-partition reorder tolerance (target: bit-identical for the FFN-4d partition since the *sum* over K is unchanged when partitioning over N; will be bit-identical in practice).
- **5.7** Stress test: 1000 forwards on a single Model from a pool of producer threads — confirms thread-safety of `ForwardBatch` (each thread has its own workspace). Phase 0 `Forward` (mutable shared `ws_`) remains documented non-reentrant.

**Acceptance criteria**
- Single-threaded throughput on the gate machine = Slice 4 throughput (no regression from thread-pool overhead at thread_count=1).
- ≥1.5× throughput at thread_count=4 vs thread_count=1 on 650M batch 16 (linear up to ~6 threads at this size; beyond that, memory bandwidth dominates).
- Layer-by-layer parity holds under every tested thread count.
- ASan/TSan green on the stress test.

**Verification**
`ESM_NUM_THREADS=4 python -m pytest tests/python/test_against_hf.py`; the new perf microbench reports near-linear scaling up to physical-core count.

**Out of slice**
NUMA-aware placement (out-of-scope on single-socket gate machine). Hyperthreading control. Work-stealing pool (use static round-robin — re-evaluate if benchmarks show idle workers).

---

### Slice 6 — Microbench harness, x86 gate measurement, retrospective

The accounting slice. Stand up the gate machine, install the comparators (PyTorch + HF + MKL), run the numbers, write the retro.

**Tasks**
- **6.1** Identify and stand up the gate machine. Recommendation: AWS `c7i.4xlarge` (8 physical cores Sapphire Rapids, AVX-512+VNNI+AMX; ~$0.71/h us-east-1) or `c7i.metal-24xl` for the 80% MKL measurement (full socket, MKL_NUM_THREADS controlled). Document the exact SKU, kernel, glibc, and compiler versions in `docs/benchmarks.md` (start of the file; Phase 3 will extend it).
- **6.2** Install Intel oneAPI base toolkit on the gate machine; record `mkl_dgemm`/`mkl_sgemm` numbers on the four shapes via the comparator script. Pin the MKL version in `docs/benchmarks.md`. If MKL licensing is friction, OpenBLAS-AVX512 from apt is the fallback comparator and the gate restates as "≥80% of best available open-source SGEMM."
- **6.3** Install PyTorch + transformers at the same versions pinned in `pyproject.toml` (Phase 0 already pinned `transformers`; pin `torch` here). Build the comparison script `python/esm_cpp/bench/compare.py` (sketched in SPEC §3 commands) that runs HF `EsmModel(..., attn_implementation="eager")` on the same input batch as esm.cpp and reports throughput. p50/p99 latency over ≥30 warmup runs after a 5-run warm-up; deterministic seeds; `OMP_NUM_THREADS` and `MKL_NUM_THREADS` pinned. Verify the script is stable to ±2% across re-runs before committing the comparison numbers.
- **6.4** Run the gate. Three configurations:
    - **A (perf):** 650M, batch 16, 300aa, AVX-512+VNNI, physical-core count threads. Record throughput (seq/sec), p50/p99 latency, memory high-water.
    - **B (microkernel):** the four critical shapes, `d ∈ {320, 640, 1280, 2560}`, single-threaded. Record GFLOP/s and the % of MKL.
    - **C (correctness on gate hardware):** 100-sequence parity for 8M, 35M, *and* 650M (Phase 0 only validated 8M and 35M; Phase 1 first time 650M runs the parity suite end-to-end on real SIMD). Logits/hidden envelope captured for the retro.
- **6.5** `notes/phase1.md` retrospective per SPEC §2. One screen of prose: what shipped, what didn't and why, measured numbers (A/B/C above), surprises encountered (likely candidates: AVX-512 microkernel register pressure on `d=320`, FlashAttention block-size tuning at small dh, thread-pool startup cost on first forward), deviations from this plan, what to carry into Phase 2.
- **6.6** Update README with Phase 1 status, the perf headline, and reproduction commands. Update `tests/python/test_against_hf.py` tolerances to the Phase 1 floor.
- **6.7** Decide on the original `< 1e-4` logits gate. Three possible outcomes, document whichever lands:
    1. We hit `< 1e-4` on 8M, 35M, and 650M — close the SPEC §2 footnote, restore the original gate text.
    2. We hit `< 1e-4` on 8M but accumulated drift on 650M (33 layers) keeps logits at `~3e-4` — adopt a model-size-aware gate.
    3. SIMD reordering still has a floor at `~5e-4` — document the realistic floor and the reason (compares to FMA-using PyTorch eager which is itself non-deterministic across BLAS backends).

**Acceptance criteria**
- Gate met: ≥2× HF on 650M batch-16 300aa with AVX-512+VNNI; SGEMM ≥80% MKL on the four shapes (or restated against the documented comparator).
- `notes/phase1.md` committed alongside the code that closes the phase.
- `docs/benchmarks.md` v0 exists with gate machine SKU and reproduction commands.
- All four checkpoints (below) cleared.

**Verification**
The Phase 1 close-out PR runs the full gate on the chosen instance, links to a `bench_results.json` artifact, and contains the retro. Reviewer (human) reads `notes/phase1.md` and confirms it covers the four required sections (what shipped, surprises, deviations, carry-forward).

**Out of slice**
INT8 quant numbers (Phase 2). PPPL throughput / ProteinGym benchmarks (Phase 2 quality gate). Public benchmark page formatting (Phase 3 ship).

---

## Checkpoints

- **Checkpoint A (after Slice 2).** Dispatch facade and arena live. Phase 0 parity numbers bit-identical, AVX-512 TU compiles on the Linux runner. Pause: confirm CI matrix green on both arch families before any SIMD code lands. Cheap to fix CMake/flag issues here; expensive after S3.
- **Checkpoint B (after Slice 3).** SGEMM SIMD lands. The first numeric improvement (and the first risk of numeric *regression*). Compare layer-by-layer envelope against Phase 0; if the SIMD path is *worse* than scalar reference at any layer, stop and bisect — likely a packing-offset bug or a beta-init mismatch. Decide here whether the `< 1e-4` logits gate is recoverable on 8M.
- **Checkpoint C (after Slice 4).** Varlen attention lands. The `cu_seqlens` interface is now load-bearing for Phase 3 — review the signature with that in mind. ESME's `flash_attn_varlen_func` is the closest reference; cross-check that we haven't painted ourselves into a corner that the Phase 3 scheduler can't consume.
- **Checkpoint D (after Slice 5).** Thread pool lands. Stress-test parity at every thread count actually used in CI. If non-determinism shows up in the FFN-4d path despite the fixed partition, fix it here — Phase 2's INT8 path will multiply the consequences.

---

## Risks specific to Phase 1

| Risk | Mitigation |
|---|---|
| Gate machine availability and cost. Phase 1's perf gate requires AVX-512+VNNI hardware that the dev host doesn't have. | Slice 6.1 stands up the gate machine. Use a spot-priced AWS `c7i` instance; budget ~$50 for the full gate run set. Can run weekly during development, full gate run only at slice boundaries. |
| MKL not available on the chosen gate machine (licensing, deployment friction). | Comparator falls back to OpenBLAS-AVX512; gate text restates as "≥80% of best available CPU SGEMM library on the gate machine." Document choice. |
| AVX-512 microkernel register pressure at `d=320` (smallest shape; 16×32 block underutilizes registers). | Allow `≥65%` of MKL at `d=320` in the acceptance criteria; document the floor. Phase 2 INT8 path widens the K dimension via VNNI and recovers the lost arithmetic intensity at small d. |
| FlashAttention numerical drift compared to the plain scaled-dot reference. FA's online rescale changes the summation order again. | Layer-0 `attn_output` golden (captured in Phase 0) is the bisection oracle. Tolerance for varlen vs scalar reference: FP32 `rtol=1e-6 atol=1e-5` per CLAUDE.md. |
| Thread non-determinism breaking parity tests. | FFN-4d partition is a *column* split — each output element is still summed over the same K-axis. So the partition is bit-identical regardless of thread count. The batch-axis parallelism is per-sequence — also bit-identical. If we observe non-determinism, it's a bug in the pool, not in the math. |
| libxsmm integration friction (build system, ARM support). | libxsmm is *secondary*. The hand-written 16×32 microkernel must pass the gate on its own. If libxsmm is a nightmare to integrate, defer to Phase 2 and document. |
| Phase 0's "FP32 max-abs-diff < 1e-4" gate is not actually recoverable on 650M (33 layers compound drift). | Slice 6.7 has three documented landing options. The honest answer is whatever the measured floor is, not a fixed promise. |
| Forward becoming non-reentrant via mutable `ws_` blocks downstream multi-threaded callers (Python users who already parallelize forwards from multiple threads). | Docs are explicit. `ForwardBatch` is reentrant; `Forward(B=1)` is not. Phase 3's scheduler is the formal multi-arena path. |
| Build complexity from per-ISA OBJECT libraries with TU-specific flags. CMake gets fiddly. | Land 1.4 with an explicit Linux CI run that builds *only* `esm_cpp_kernels_avx512` (no test execution) — confirms compile path works before any actual AVX-512 code lands. |
| Local dev throughput on NEON via Accelerate is *too good* and masks bugs that only show on AVX-512. | The dispatch test (`test_dispatch.cpp`) cross-checks every ISA's output against the scalar reference. Any ISA-specific numeric divergence beyond `rtol=1e-6 atol=1e-5` fails the test. |

---

## Out of scope for Phase 1 (explicit reminders)

- No INT8 quantization, no SmoothQuant, no observer. Phase 2.
- No AMX, no VNNI. Phase 2.
- No GGUF reader/writer. Phase 3.
- No `cu_seqlens`-packing scheduler. The *interface* lands in S4; the scheduler that fills it lands in Phase 3. Phase 1 always passes B=1 or `cu_seqlens = {0, L_0, L_0+L_1, …}` from a naive packer.
- No 150M / 3B / 15B parity. 650M is the gate; larger models become Phase 2 once INT8 lands.
- No backward pass, training, LoRA. Out forever (ESME's niche).
- No ARM/NEON as a co-equal target. The NEON code path (or Accelerate wrapper) exists as a dev fallback only.
- No public Python API additions beyond what Phase 0 already exposes. Workspace, thread pool size, and ISA selection are env-var-only knobs in Phase 1.
- No pre-compaction or AOT-tuned microkernels. The runtime dispatch + libxsmm JIT covers Phase 1.
