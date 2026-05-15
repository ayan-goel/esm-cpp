# Phase 1 — Task List

Companion to [plan.md](plan.md). Tick tasks as they land. Acceptance criteria and verification steps live in plan.md; this file is the running ledger.

## Slice 1 — Kernel dispatch + CPU feature detect

- [x] **1.1** `src/kernels/cpu_features.cpp` + `include/esm_cpp/cpu_features.h` (`Isa` enum, `DetectIsa()`, `ESM_FORCE_ISA` / `ESM_LOG_ISA` env vars)
- [ ] **1.2** Kernel dispatch facade in `include/esm_cpp/kernels.h` (`Linear`, `LayerNorm`, `Gelu`, `RopeApplyInplace`, `Attention` entry points pick reference/AVX-512/NEON at first use)
- [ ] **1.3** Reorganize `src/kernels/*.cpp`: scalar reference behind `#ifdef ESM_KERNEL_REFERENCE` in same file as dispatched entry point
- [ ] **1.4** CMake per-ISA `OBJECT` libraries (`esm_cpp_kernels_ref`, `esm_cpp_kernels_avx512`, `esm_cpp_kernels_neon`) with TU-specific flags; arch-gated by `CMAKE_SYSTEM_PROCESSOR`
- [ ] **1.5** CI matrix: Linux/x86 runner with AVX-512 (real or SDE-emulated); `ESM_FORCE_ISA=ref` and `ESM_FORCE_ISA=avx512` both green
- [ ] **1.6** `tests/cpp/test_dispatch.cpp` cross-checks every registered ISA against the scalar reference

## Slice 2 — Arena allocator + Workspace

- [ ] **2.1** `include/esm_cpp/workspace.h` declares `esm::Workspace` (bump allocator, `allocate<T>(n, align)`, `reset()`)
- [ ] **2.2** `Model` gains `mutable Workspace ws_`; sized at construction from `cfg` + max_seqlen=1024 estimate
- [ ] **2.3** Refactor `Model::ForwardWithHiddenStates` to pull every scratch buffer from `ws_`; `ws_.reset()` at entry
- [ ] **2.4** Document non-reentrancy; debug `assert(!ws_.in_use_)` flag (RAII-managed)
- [ ] **2.5** `tests/cpp/test_arena.cpp` — 10 forwards at varying L, no arena growth after first
- [ ] **2.6** Phase 0 parity tests re-run; expect bit-identical numerics (no summation order change)
- [ ] **Checkpoint A** — dispatch + arena live, CI matrix green on both arch families

## Slice 3 — Goto-packed SGEMM (AVX-512 + NEON dev fallback)

- [ ] **3.1** Lock register block size (recommend 16×32 for AVX-512) and macrokernel tile params (M_C≈256, K_C≈512, N_C≈4096)
- [ ] **3.2** Goto packing routines: `pack_a_16` (M-major, K-fast), `pack_b_32` (N-major, K-fast); reference `salykova.github.io/matmul-cpu`
- [ ] **3.3** `src/kernels/gemm_fp32_avx512.cpp` — microkernel `gemm_kernel_16x32` in `_mm512_*` intrinsics; macrokernel walks tiles; bias applied at C-init
- [ ] **3.4** `src/kernels/gemm_fp32_neon.cpp` — Accelerate `cblas_sgemm` wrapper (dev fallback)
- [ ] **3.5** Register both with the S1 dispatch facade; dispatch picks per `current_isa()`
- [ ] **3.6** `libxsmm` FetchContent at pinned tag; `libxsmm_smmdispatch` at `Model::load` for the four critical shapes; small-shape fallback
- [ ] **3.7** `bench/bench_gemm.cpp` (Google Benchmark) on the four shapes × `d ∈ {320, 640, 1280, 2560}` × `B·L ∈ {300, 4800, 8192}`
- [ ] **3.8** Correctness shape sweep against `LinearRef` (`allclose(rtol=1e-6, atol=1e-5)`); tail-handling at non-block-multiple dims
- [ ] **3.9** Re-run HF parity on 8M / 35M; tighten `final_logits_tol` if FMA + Goto closes toward `< 1e-4`
- [ ] **Checkpoint B** — SGEMM SIMD lands; bisect any numeric regression vs scalar; decide `< 1e-4` gate recoverability

## Slice 4 — FlashAttention varlen with `cu_seqlens`

- [ ] **4.1** Declare `AttentionVarlen` in `include/esm_cpp/kernels.h` (`q/k/v: [T, H, dh]`, `cu_seqlens: [B+1]`, `out: [T, H*dh]`)
- [ ] **4.2** `src/kernels/attention_varlen_ref.cpp` — scalar reference (tile size 1, FP32 softmax); HF golden cross-check
- [ ] **4.3** `src/kernels/attention_varlen_avx512.cpp` — FA-2-style block streaming (`B_C = B_R = 64`); FP32 `m, l, O`; reuses S3 packing primitives
- [ ] **4.4** `src/kernels/attention_varlen_neon.cpp` — dev fallback NEON
- [ ] **4.5** `src/model.cpp` refactor: Q/K/V projections in `[L, H, dh]` layout (kill `SplitHeads`), B=1 `cu_seqlens`, dispatch to `AttentionVarlen`, arena for per-block scratch
- [ ] **4.6** `tests/cpp/test_attention_varlen.cpp` — B=1, B=2 packed, padding isolation, vs scalar reference
- [ ] **4.7** `bench/bench_attention.cpp` — single-seq + varlen at L ∈ {64, 128, 256, 512, 1024}
- [ ] **Checkpoint C** — varlen attention lands; review `cu_seqlens` signature with Phase 3 scheduler in mind

## Slice 5 — Thread pool + parallelism

- [ ] **5.1** `src/threading/thread_pool.cpp` + `include/esm_cpp/thread_pool.h` (hand-rolled, ~200 LOC, `parallel_for(begin, end, grain, fn)`)
- [ ] **5.2** Process-global pool initialized at first `Model::load`; size from `ESM_NUM_THREADS`, default physical-core count (llama.cpp `cpu_get_num_physical_cores` recipe)
- [ ] **5.3** FFN-4d parallelism (fc1 and fc2 split along the 4d axis with a fixed partition for determinism)
- [ ] **5.4** Batch parallelism via new `Model::ForwardBatch`; dispatch axis selection (`B * grain >= num_threads` → batch; else FFN)
- [ ] **5.5** Per-thread `Workspace` in `ForwardBatch` (thread-local arena, sized at pool init)
- [ ] **5.6** Parity tests at thread counts {1, 2, 4, physical_cores} — must remain bit-identical given the fixed FFN-4d partition
- [ ] **5.7** TSan/ASan stress test: 1000 forwards from a multi-producer pool against one `Model`
- [ ] **Checkpoint D** — thread pool lands; non-determinism (if observed) is a pool bug, not a math bug

## Slice 6 — Microbench, x86 gate measurement, retrospective

- [ ] **6.1** Stand up gate machine (recommend AWS `c7i.4xlarge` Sapphire Rapids; document exact SKU, kernel, glibc, compiler in `docs/benchmarks.md`)
- [ ] **6.2** Install Intel oneAPI MKL on the gate machine; record MKL `sgemm` numbers on the four shapes; pin MKL version
- [ ] **6.3** `python/esm_cpp/bench/compare.py` — HF `EsmModel(attn_implementation="eager")` vs esm.cpp; p50/p99 latency, warm-up, deterministic seeds, threads pinned
- [ ] **6.4** Run the gate (three configs: perf, microkernel %-of-MKL, correctness on 8M/35M/650M)
- [ ] **6.5** `notes/phase1.md` retrospective (what shipped, surprises, deviations, carry-forward to Phase 2)
- [ ] **6.6** Update README with Phase 1 status and reproduction commands; commit Phase 1 tolerances to `test_against_hf.py`
- [ ] **6.7** Decide `< 1e-4` logits gate landing — one of: closed and restored; model-size-aware; documented floor

## Phase 1 done

- [ ] All slice gates green
- [ ] All checkpoints (A, B, C, D) cleared
- [ ] `notes/phase1.md` committed
- [ ] CI matrix green on `main` (Linux + macOS, x86 + ARM, AVX-512-on, AVX-512-off)
- [ ] Gate met on the x86 instance: ≥2× HF on 650M batch-16 300aa, SGEMM ≥80% of comparator on the four shapes
- [ ] Ready to start Phase 2 planning (INT8 + AMX)
