# esm.cpp — Specification

## 1. Objective

Build a CPU-first, INT8-quantized, continuously-batched C++ inference engine for the ESM-2 family of protein language models, distributed as a `pip install`-able Python package. The defensible niche is the intersection no existing project occupies: production-grade CPU inference + ahead-of-time W8A8 quantization + variable-length packed-batch scheduling for encoder-only PLMs.

**Target users.** Computational biologists and ML engineers running large-scale protein language model inference on CPU: deep mutational scanning (10⁴–10⁷ variants per protein), antibody developability screening (10⁵–10⁶ candidates), pseudo-perplexity evaluation on held-out sets, embedding extraction for downstream tasks. Specifically: academic labs without GPU clusters, on-prem regulated environments where cloud GPUs are restricted, and antibody-discovery teams whose workloads are throughput-per-dollar-bound rather than per-sequence-latency-bound.

**Non-objective.** Training, fine-tuning, LoRA, or any backward pass. ESME owns that on GPU. Encoder-only PLMs only — no decoder/generative models. No GPU backend in v1.

**Headline target.** ESM-2-650M, single-socket x86 with AVX-512+VNNI, beats HuggingFace `EsmModel` PyTorch eager FP32 throughput on `cu_seqlens`-packed batches of real OAS antibody sequences, with PPPL drift < 0.1 and ProteinGym Spearman drift < 0.01 vs FP32 reference.

**Models supported in v1.** ESM-2 at 8M, 35M, 150M, 650M, 3B. 15B is a v2 stretch goal (requires INT4 weight-only quant; bandwidth-bound on commodity hardware).

## 2. Phases and Deliverables

The project ships in four phases. Each phase has a build list, a measurable gate that must pass before advancing, and a written retrospective at `notes/phaseN.md`. The retro is committed alongside the code that closes the phase. Keep it short — one screen of prose, not a bullet farm — covering: what shipped, what didn't and why, measured numbers (perf and quality), surprises and deviations from this spec, and what to carry into the next phase.

### Phase 0 — FP32 Reference Forward (Weeks 1–2)

**Build**
- Tokenizer: 33-token alphabet in UR50 frequency order, `prepend_bos=True`, `append_eos=True`, `<null_1>=31`, max length 1024.
- Safetensors reader for HuggingFace ESM-2 checkpoints.
- FP32 forward graph: embed → `token_dropout` 0.88 rescale → per-layer (LN → QKV → Q-scale → half-then-half RoPE on Q/K → scaled-dot attention → out_proj → residual → LN → FFN GELU → residual) → final LN → tied `lm_head`.
- Layer-by-layer golden-tensor capture from HF for ESM-2-8M and 35M, stored in `tests/golden/`.
- C++ unit tests asserting `allclose(rtol=1e-3, atol=1e-3)` at every hidden layer.

**Gate to advance**
FP32 max abs diff < 1e-4 on final logits vs HF `EsmModel` on 100 random sequences, ESM-2-8M and 35M.

**Retrospective:** `notes/phase0.md`

### Phase 1 — SIMD FP32 Baseline (Weeks 3–6)

**Build**
- Runtime CPU feature detect and kernel dispatch (`src/kernels/cpu_features.cpp`).
- Goto-style packed SGEMM microkernel for AVX-512 (24×16 or 16×32 register block); link `libxsmm` for small shapes.
- FlashAttention-style packed-varlen attention with FP32 softmax accumulator, `cu_seqlens` interface (even if v1 scheduler isn't packing yet).
- Arena allocator; zero-allocation forward loop.
- Thread pool, parallelism on batch and FFN-4d dimensions only.
- Microbenchmarks (Google Benchmark) on the four critical shapes: `[B·L, d, 3d]`, `[B·L, d, d]`, `[B·L, d, 4d]`, `[B·L, 4d, d]` for `d ∈ {320, 640, 1280, 2560}`.

**Gate to advance**
≥2× HuggingFace PyTorch eager FP32 throughput on ESM-2-650M, batch 16, 300aa, single socket. SGEMM microkernels reach ≥80% of MKL on the four critical shapes.

**Retrospective:** `notes/phase1.md`

### Phase 2 — INT8 + AMX (Weeks 7–10)

**Build**
- Per-channel symmetric INT8 weight packing for all Linear layers.
- SmoothQuant offline calibration with α=0.5 default; α sweep on PPPL recorded in `notes/phase2.md`.
- 99.9th-percentile per-tensor activation observer calibrated on 256–1024 real UniRef50 sequences (length 100–500).
- AVX-512 VNNI INT8 microkernel using `VPDPBUSD`.
- AMX `TDPBSSD` path, runtime-gated via `cpuid` (`AMX-INT8` feature flag).
- Optional INT8 K/V in attention with FP32 softmax accumulator.
- Quality eval harness: exact pseudo-perplexity on a fixed 1000-sequence UniRef50 holdout; ProteinGym v1.3 zero-shot masked-marginal scoring.
- Sensitivity escapes kept higher precision: `lm_head.dense` and `lm_head.layer_norm` stay FP32; first transformer block's `fc1` input falls back to FP16 if PPPL drift > 0.2.

**Gate to advance**
PPPL drift < 0.1 and ProteinGym Spearman drift < 0.01 vs FP32 reference, on ESM-2-150M and 650M. Below 150M ships FP32/BF16 only.

**Retrospective:** `notes/phase2.md`

### Phase 3 — Continuous Batching + GGUF + Ship (Weeks 11–14)

**Build**
- `cu_seqlens` packing scheduler with iteration-level admission. Length-bucket only when packed-work imbalance > 20%.
- GGUF reader/writer with `LLM_ARCH_ESM` tag; standalone `convert_esm_to_gguf.py`.
- pip-installable `esm-cpp` Python package with pybind11 bindings. `py::call_guard<py::gil_scoped_release>` on every forward call.
- Public benchmark script (`python -m esm_cpp.bench.compare`) reporting PPPL throughput, ProteinGym Spearman, and p50/p99 latency against HuggingFace PyTorch eager.
- Reproduction guide at `docs/benchmarks.md` with hardware/SKU, core count, batch shape, and full reproduction commands.

**Gate to ship**
Reproducible public benchmark on a single-socket x86 with AVX-512+VNNI, beating HuggingFace PyTorch eager on PPPL throughput, with quality drift inside Phase 2 bounds. Numbers, commands, and SKU all in the repo.

**Retrospective:** `notes/phase3.md`

## 3. Commands

All builds via CMake + a Python frontend. The Python install is the user-facing surface; the C++ binaries are for developers.

| Purpose | Command |
|---|---|
| Configure + build (Release) | `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j` |
| Configure + build (Debug) | `cmake -B build-dbg -DCMAKE_BUILD_TYPE=Debug -DESM_SANITIZERS=ON && cmake --build build-dbg -j` |
| C++ unit + integration tests | `ctest --test-dir build --output-on-failure` |
| Python install (editable, from source) | `pip install -e ".[dev]"` |
| Python tests | `pytest tests/python -v` |
| Microbenchmarks (GEMM, attention) | `./build/bench/esm-microbench` |
| End-to-end benchmark vs HF | `python -m esm_cpp.bench.compare --model esm2_t33_650M --batch 16 --len 300` |
| Convert HF checkpoint to internal weights | `python -m esm_cpp.convert --hf facebook/esm2_t33_650M_UR50D --out weights/esm2_650m.gguf` |
| Calibrate INT8 (per-channel + SmoothQuant) | `python -m esm_cpp.quantize --weights weights/esm2_650m.gguf --calib data/uniref50_1k.fasta --alpha 0.5` |
| Quality gate (PPPL on UniRef50 holdout) | `python -m esm_cpp.eval.pppl --weights <path> --data data/uniref50_holdout.fasta` |
| Quality gate (ProteinGym v1.3) | `python -m esm_cpp.eval.proteingym --weights <path>` |
| Lint (C++) | `clang-tidy` via `cmake --build build --target lint` |
| Lint (Python) | `ruff check python/ tests/` |
| Format | `clang-format -i` + `ruff format` |

## 4. Project Structure

```
esm-cpp/
├── SPEC.md                      # this file
├── research-report.md           # background research
├── CMakeLists.txt               # top-level build
├── pyproject.toml               # Python package, pybind11, scikit-build-core
├── include/esm_cpp/             # public C++ headers (stable API)
│   ├── model.h                  # esm::Model load/forward
│   ├── tokenizer.h              # 33-token alphabet, prepend_bos/append_eos
│   ├── batch.h                  # cu_seqlens packed-batch type
│   └── version.h
├── src/                         # C++ implementation
│   ├── model.cpp                # transformer forward graph
│   ├── tokenizer.cpp
│   ├── kernels/
│   │   ├── gemm_fp32.cpp        # Goto-style packed SGEMM (AVX-512)
│   │   ├── gemm_int8.cpp        # VPDPBUSD microkernel
│   │   ├── gemm_amx.cpp         # AMX TDPBSSD path (runtime-gated)
│   │   ├── attention_varlen.cpp # FlashAttention-style, cu_seqlens, FP32 accum
│   │   ├── rope.cpp             # half-then-half (Llama/GPT-NeoX), NOT interleaved
│   │   ├── layernorm.cpp
│   │   ├── gelu.cpp             # tanh approximation (matches PyTorch)
│   │   └── cpu_features.cpp     # runtime CPUID detect
│   ├── quant/
│   │   ├── observer.cpp         # 99.9th-percentile per-tensor activation observer
│   │   ├── smoothquant.cpp      # offline alpha-migration
│   │   └── pack.cpp             # per-channel symmetric INT8 weight packing
│   ├── io/
│   │   ├── gguf.cpp             # GGUF reader/writer (LLM_ARCH_ESM)
│   │   └── safetensors.cpp      # HF source format reader
│   └── sched/
│       └── scheduler.cpp        # cu_seqlens packing scheduler
├── python/esm_cpp/              # pybind11 bindings + pure-Python helpers
│   ├── __init__.py
│   ├── _core.cpp                # pybind11 module
│   ├── convert.py               # HF -> GGUF converter
│   ├── quantize.py              # calibration + quant driver
│   ├── bench/compare.py
│   └── eval/
│       ├── pppl.py
│       └── proteingym.py
├── tests/
│   ├── cpp/                     # GoogleTest: kernels, tokenizer, gguf round-trip
│   ├── python/                  # pytest: end-to-end vs HF EsmModel
│   └── golden/                  # captured HF hidden states (8M, 35M; small)
├── bench/                       # Google Benchmark microbenchmarks
│   ├── bench_gemm.cpp
│   └── bench_attention.cpp
├── third_party/                 # vendored or fetched via CMake FetchContent
│   ├── libxsmm                  # small-shape GEMM (link, don't reimplement)
│   └── pybind11
├── tools/
│   └── capture_golden.py        # produces tests/golden/*.npz from HF
├── notes/                       # phase retrospectives (one per phase)
│   ├── phase0.md
│   ├── phase1.md
│   ├── phase2.md
│   └── phase3.md
└── docs/
    ├── architecture.md
    ├── quant-recipe.md
    └── benchmarks.md
```

## 5. Code Style

**C++20.** No exceptions in hot paths (kernels) — kernels return error codes; the model-level API may throw at load time. RAII everywhere; no raw `new`/`delete`. `snake_case` for functions and variables, `PascalCase` for types — matches ggml/llama.cpp conventions so contributors crossing over feel at home. Public headers in `include/esm_cpp/`, namespaced under `esm::`. `clang-format` style: Google, column limit 100. `clang-tidy` enforced in CI with `bugprone-*`, `cert-*`, `performance-*`, `readability-identifier-naming`.

**Kernel discipline.** SIMD kernels live in `src/kernels/` and are dispatched at load time via a runtime feature-detect table in `cpu_features.cpp`. No SIMD intrinsics outside `src/kernels/`. Each kernel has (a) a scalar reference implementation in the same file, behind `#ifdef ESM_KERNEL_REFERENCE`, used by tests, and (b) one or more vectorized implementations. The reference and vectorized paths are exercised by the same unit tests with `allclose` tolerances.

**Memory.** Allocate once per model load; reuse scratch buffers across forward passes via an arena allocator. No `malloc`/`free` in the inner forward loop. Weights are mmap'd from GGUF (zero-copy for FP32; INT8 paths read packed blocks directly).

**Python.** Python 3.10+. `ruff` for lint + format. Type-annotated public APIs; `mypy --strict` on `python/esm_cpp/`. Pybind11 bindings expose only stable C++ headers (`include/esm_cpp/`), never internal kernels.

**Comments.** Default to none. Write a comment only when the *why* is non-obvious — for example, the `rotate_half` half-then-half convention (vs interleaved), the `token_dropout` 0.88 rescale, the Q-scale-before-RoPE ordering. These are silent-failure traps and warrant one-line `// NOTE:` markers with a pointer to the ESM-2 source.

**Build flags.** Release builds use `-O3 -march=x86-64-v3` baseline; AVX-512/VNNI paths compiled with `-mavx512f -mavx512bw -mavx512vnni` into separate translation units and selected at runtime. AMX path compiled with `-mamx-tile -mamx-int8 -mamx-bf16` and runtime-gated. `-fno-exceptions -fno-rtti` for kernel TUs.

## 6. Testing Strategy

Testing is the project's load-bearing wall — silent numerical bugs (RoPE convention, token_dropout, Q-scale order) are the #1 failure mode for ESM ports. The strategy is layered:

**Layer 1 — Kernel unit tests (GoogleTest, `tests/cpp/`).** Each SIMD kernel tested against its scalar reference for representative shapes. Tolerances: FP32 `rtol=1e-6 atol=1e-6`; INT8 `rtol=1e-3 atol=1`. Property-based shape sweeps via `rapidcheck`.

**Layer 2 — Layer-by-layer golden-tensor tests (pytest, `tests/python/test_against_hf.py`).** Capture hidden states from `EsmModel(output_hidden_states=True)` at every layer for 100 random UniRef50 sequences (length 50–300) at 8M and 35M scale. Stored as `.npz` in `tests/golden/`. esm.cpp's FP32 forward must match with `rtol=1e-3 atol=1e-3` at every layer. This is the test that catches RoPE bugs immediately.

**Layer 3 — Tokenizer conformance.** Round-trip 10K sequences (canonical + rare aa: B/U/Z/O/X/. ) through HF's `EsmTokenizer` and ours; require byte-exact ID sequences. Specifically exercises `prepend_bos=True`, `append_eos=True`, `<null_1>=31`, and the non-alphabetical UR50-frequency alphabet order.

**Layer 4 — Quality benchmarks (the Phase 2 quality gate).**
- **Pseudo-perplexity (PPPL).** Exact form (mask one residue at a time, sum NLLs, exponentiate the mean) on a fixed 1000-sequence UniRef50 holdout. Gate: `|PPPL_int8 − PPPL_fp32| < 0.1`.
- **ProteinGym v1.3.** OATML-Markslab/ProteinGym, 217 DMS substitution assays, zero-shot masked-marginal scoring. Primary metric: Spearman correlation. Gate: `|Spearman_int8 − Spearman_fp32| < 0.01` averaged across assays.
- Below 150M (8M, 35M): no INT8 path shipped; FP32/BF16 only. (Mirrors ESME's empirical finding.)

**Layer 5 — Performance benchmarks (the Phase 1/3 perf gate).**
- Microbenchmarks (Google Benchmark): SGEMM and INT8 GEMM on the four critical shapes — `[B·L, d, 3d]`, `[B·L, d, d]`, `[B·L, d, 4d]`, `[B·L, 4d, d]` for `d ∈ {320, 640, 1280, 2560}`. Target: ≥80% of MKL on each.
- End-to-end: `python -m esm_cpp.bench.compare` runs HF PyTorch eager FP32 and esm.cpp on identical inputs, reports throughput (seqs/sec) and p50/p99 latency.

Phase advance gates are defined in §2; this section defines *how* they are measured.

**CI.** GitHub Actions matrix: Ubuntu + macOS (build only on Mac; AVX-512 tests skipped), GCC 13 + Clang 17, Debug + Release. ASan/UBSan on Debug. Golden-tensor tests run on every PR; ProteinGym runs nightly (it is slow). No PR merges with failing kernel unit tests or HF parity tests.

## 7. Boundaries

**Always do**
- Verify FP32 bit-equivalence to HuggingFace `EsmModel` at every transformer layer before any optimization work. Layer-by-layer golden tensors are the single most important regression guard.
- Replicate ESM-2's three architectural quirks exactly: half-then-half RoPE (`rotate_half`), `token_dropout` 0.88 rescale at inference, query-scaled-before-RoPE.
- Compute INT8 quantization scales on real UniRef50 calibration data (256–1024 sequences, length 100–500). Never use random or synthetic calibration.
- Pin all third-party dependency versions (`libxsmm`, `pybind11`, HF transformers used for golden capture) in CMake/pyproject. Reproducibility is the deliverable.
- Lead public claims with AVX-512+VNNI numbers (widely reproducible). Report AMX as a bonus, gated behind runtime detect.
- Run quality gates (PPPL, ProteinGym) before merging any change to a kernel, quant recipe, or forward graph.

**Ask first**
- Before changing the quant recipe (SmoothQuant α, observer percentile, per-channel/per-tensor) once Phase 2 is passing — easy to silently regress quality.
- Before changing the public Python API surface (`esm_cpp` module) once Phase 3 ships.
- Before adding a second hardware target (ARM NEON/SVE2, Apple Accelerate as a perf backend rather than dev fallback). v1 is AVX-512 primary, AMX bonus.
- Before vendoring or pinning to a new third-party dependency.

**Never do**
- Implement training, backward passes, LoRA, or any gradient computation. ESME owns that niche.
- Hand-write GEMM from absolute scratch under deadline pressure for shapes where libxsmm or ruy are already optimal. Focus the project's value on schedulers, quantization recipe, and varlen attention — not on reinventing dgemm.
- Chase ESM-2-15B in v1. Bandwidth-bound on commodity CPUs; only viable with W4 weight-only quant. v2 roadmap with AWQ.
- Quantize models below 150M to INT8. ESME shows quality collapses; ship FP32/BF16 only at 8M and 35M.
- Skip the `token_dropout` 0.88 rescale or the half-then-half RoPE convention to "simplify" the forward pass. Both produce qualitatively correct but quantitatively wrong outputs and are silent.
- Accumulate softmax in FP16 inside attention. FP32 accumulator inside the FlashAttention inner block is mandatory, especially under INT8 K/V.
- Position esm.cpp publicly as "ESME for CPU." Position it as "CPU-first, ahead-of-time INT8, continuous batching for encoder-only PLMs." ESME is GPU + bitsandbytes weight-only; esm.cpp is CPU + W8A8 SmoothQuant + cu_seqlens scheduler. The distinction matters.
