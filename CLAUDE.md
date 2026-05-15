# CLAUDE.md — esm.cpp engineering practices

This is a CPU-first C++ inference engine for ESM-2 protein language models with Python bindings. See [SPEC.md](SPEC.md) for objective, structure, and stage gates. This file is the working contract for *how* code gets written.

## Orientation

- Public C++ API lives in `include/esm_cpp/` under namespace `esm::`. Implementation lives in `src/`. Python bindings (pybind11) live in `python/esm_cpp/` and may only depend on public headers.
- Kernels in `src/kernels/` are the hot path. Everything else is the cold path (model load, tokenizer, scheduler setup, format conversion).
- Numerical correctness is the load-bearing wall. Three quirks of ESM-2 silently break ports if missed: half-then-half RoPE, `token_dropout` 0.88 rescale at inference, and Q-scale-before-RoPE. Treat changes near these with paranoia.

## Language and toolchain

- **C++20.** Use `<concepts>`, `std::span`, designated initializers, `if constexpr`. Avoid `<ranges>` views in hot paths (compile-time and codegen costs).
- **No exceptions in hot paths** (anything in `src/kernels/`). Kernels return `esm::Status` error codes. The model-level API may throw at load time only.
- **`-fno-exceptions -fno-rtti`** on kernel translation units. Public API TUs keep exceptions.
- Compile baseline: `-O3 -march=x86-64-v3`. AVX-512/VNNI and AMX paths are separate TUs with their own flags, selected at runtime via `cpu_features.cpp`. Never sprinkle `__attribute__((target("avx512f")))` across general code — it makes the dispatch surface impossible to reason about.

## Memory and ownership

- **RAII everywhere.** No raw `new`/`delete`. No `malloc`/`free` outside the arena allocator.
- **Allocate once at load.** All scratch buffers used by the forward pass are owned by the `Model` instance and reused. The inner forward loop must allocate zero bytes.
- **Weights are mmap'd from GGUF**, zero-copy for FP32, block-decoded on the fly for INT8. Do not copy weight tensors into RAM at load.
- `std::unique_ptr` for owning, raw pointers or references for non-owning. `std::shared_ptr` only when ownership is genuinely shared — almost never in this codebase. If you find yourself reaching for `shared_ptr`, the ownership graph is probably wrong.
- Pass large objects by `const&` or `std::span<const T>`. Return by value and trust RVO. Never return raw pointers to internal storage from a public API.

## Naming and style

- `snake_case` for functions and variables, `PascalCase` for types, `SCREAMING_SNAKE` for compile-time constants. Matches ggml/llama.cpp — contributors crossing over should feel at home.
- `clang-format` with Google style, column limit 100. Enforced in CI.
- `clang-tidy` checks: `bugprone-*`, `cert-*`, `performance-*`, `readability-identifier-naming`, `modernize-*` (excluding `modernize-use-trailing-return-type`). Warnings are errors in CI.
- One public type per header. Header guards via `#pragma once`.
- Never `using namespace std;` anywhere. Never `using namespace ...` in a header, ever.
- Forward-declare in headers when possible. Heavy includes (`<vector>`, `<unordered_map>`, intrinsics headers) belong in `.cpp` files.

## Don't reinvent

- **GEMM:** link `libxsmm` for small-shape kernels and our own packed-AVX512/AMX paths for the four critical shapes (`[B·L, d, 3d]`, `[B·L, d, d]`, `[B·L, d, 4d]`, `[B·L, 4d, d]`). Do not write a general SGEMM from scratch.
- **STL algorithms:** use `std::sort`, `std::transform`, `std::accumulate`, etc. Don't roll loops where a one-liner exists.
- **Containers:** `std::vector` is the default. `std::array` when size is compile-time. `absl::flat_hash_map` if hashing shows up on a profile; until then plain `std::unordered_map` is fine. Avoid `std::list`, `std::deque`, `std::map` outside specific reasons.

## Kernel discipline

Every SIMD kernel has two implementations in the same `.cpp`:

1. A scalar reference behind `#ifdef ESM_KERNEL_REFERENCE`. Trivially correct. Used by unit tests as the ground truth.
2. One or more vectorized implementations dispatched by `cpu_features.cpp`.

Both paths are exercised by the same `GoogleTest` cases with `allclose` tolerances (FP32: `rtol=1e-6 atol=1e-6`; INT8: `rtol=1e-3 atol=1`). If they disagree, the vectorized path is wrong — never the test.

**FP32 accumulator inside FlashAttention's inner block is mandatory**, especially under INT8 K/V. FP16 softmax accumulation silently loses precision on long sequences. Don't "save a register" by downcasting the accumulator.

## Threading

- One thread pool per process, initialized at `Model::load`. Configurable via env var `ESM_NUM_THREADS`, defaults to physical core count.
- Parallelize across the batch dimension (`cu_seqlens` packed batch) and across the FFN `4d` dimension. Do not parallelize per-token in the attention kernel — overhead dominates.
- No raw `std::thread` in the forward path. No `std::async`. No locks in the inner loop.

## Error handling

- **Kernels:** return `esm::Status`. Callers check. No exceptions cross a kernel boundary.
- **Model load and quantize:** may throw `esm::LoadError`, `esm::QuantError`. These are user-facing and report which file/tensor failed.
- **Python bindings:** translate C++ exceptions to Python exceptions via pybind11's default mechanism. Don't catch in pybind11 wrappers — let them propagate.

## Comments — be deliberate, not generous

**Default: no comments.** Well-named functions and variables narrate the code. The reader can see *what* the code does; they came here to understand *why* it does it that way, and only when that's non-obvious.

A comment is justified when **a reasonable C++ engineer reading the code would arrive at a wrong conclusion or wonder "is this a bug?"** without it. Otherwise it is noise.

**Examples where a comment IS warranted in this codebase** (one short line, prefixed `// NOTE:` for invariants or `// WHY:` for rationale):

```cpp
// NOTE: half-then-half (Llama/GPT-NeoX), NOT interleaved (RoFormer/GPT-J).
// See esm/rotary_embedding.py in facebookresearch/esm.
void rotate_half(...);

// WHY: ESM-2 rescales all embeddings by 0.88 at inference even when no
// tokens are masked. Skipping this gives qualitatively correct but
// quantitatively wrong logits. See esm/model/esm2.py.
x *= 0.88f;

// NOTE: scale Q before RoPE, not after. RoPE is norm-preserving only in
// exact arithmetic; order matters for bit-equivalence to the checkpoint.
q *= inv_sqrt_head_dim;
apply_rope(q, ...);

// WHY: SmoothQuant alpha=0.5 is the OPT/BLOOM default. Sweep on PPPL before
// shipping a different value — earlier layers have stronger activation outliers.
constexpr float kSmoothQuantAlpha = 0.5f;
```

**Examples that are NOT justified:**

```cpp
// BAD: restates the code
i++;  // increment i

// BAD: narrates the obvious flow
// First, load the weights from disk
load_weights(...);

// BAD: section banner
// =====================
//   ATTENTION KERNEL
// =====================

// BAD: PR/task reference (rots; belongs in the commit message)
// Added for the antibody benchmark in PR #42
void score_batch(...);

// BAD: author/date (use git blame)
// Ayan, 2026-05-15: refactored this

// BAD: a comment that just rephrases the function name
// Computes the layer norm
void layer_norm(...);

// BAD: TODO without an owner, condition, or date
// TODO: optimize this later
```

**Rules of thumb**

- If removing the comment would not confuse a future reader, delete it.
- Prefer to fix the name. "`x *= scale_for_mask_dropout_inference;`" is better than `// rescale for token dropout`.
- Comments are code — they go stale and they get reviewed. A wrong comment is worse than no comment.
- One line is almost always enough. If you need a paragraph, write a doc in `docs/` and link to it from a one-liner.
- For non-obvious external references (e.g., a specific line in the ESM source, a specific paper section), include the link in the comment so the next reader can verify.
- Doc comments (`///` or `/** */`) on public headers only, and only when the function signature alone doesn't tell the whole story — e.g., units, ownership semantics, threading guarantees.

## Performance discipline

- **Measure before optimizing.** Google Benchmark for microkernels, `perf stat` for cycles/IPC/cache misses, `llvm-mca` for static throughput on microkernel asm. Don't guess.
- Don't sacrifice readability for unmeasured perf gains. A 1% speedup on a non-hot path is not worth a confused reader.
- Branch hints (`[[likely]]`, `[[unlikely]]`) only when a profile shows they matter and the prediction is reliably skewed. Sprinkled blindly they often regress.
- Prefetching: same rule. Show me the cache-miss profile first.
- If you find yourself writing the same micro-optimization in two places, extract a helper. Don't copy-paste SIMD.

## Testing discipline

- **Never relax a test tolerance to make it pass.** If FP32 parity vs HF starts failing, the forward graph is wrong — fix it, don't widen the bound.
- Golden tensors in `tests/golden/` are authoritative. If they need to be regenerated (e.g., HF transformers version bump), regenerate them in a dedicated PR with a clear commit message and the new HF version pinned.
- Every kernel gets a scalar-reference cross-check before it gets a perf benchmark.
- ProteinGym and PPPL gates are slow — run them nightly in CI, not per-PR. But a PR that touches the quant recipe must trigger them manually before merge.

## Python binding hygiene

- Bindings expose only the public C++ API. Never bind a `src/` internal type.
- Use pybind11's `py::array_t<float>` for tensors; the C++ side validates shape and contiguity at the boundary, then drops to raw pointers internally.
- Release the GIL (`py::call_guard<py::gil_scoped_release>`) on any C++ call that runs the forward pass or a kernel. The whole point of a C++ engine is that Python concurrency works.
- Type stubs in `python/esm_cpp/_core.pyi`. `mypy --strict` on `python/esm_cpp/`.

## What to refuse

- A request to "just add a backward pass / training / LoRA." Out of scope. ESME owns that.
- A request to remove a `token_dropout`, RoPE-half, or Q-scale-order check to "simplify" the forward. These are not safe to remove.
- A request to widen a numerical tolerance because tests are failing. Find the bug.
- A request to chase 15B in v1. Bandwidth-bound; needs W4 quant; v2 work.
- A request to add ARM/Apple Silicon as a co-equal target before v1 ships on x86. Accelerate is a dev fallback only.

## When in doubt

Read the relevant section of `research-report.md` and `SPEC.md`. If they don't answer the question, ask before guessing. Silent assumptions are how ESM ports go wrong.
