# Phase 0 — Implementation Plan

**Scope.** FP32 reference forward for ESM-2-8M and 35M, bit-equivalent to HuggingFace `EsmModel`. Foundation for everything that follows. No SIMD, no quantization, no scheduler, no GGUF — those are later phases. See [SPEC.md §2](../SPEC.md) for the build list, gate, and retro requirement.

**Gate (must hold to declare Phase 0 done).**
- FP32 max-abs-diff < 1e-4 on final logits vs HF `EsmModel(output_hidden_states=True)` on 100 random sequences for both ESM-2-8M and ESM-2-35M.
- Layer-by-layer hidden states `allclose(rtol=1e-3, atol=1e-3)` at every hidden layer.
- Python e2e test (`pytest tests/python`) green; CI matrix green.

**Time budget.** 2 weeks of focused work. ~60% of that is Slice 4 (the forward graph); the rest is scaffolding, tokenizer, golden capture, bindings, and retro.

---

## Why slice this way

The risk profile is asymmetric. The build system, tokenizer, golden-capture script, and Python bindings are well-understood work — low surprise. The FP32 forward graph is a bug-trap field: half-then-half RoPE vs interleaved, `token_dropout` 0.88, Q-scale-before-RoPE, the tokenizer's non-alphabetical alphabet. A single wrong sign in `rotate_half` looks "qualitatively correct" and matches at position 0 but diverges everywhere else.

So: build infrastructure that surfaces these bugs *before* writing the forward graph, then walk the forward graph component-by-component with goldens captured at every intermediate tensor. The goldens are not a polish step at the end — they are the test harness.

---

## Dependency graph

```
S1 Scaffolding ──────────────────────────────────────────┐
                                                         │
        ┌────────────────────┬───────────────────┐       │
        ▼                    ▼                   ▼       │
   S2 Tokenizer       S3 Golden Capture    S4.1 Loader   │
        │                    │                   │       │
        └───────┬────────────┘                   │       │
                ▼                                ▼       │
                              S4 FP32 Forward (8M)       │
                              (sub-tasks 4.2 → 4.10)     │
                                       │                 │
                                       ▼                 │
                              S5 35M passes the gate     │
                                       │                 │
                                       ▼                 │
                              S6 Python bindings + e2e   │
                                       │                 │
                                       ▼                 │
                              S7 notes/phase0.md retro   │
                                       │                 │
                                       └─────► Phase 1 ──┘
```

Parallelizable: S2 and S3 are independent after S1. Inside S4, the loader (4.1) can begin in parallel with the capture tool (S3).

---

## Vertical slices

### Slice 1 — Scaffolding

Stand up the build system, directory layout from [SPEC §4](../SPEC.md), and CI. No real functionality; the deliverable is "everything compiles and a hello-world test runs."

**Tasks**
- **1.1** Create directory tree per SPEC §4 (`include/esm_cpp/`, `src/`, `src/kernels/`, `src/quant/`, `src/io/`, `src/sched/`, `python/esm_cpp/`, `tests/cpp/`, `tests/python/`, `tests/golden/`, `bench/`, `tools/`, `notes/`, `docs/`, `third_party/`).
- **1.2** Top-level `CMakeLists.txt`: C++20, `-O3 -march=x86-64-v3` Release, `-O0 -g -fsanitize=address,undefined` Debug, FetchContent for `pybind11` and `GoogleTest`. `clang-format` and `clang-tidy` targets.
- **1.3** `pyproject.toml` with `scikit-build-core` backend; declares `esm_cpp` package, dev extras (`pytest`, `numpy`, `torch`, `transformers`, `ruff`, `mypy`).
- **1.4** Skeleton public headers: `include/esm_cpp/version.h`, `include/esm_cpp/status.h` (with `enum class StatusCode` and `esm::Status` value type — no exceptions in hot paths per CLAUDE.md).
- **1.5** Placeholder GoogleTest in `tests/cpp/test_smoke.cpp` that links the lib and passes; placeholder `tests/python/test_smoke.py` that imports the (empty) pybind11 module.
- **1.6** `.github/workflows/ci.yml`: matrix (Ubuntu 22.04 + macOS-14) × (GCC 13 + Clang 17) × (Debug + Release). Run `ctest` and `pytest`. ASan only on Debug. No nightly job yet.
- **1.7** Pin the HF `transformers` version used for golden capture in `pyproject.toml` dev extras and in a top-level comment in `tools/capture_golden.py`.

**Acceptance criteria**
- `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure` succeeds on Ubuntu and macOS.
- `pip install -e ".[dev]" && pytest tests/python -v` succeeds.
- `clang-tidy` runs (may produce warnings on empty code; must not error).
- CI matrix is green on a placeholder PR.

**Verification**
Push a no-op PR; observe matrix green. Run both build modes locally.

**Out of slice**
No model code, no kernels, no real Python module body — just the scaffolding.

---

### Slice 2 — Tokenizer

Implement the 33-token alphabet with the UR50 frequency order, special-token handling, and BOS/EOS injection. Tokenizer has no dependencies on the forward graph and can be verified independently. This is the cheapest place to catch the "alphabetical-order assumption" bug.

**Tasks**
- **2.1** Implement `esm::Tokenizer` in `include/esm_cpp/tokenizer.h` + `src/tokenizer.cpp`. Hardcoded vocabulary in UR50 frequency order exactly as listed in [research-report.md §1](../research-report.md): `<cls>=0, <pad>=1, <eos>=2, <unk>=3, L=4, A=5, G=6, V=7, S=8, E=9, R=10, T=11, I=12, D=13, P=14, K=15, Q=16, N=17, F=18, Y=19, M=20, H=21, W=22, C=23, X=24, B=25, U=26, Z=27, O=28, .=29, -=30, <null_1>=31, <mask>=32`. Methods: `encode(std::string_view, bool add_special=true) -> std::vector<int32_t>`, `decode(std::span<const int32_t>) -> std::string`.
- **2.2** Enforce `prepend_bos=True`, `append_eos=True` when `add_special=true`. Enforce max length 1024 (truncate to 1022 residues + bos + eos).
- **2.3** GoogleTest unit tests covering: canonical 20 aa, rare aa (B/U/Z/O/.), unknown chars → `<unk>=3`, `<null_1>=31` round-trip, max-length truncation.
- **2.4** Python test `tests/python/test_tokenizer.py`: tokenize 10,000 random sequences (mix of canonical-only, rare-aa, with special chars) and compare byte-exact against `EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")`.

**Acceptance criteria**
- 10,000 random sequences tokenize byte-exact to HF (`assert our_ids == hf_ids` for all).
- Round-trip decode also exact for sequences without `<unk>`.
- Unit tests cover the alphabet edge cases enumerated above.

**Verification**
`pytest tests/python/test_tokenizer.py -v` green; `ctest -R tokenizer` green.

**Out of slice**
Subword tokenization (not used by ESM-2). Streaming/incremental tokenization.

---

### Slice 3 — Golden tensor capture

Build the Python tool that produces ground truth tensors from HF `EsmModel`. This must exist before Slice 4 starts. Capture hidden states at *every* layer plus final logits — these are the layer-by-layer assertions Slice 4 will run against.

**Tasks**
- **3.1** `tools/capture_golden.py` — args: `--model {esm2_t6_8M, esm2_t12_35M}`, `--num-seqs 100`, `--seed 0`, `--out tests/golden/<model_short>/`. Generates 100 random protein sequences with realistic length distribution (uniform in [50, 300] aa for Phase 0; longer sequences in Phase 1+).
- **3.2** For each sequence, capture: `input_ids` (1-D int32), `attention_mask` (1-D int32), `hidden_states[0..N]` (post-embed, post-each-layer; 2-D float32 `[seq_len, d]`), and `logits` (2-D float32 `[seq_len, 33]`). Also capture intermediate tensors needed by Slice 4 sub-tasks — see 3.3.
- **3.3** Add hooks to capture per-layer intermediates at layer 0 (the hardest one to debug): `pre_attn_ln_input`, `pre_attn_ln_output`, `qkv_raw` (before RoPE), `q_after_rope`, `k_after_rope`, `attn_output` (post-softmax × V, before out_proj), `post_attn_residual`, `pre_ffn_ln_output`, `ffn_output`, `post_ffn_residual`. These are the layer-0-only "debug goldens" — for layers 1..N-1 we only capture the post-layer hidden state.
- **3.4** Write all tensors to `tests/golden/<model_short>/seq_<i>.npz` via `numpy.savez_compressed`. Write `tests/golden/<model_short>/manifest.json` with HF transformers version, model HF ID, commit hash, capture date, and seed.
- **3.5** Capture for both `esm2_t6_8M` and `esm2_t12_35M`.

**Acceptance criteria**
- Tool produces 100 npz files per model in `tests/golden/{8m,35m}/`.
- Each npz contains all keys listed in 3.2 + the layer-0 debug keys in 3.3.
- Total size per model is reasonable for git (use `.gitattributes` + Git LFS, or commit a small representative subset — e.g., 5 seqs — and have CI run the full capture; decide during 3.1).
- Manifest.json records HF version pin matching `pyproject.toml`.

**Verification**
`python tools/capture_golden.py --model esm2_t6_8M --num-seqs 100 --out tests/golden/8m/` runs end-to-end. Spot-check one npz: shapes match expected `[seq_len, 320]` for 8M hidden states.

**Out of slice**
Capturing 150M/650M/3B goldens (Phase 0 only needs 8M and 35M; bigger models come in Phase 1 perf work).

---

### Slice 4 — FP32 forward graph (ESM-2-8M passes the gate)

The heart of Phase 0. Component-by-component build, each sub-task verified against the goldens captured in Slice 3. The slice itself isn't "done" until the gate criterion (`max_abs_diff < 1e-4` on final logits for 8M) holds across all 100 sequences.

Sub-tasks are sequenced in dependency order. Each has a layer-0 golden check as its acceptance criterion; the full N-layer check comes after 4.8.

**Tasks**

- **4.1 Safetensors weight loader.** `src/io/safetensors.cpp` reads HF ESM-2 safetensors. Returns a `WeightMap` keyed by HF parameter name (e.g., `esm.encoder.layer.0.attention.self.query.weight`). Validates expected shapes against the loaded `EsmConfig`. *Verification:* unit test loads `facebook/esm2_t6_8M_UR50D`, asserts presence of all expected keys and correct shapes.

- **4.2 Scalar reference matmul.** `src/kernels/gemm_fp32.cpp` — `matmul_ref(A[M,K], B[K,N], C[M,N], bias[N] or null)`. Naive 3-loop, FP32 in, FP32 out. *Verification:* compare against `numpy.matmul` on 100 random shape combinations; `allclose(atol=1e-5)`.

- **4.3 LayerNorm.** `src/kernels/layernorm.cpp` — `layernorm_ref(x[L,d], gamma[d], beta[d], eps=1e-5, out[L,d])`. *Verification:* against the `pre_attn_ln_output` golden at layer 0, seq 0. `max_abs_diff < 1e-5`.

- **4.4 Embed + token_dropout.** `src/model.cpp` — `embed_lookup(token_ids[L], embed_table[33,d], out[L,d])` followed by the **0.88 rescale** for inference (with zero observed masks). *Verification:* against the `hidden_states[0]` golden (post-embed) at seq 0. `max_abs_diff < 1e-5`. **Bug trap:** forgetting the 0.88 multiplier produces ~12% wrong embeddings.

- **4.5 RoPE half-then-half + Q-scale-before-RoPE.** `src/kernels/rope.cpp` — precompute `inv_freq[i] = 1/10000^(2i/head_dim)` for `i ∈ [0, head_dim/2)`, build per-seqlen cos/sin tables. `rotate_half(x[..., d])`: `x1, x2 = chunk(x, 2, -1); return cat(-x2, x1, -1)`. **Apply Q-scale (`1/√head_dim`) BEFORE rope**, then rotate Q and K independently. *Verification:* against `q_after_rope` and `k_after_rope` goldens at layer 0, seq 0. `max_abs_diff < 1e-5`. **Bug trap:** interleaved-pairs RoPE (`x[..., 0::2], x[..., 1::2]`) is equivalent only at position 0 and silently diverges everywhere else.

- **4.6 Scaled-dot attention.** `src/kernels/attention.cpp` — scalar reference (no FlashAttention yet — that's Phase 1). Compute `scores = Q @ K^T` (Q already scaled in 4.5), apply attention mask (`-inf` on pad positions), `softmax(dim=-1)` with FP32 accumulator, `out = attn @ V`. *Verification:* against `attn_output` golden at layer 0, seq 0. `max_abs_diff < 1e-4`.

- **4.7 Out-projection + residual.** Apply `out_proj` weights, add to pre-layernorm input. *Verification:* against `post_attn_residual` golden at layer 0.

- **4.8 FFN with GELU.** `src/kernels/gelu.cpp` — use the **tanh approximation** to match PyTorch's default: `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`. FFN: `fc1(d → 4d)`, GELU, `fc2(4d → d)`, add residual. *Verification:* against `post_ffn_residual` golden at layer 0. **Bug trap:** `torch.nn.functional.gelu` defaults to the exact erf form, but `EsmModel` uses tanh approximation — confirm against the actual golden.

- **4.9 Transformer block + stack.** Compose 4.3–4.8 into one `transformer_block` function, then loop N times. *Verification:* against `hidden_states[1..N]` goldens for seq 0. `max_abs_diff < 5e-4` per layer (small float drift accumulates).

- **4.10 Final LN + tied lm_head.** Final `layernorm` (after last block), then `lm_head`: `dense (d → d)` → `gelu` → `layernorm` → `decoder (d → 33)` using **tied embed weights** (`weight = embed_table.T`) plus a separate bias. *Verification:* against `logits` golden, all 100 sequences. **This is the Phase 0 gate criterion (8M leg).** `max_abs_diff < 1e-4`.

**Acceptance criteria (slice-level)**
- All 4.1–4.10 sub-task golden checks pass.
- The 100-sequence final-logits gate (`max_abs_diff < 1e-4` on ESM-2-8M) passes.
- Layer-by-layer parity (`rtol=1e-3, atol=1e-3`) holds at every hidden layer, all 100 sequences, ESM-2-8M.

**Verification**
`pytest tests/python/test_against_hf.py::test_8m_parity -v` green. CI runs this on every PR.

**Out of slice**
- Any SIMD or thread parallelism (Phase 1).
- INT8/quantization (Phase 2).
- The `cu_seqlens`/varlen attention API (Phase 1 introduces it; Phase 0 attention is plain `[B, H, L, L]`).
- Arena allocator / zero-allocation hot path (Phase 1).

---

### Slice 5 — ESM-2-35M passes the gate

Same forward, larger model. If Slice 4 was implemented genuinely generically (no hardcoded `d=320`, no hardcoded layer count), this should be near-free. The slice exists as an explicit gate to catch dimension assumptions.

**Tasks**
- **5.1** Run the existing `test_against_hf` suite with `model=esm2_t12_35M`. Capture any failures.
- **5.2** Fix any non-generic code paths exposed (likely candidates: hardcoded `head_dim`, hardcoded layer count in tests, off-by-one in mask broadcasting). No new files expected.
- **5.3** Add `pytest tests/python/test_against_hf.py::test_35m_parity` to CI.

**Acceptance criteria**
- 100-sequence final-logits gate passes on ESM-2-35M (`max_abs_diff < 1e-4`).
- Layer-by-layer parity holds across all 12 layers.

**Verification**
`pytest tests/python/test_against_hf.py -v` (both 8m and 35m) green in CI.

**Out of slice**
150M, 650M, 3B parity. Those run in Phase 1's perf testing once SIMD is in.

---

### Slice 6 — Python bindings + end-to-end test

Expose `esm::Model` to Python via pybind11. Minimal surface: load, forward, tokenize. Enough to demonstrate the library works from Python and for downstream test code to consume it.

**Tasks**
- **6.1** `python/esm_cpp/_core.cpp` — pybind11 module. Bind `esm::Tokenizer` (`encode`, `decode`) and `esm::Model` (`load(path) -> Model`, `forward(input_ids: numpy.ndarray[int32]) -> numpy.ndarray[float32]`).
- **6.2** `py::call_guard<py::gil_scoped_release>()` on `forward`. Even though Phase 0 is single-threaded, the binding contract is established here.
- **6.3** `python/esm_cpp/__init__.py` — re-exports `Tokenizer`, `Model`. Type stubs `python/esm_cpp/_core.pyi`.
- **6.4** `tests/python/test_e2e.py` — load HF safetensors via our loader, tokenize a known sequence, run `Model.forward`, compare against HF `EsmForMaskedLM` logits. `max_abs_diff < 1e-4`.

**Acceptance criteria**
- `import esm_cpp` works after `pip install -e .`.
- `m = esm_cpp.Model.load("weights/esm2_8m"); logits = m.forward(token_ids)` produces logits matching HF within gate tolerance.
- `mypy --strict python/esm_cpp/` clean.

**Verification**
`pytest tests/python/test_e2e.py -v` green. CI green.

**Out of slice**
Batched inference (`forward(input_ids[B, L])` works because of broadcasting, but no scheduler yet). Streaming, async, GIL-free calling — Phase 1+.

---

### Slice 7 — Phase 0 retrospective

Write `notes/phase0.md` per [SPEC §2](../SPEC.md) guidance.

**Tasks**
- **7.1** Write `notes/phase0.md`. One screen of prose. Cover: what shipped, what didn't and why, measured numbers (e.g., FP32 max-abs-diff and runtime on 8M/35M on dev machine), surprises (the bug-traps that bit us — likely RoPE convention, GELU variant, or token_dropout), deviations from SPEC, what to carry into Phase 1.
- **7.2** Update top-level README with "Phase 0 complete" note and a one-liner reproduction command.

**Acceptance criteria**
- `notes/phase0.md` exists, ≤ ~80 lines of prose.
- README reflects current state.

---

## Checkpoints

- **Checkpoint A (after Slice 1).** Build, CI, and Python install are working. Stop here, push to a branch, confirm matrix green before continuing. Catching CI issues now is cheap; catching them after Slice 4 is expensive.
- **Checkpoint B (after Slices 2 + 3).** Tokenizer is byte-exact to HF and goldens are captured. The test harness exists. Pause and review the captured tensor file sizes — decide on Git LFS vs. CI-time regeneration if not already.
- **Checkpoint C (after Slice 4, BEFORE Slice 5).** This is the de-risking moment. ESM-2-8M passes the gate. If max-abs-diff is between 1e-4 and 1e-3, do NOT widen tolerances — diagnose. Likely diagnoses: RoPE convention, GELU variant, token_dropout, Q-scale order, off-by-one mask. Use the layer-0 debug goldens captured in 3.3 to bisect.
- **Checkpoint D (after Slice 6).** Phase 0 gate met end-to-end from Python. Schedule retro writing and Phase 1 planning kickoff.

---

## Risks specific to Phase 0

| Risk | Mitigation |
|---|---|
| RoPE convention bug (half-then-half vs interleaved) silently produces "qualitatively correct" output. | Layer-0 `q_after_rope` golden check in 4.5. Test with `seq_len ≥ 5` so non-zero positions are exercised. |
| `token_dropout` 0.88 rescale forgotten. | Layer-0 post-embed golden in 4.4. Will fail by ~12% — unmissable. |
| GELU variant mismatch (exact erf vs tanh approximation). | Use the tanh approximation explicitly. Verify against `post_ffn_residual` golden in 4.8. |
| Q-scale applied AFTER RoPE instead of before. | Equivalent in exact arithmetic, numerically different. Layer-0 attention-output golden in 4.6 will catch any deviation > 1e-5. |
| Tokenizer alphabet ordering (UR50 frequency, not alphabetical). | 10K-sequence byte-exact check vs HF in 2.4. |
| Golden tensors balloon the git repo. | Decided in Slice 3: Git LFS or commit a 5-seq subset and regenerate at CI time. |
| Float drift accumulates across 33 layers (650M) and breaks the 1e-4 gate. | Phase 0 only targets 8M (6 layers) and 35M (12 layers); deeper drift is a Phase 1 concern. If 35M shows > 5e-4 per-layer drift, investigate before assuming Phase 0 is done. |

---

## Out of scope for Phase 0 (explicit reminders)

- No SIMD intrinsics anywhere. Scalar reference only.
- No INT8, no quantization, no calibration.
- No `cu_seqlens` scheduler, no length bucketing, no batched scoring service.
- No GGUF reader/writer.
- No 150M, 650M, 3B, or 15B work.
- No arena allocator; `std::vector` scratch buffers are fine.
- No thread pool, no parallel anything. Single-threaded reference.
- No public Python API beyond `Tokenizer` and `Model.{load, forward}`.

These come back in Phases 1–3.
