# Phase 0 — Task List

Companion to [plan.md](plan.md). Tick tasks as they land. Acceptance criteria and verification steps live in plan.md; this file is the running ledger.

## Slice 1 — Scaffolding

- [ ] **1.1** Create directory tree per [SPEC §4](../SPEC.md)
- [ ] **1.2** Top-level `CMakeLists.txt` (C++20, Release/Debug flags, FetchContent for pybind11 + GoogleTest, clang-format/clang-tidy targets)
- [ ] **1.3** `pyproject.toml` with scikit-build-core, dev extras (`pytest`, `numpy`, `torch`, `transformers`, `ruff`, `mypy`)
- [ ] **1.4** `include/esm_cpp/version.h` and `include/esm_cpp/status.h`
- [ ] **1.5** Placeholder smoke tests (`tests/cpp/test_smoke.cpp`, `tests/python/test_smoke.py`)
- [ ] **1.6** `.github/workflows/ci.yml` (Ubuntu + macOS × GCC + Clang × Debug + Release)
- [ ] **1.7** Pin HF `transformers` version in `pyproject.toml` and `tools/capture_golden.py`
- [ ] **Checkpoint A** — push branch, confirm matrix green

## Slice 2 — Tokenizer

- [ ] **2.1** `esm::Tokenizer` in `include/esm_cpp/tokenizer.h` + `src/tokenizer.cpp` with hardcoded 33-token vocab in UR50 frequency order
- [ ] **2.2** `prepend_bos=True`, `append_eos=True`, max length 1024
- [ ] **2.3** GoogleTest unit tests (canonical aa, rare aa, unknowns, `<null_1>`, truncation)
- [ ] **2.4** `tests/python/test_tokenizer.py` — 10K random sequences byte-exact vs HF `EsmTokenizer`

## Slice 3 — Golden tensor capture

- [ ] **3.1** `tools/capture_golden.py` with `--model`, `--num-seqs`, `--seed`, `--out` args; uniform length distribution [50, 300]
- [ ] **3.2** Capture per sequence: `input_ids`, `attention_mask`, `hidden_states[0..N]`, `logits`
- [ ] **3.3** Capture layer-0 debug intermediates: `pre_attn_ln_input/output`, `qkv_raw`, `q_after_rope`, `k_after_rope`, `attn_output`, `post_attn_residual`, `pre_ffn_ln_output`, `ffn_output`, `post_ffn_residual`
- [ ] **3.4** Write npz files + `manifest.json` with HF version + commit + date + seed
- [ ] **3.5** Capture for both `esm2_t6_8M` and `esm2_t12_35M`
- [ ] **Checkpoint B** — review file sizes, decide Git LFS vs CI-time regeneration

## Slice 4 — FP32 forward graph (ESM-2-8M passes the gate)

- [ ] **4.1** Safetensors weight loader (`src/io/safetensors.cpp`); validates shapes against `EsmConfig`
- [ ] **4.2** Scalar reference `matmul_ref` (`src/kernels/gemm_fp32.cpp`)
- [ ] **4.3** Scalar reference `layernorm_ref` (`src/kernels/layernorm.cpp`); verify against `pre_attn_ln_output` layer-0 golden
- [ ] **4.4** Embed lookup + `token_dropout` 0.88 rescale; verify against `hidden_states[0]` golden — **bug trap: 0.88 multiplier**
- [ ] **4.5** RoPE half-then-half (`rotate_half`) + Q-scale-BEFORE-RoPE; verify against `q_after_rope` and `k_after_rope` goldens — **bug trap: half-then-half vs interleaved**
- [ ] **4.6** Scaled-dot attention (FP32 softmax accumulator, scalar reference); verify against `attn_output` golden
- [ ] **4.7** Out-projection + residual; verify against `post_attn_residual` golden
- [ ] **4.8** FFN with GELU **tanh approximation**; verify against `post_ffn_residual` golden — **bug trap: GELU variant**
- [ ] **4.9** Transformer block + N-layer stack; verify against `hidden_states[1..N]` for seq 0
- [ ] **4.10** Final LN + tied `lm_head` (dense → gelu → layernorm → tied-decoder + bias); verify against `logits` golden across all 100 sequences
- [ ] **Slice gate** — `pytest tests/python/test_against_hf.py::test_8m_parity` green, `max_abs_diff < 1e-4` across 100 sequences
- [ ] **Checkpoint C** — Phase 0 de-risking complete; do NOT widen tolerances if you see drift, diagnose

## Slice 5 — ESM-2-35M

- [ ] **5.1** Run `test_against_hf` with `model=esm2_t12_35M`; capture failures
- [ ] **5.2** Fix any non-generic code paths (hardcoded dims, layer counts, mask broadcasting)
- [ ] **5.3** Add `test_35m_parity` to CI
- [ ] **Slice gate** — 100-sequence final-logits gate passes for 35M; layer-by-layer parity holds across 12 layers

## Slice 6 — Python bindings + e2e

- [ ] **6.1** `python/esm_cpp/_core.cpp` (pybind11): bind `Tokenizer` and `Model.{load, forward}`
- [ ] **6.2** `py::call_guard<py::gil_scoped_release>()` on `forward`
- [ ] **6.3** `python/esm_cpp/__init__.py` + `_core.pyi` type stubs
- [ ] **6.4** `tests/python/test_e2e.py` — load via our loader, forward, compare against HF `EsmForMaskedLM`
- [ ] **Slice gate** — `import esm_cpp` works; e2e test green; `mypy --strict` clean
- [ ] **Checkpoint D** — Phase 0 gate met end-to-end from Python

## Slice 7 — Retrospective

- [ ] **7.1** `notes/phase0.md` (one screen of prose covering what shipped, surprises, deviations, carry-forward)
- [ ] **7.2** Update README with Phase 0 status + reproduction command

## Phase 0 done

- [ ] All slice gates green
- [ ] All 4 checkpoints (A, B, C, D) cleared
- [ ] `notes/phase0.md` committed
- [ ] CI matrix green on `main`
- [ ] Ready to start Phase 1 planning
