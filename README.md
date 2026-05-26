# esm.cpp

A CPU-first C++ inference engine for ESM-2 protein language models, with
Python bindings. Production throughput on commodity hardware, no GPU required.

## Why

ESM-2 is the backbone of modern protein ML, but the reference stack is
PyTorch + CUDA. That's the wrong shape for several real workloads:

- **Deep mutational scanning** (10⁴–10⁷ variants per protein) — embarrassingly
  parallel scoring across millions of mutants. CPU throughput matters more
  than per-call latency.
- **Antibody developability screening** (10⁵–10⁶ candidates) — rank by PLM
  likelihood for aggregation, solubility, immunogenicity prefiltering.
  Nightly batch jobs on lab CPUs.
- **Embedding extraction at corpus scale** — per-residue or per-sequence
  vectors for downstream classifiers, alignment, or retrieval.
- **On-prem / regulated environments** — clinical, compliance-restricted,
  or air-gapped pipelines that cannot reach a cloud GPU.

The defensible niche is the intersection no existing project occupies:
**production-grade CPU inference + ahead-of-time W8A8 quantization +
variable-length packed-batch scheduling** for encoder-only PLMs. esm.cpp
targets it. v0.2 ships ESM-2 at 8M, 35M, 150M, 650M, and 3B, with W8A8
INT8 (SmoothQuant) for 150M and above and FP32 for the smaller two.

## Performance

ESM-2-650M, esm.cpp vs HuggingFace eager FP32, p50 over 5 runs:

| Host | Varlen 256-seq (OAS) | Uniform 8 × 256 | Mechanism |
|---|---:|---:|---|
| Intel Xeon 8481C Sapphire Rapids (22 vCPU) | **9.31× HF** (12.4 s) | 4.09× HF (0.92 s) | AMX-INT8, default |
| Apple M3 Pro (after one fetch step) | 3.97× HF (37.8 s) | **10.05× HF** (459 ms) | Whole-graph CoreML → ANE + GPU |
| GCP C4A / Neoverse V2 (8 vCPU, Graviton-class) | **5.04× HF** (29.3 s) | 2.30× HF (2.02 s) | NEON SDOT, default |
| Apple M3 Pro (default, no fetch) | 3.97× HF (37.8 s) | 1.79× HF (2.17 s) | NEON SDOT, default |

Two structural wins drive these numbers.

**Variable-length packed-batch scheduling** (the x86 + Linux ARM headline).
HuggingFace pads every sequence in a batch to `max(len)`. esm.cpp packs
sequences back-to-back along the token axis and isolates per-sequence
attention via `cu_seqlens`. On antibody-shaped data (mean ~120 residues,
max ~250) that saves roughly 3× of HF's attention compute and 2× of its
FFN compute on top of the INT8 baseline.

**Whole-graph CoreML compilation** (the Apple uniform-shape headline).
The entire ESM-2 forward — 33 encoder layers + LM head — is compiled into
ONE `.mlmodelc` at convert time and dispatched through an Objective-C++
MLModel bridge. Keeps one op-fused fp16 graph hot on the Apple Neural
Engine + GPU instead of the per-Linear pattern that thrashes the ANE
compiled-state cache. Logit correlation vs HF FP32 is 0.999998 at 650M
with strict pseudo-perplexity drift below 0.001 across the holdout
subset.

## How it works

- **Kernels.** Hand-written, runtime-dispatched per ISA. x86: AVX-512 +
  VNNI + AMX-INT8. ARM: NEON FMLA + SDOT + opt-in SMMLA/i8mm. Every
  vectorized kernel has a scalar-reference twin behind `#ifdef
  ESM_KERNEL_REFERENCE`; the same tests cross-check both with strict
  tolerances (FP32 `rtol/atol=1e-6`, INT8 `rtol=1e-3 atol=1`).
- **Quantization.** Ahead-of-time W8A8 INT8 with SmoothQuant, calibrated
  on UniRef50. Pseudo-perplexity drift < 0.1 and ProteinGym Spearman
  drift < 0.01 are non-negotiable gates.
- **Scheduler.** Variable-length sequences pack back-to-back along the
  token axis; attention isolates per-sequence via `cu_seqlens`; FFNs see
  one fused `[ΣL, d]` matmul instead of `B` padded `[L_max, d]` ones.
- **Loaders.** Both safetensors (HF native, zero-copy mmap) and GGUF
  (esm.cpp native, block-decoded INT8). Weight tensors are never copied
  into RAM at load.
- **Apple path.** On Apple Silicon, if pre-built artifacts are on disk,
  `Model.load_*` auto-engages them: a whole-graph CoreML model for
  registered uniform shapes, and a per-Linear AMX-fp16 BNNSGraph stack
  for everything else. With no artifacts installed, the engine falls
  back to the same NEON SDOT kernels Linux ARM uses. No Apple framework
  appears in the canonical kernel stack itself.

## Install

```bash
pip install esm-cpp
```

That's the whole install on Linux x86, Linux ARM, and as a working
baseline on Apple Silicon. On Apple Silicon, one extra step pulls the
pre-built whole-graph + AMX artifacts that unlock the 10× headline:

```bash
esm-cpp-fetch-artifacts --model esm2_t33_650M
```

The artifacts land in `~/.cache/esm_cpp/<model>/` (~5 GB for 650M) and
`Model.load_*` auto-discovers them on every load. The fetch CLI is pure
stdlib — no coremltools, torch, or transformers needed at user time.

| OS / arch | What `pip install esm-cpp` gets you | Extra step for headline |
|---|---|---|
| Linux x86_64 (Sapphire Rapids+) | AMX-INT8 → 9.31× HF varlen, 4.09× HF uniform | none |
| Linux x86_64 (Cascade Lake / Ice Lake) | AVX-512 + VNNI INT8 baseline | none |
| Linux ARM64 (Graviton 3/4, Axion, Ampere) | NEON SDOT → 5.04× HF varlen, 2.30× HF uniform | none |
| Apple Silicon (M1 / M2 / M3) | NEON SDOT → 3.97× HF varlen, 1.79× HF uniform | `esm-cpp-fetch-artifacts` → **10.05× HF uniform** |

## Quick start

```python
import esm_cpp

# Load FP32 weights from a HF safetensors file. On Apple Silicon, if you
# ran esm-cpp-fetch-artifacts, this auto-engages the whole-graph CoreML
# path — no env vars, no register calls.
model = esm_cpp.Model.load_from_safetensors(
    "/path/to/esm2_t33_650M_UR50D/model.safetensors")
tokenizer = esm_cpp.Tokenizer()

# Single sequence.
ids = tokenizer.encode("MKTGVAQRLELDSPMVLQKRSGE")
logits = model.forward(ids)  # [seq_len, vocab_size=33]

# Packed batch — variable-length sequences in one forward.
seqs = ["MKTGVA", "MAGAASPCANGCGPSAPS", "MSEEKRGGQATKLP"]
batch_ids = [tokenizer.encode(s) for s in seqs]
batch_logits = model.forward_scheduled(batch_ids)
# returns list of [L_i, vocab_size] arrays in input order
```

The same `Model.load_from_safetensors` (or `Model.load_from_gguf` for the
quantized artifact) is the entry point on every supported OS; the
correct kernel path is selected at load time from `cpu_features` plus
whatever Apple artifacts are present.

## Convert + quantize

```bash
# 1. HF safetensors -> esm-cpp GGUF (FP32).
esm-cpp-convert --hf facebook/esm2_t30_150M_UR50D --out weights/esm2_150m.gguf

# 2. Calibrate on UniRef50.
esm-cpp-quantize --calibrate \
    --model esm2_t30_150M \
    --calib data/uniref50_calib_v1.fasta \
    --out weights/esm2_150m_calib.json

# 3. Apply SmoothQuant + quantize to INT8.
esm-cpp-quantize --apply-smoothquant \
    --model esm2_t30_150M \
    --stats weights/esm2_150m_calib.json \
    --alpha 0.5 \
    --output weights/esm2_150m_q8.gguf
```

Then load the quantized artifact directly:

```python
m = esm_cpp.Model.load_from_gguf("weights/esm2_150m_q8.gguf")
assert m.config.weights_quantized
```

## Benchmark vs HuggingFace

```bash
esm-cpp-bench \
    --model facebook/esm2_t6_8M_UR50D \
    --dataset data/oas_sample_v1.fasta \
    --modes esm-cpp-fp32,hf-eager-fp32 \
    --output benchmarks/results/my_run.json
```

`--modes` accepts any subset of `esm-cpp-fp32`, `esm-cpp-int8`,
`hf-eager-fp32`, `hf-sdpa-fp32`. The harness reports p50 / p90 /
throughput on uniform and varlen workloads and writes a JSON record per
run for tracking.

## Scope

Inference only. No training, no LoRA, no backward pass — those belong
upstream. Hardware targets are x86_64 (AVX-512 / VNNI / AMX) and AArch64
(NEON / SDOT / SMMLA, plus opt-in Apple ANE/AMX via CoreML artifacts on
Apple Silicon). GPU, ARM SVE2, and RISC-V are out of scope for v0.x.
ESM-2-15B is bandwidth-bound on CPU at FP32/INT8 and needs W4 quant
before it makes sense; that's v2 work.
