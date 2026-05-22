# esm.cpp

A CPU-first C++ inference engine for ESM-2 protein language models, with Python bindings. Built for production throughput on commodity x86 CPUs.

## What it does

esm.cpp runs ESM-2 forward passes on CPU. The defensible niche is the intersection no existing project occupies: **production-grade CPU inference + ahead-of-time W8A8 quantization + variable-length packed-batch scheduling** for encoder-only PLMs.

Target workloads:

- **Deep mutational scanning** (10⁴–10⁷ variants per protein) — zero-shot fitness or masked-marginal scoring across millions of mutants.
- **Antibody developability screening** (10⁵–10⁶ candidates) — rank by PLM likelihood for aggregation / solubility / immunogenicity prefiltering.
- **Embedding extraction at corpus scale** — per-residue or per-sequence vectors for downstream classifiers, alignment, or retrieval.
- **On-prem / regulated environments** — runs on CPU without cloud GPU access. Useful for clinical, compliance-restricted, or air-gapped pipelines.

## Status: v0.1.0

ESM-2 at 8M, 35M, 150M, 650M, 3B. W8A8 INT8 with SmoothQuant ships for 150M and above; smaller models stay FP32. Both safetensors (HF native) and GGUF (esm.cpp native) load paths.

**Headline performance** (Intel Xeon 8481C Sapphire Rapids, AMX-INT8, 22 vCPUs, ESM-2-650M):

| Workload | esm-cpp-int8 | hf-eager-fp32 | Speedup |
|---|---:|---:|---:|
| Variable-length 256-seq (OAS-shape) | **12.4 s** | 115.1 s | **9.31× HF** |
| Variable-length + opt-in env vars   | 11.5 s | 115.0 s | **10.04× HF** |
| Uniform 8-seq × 256-tokens          | 0.92 s | 3.42 s | 4.09× HF |

The variable-length advantage comes from the `cu_seqlens` packed-batch forward: HuggingFace pads every sequence in a batch to `max(len)` and processes the resulting padded tensor uniformly; esm.cpp packs sequences back-to-back along the token axis and isolates per-sequence attention via `cu_seqlens`. On antibody-shaped data (mean ~120 residues, max ~250) that saves ~3× of HF's attention compute and ~2× of its FFN compute on top of the INT8/AMX baseline.

### ARM (Apple Silicon / Linux ARM)

esm.cpp also runs on AArch64 with a hand-written NEON kernel stack — FMLA FP32, SDOT INT8 (the VNNI analog), and an opt-in SMMLA/i8mm path (the AMX analog) — runtime-dispatched the same way as x86. No Apple Accelerate dependency in the default build, so it runs on Linux ARM / AWS Graviton too.

**Apple M3 Pro, ESM-2, esm-cpp-int8 (NEON SDOT) vs HF eager FP32:**

| Workload | esm-cpp-int8 | hf-eager-fp32 | Speedup |
|---|---:|---:|---:|
| Variable-length 256-seq (OAS-shape), 8M | **1.02 s** | 3.28 s | **3.21× HF** |
| Variable-length 256-seq (OAS-shape), 35M | **3.20 s** | 9.46 s | **2.95× HF** |
| Uniform 8-seq × 100-tokens, 8M | 30.5 ms | 34.5 ms | 1.13× HF |
| Uniform 8-seq × 100-tokens, 35M | 78.5 ms | 99.8 ms | 1.27× HF |

The SMMLA/i8mm tier is opt-in (`ESM_NEON_I8MM=on`): on Apple M3 it does not out-throughput SDOT, but it is expected to win on Graviton3-class cores with stronger i8mm units. INT8 quality on ARM matches FP32 — 0.9999 logit correlation, 100% masked-marginal argmax agreement.

## Install

```bash
pip install esm-cpp
```

## Quick start

```python
import esm_cpp

# Load FP32 weights directly from a HF safetensors file (or shorthand).
model = esm_cpp.Model.load_from_safetensors(
    "/path/to/esm2_t6_8M_UR50D/model.safetensors")
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

## Convert + quantize

Convert an HF checkpoint to GGUF and quantize:

```bash
# 1. Convert HF safetensors -> esm-cpp GGUF (FP32).
esm-cpp-convert --hf facebook/esm2_t30_150M_UR50D --out weights/esm2_150m.gguf

# 2. Calibrate on UniRef50 sequences.
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