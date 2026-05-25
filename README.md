# esm.cpp

A CPU-first C++ inference engine for ESM-2 protein language models, with Python bindings. Built for production throughput on commodity x86 CPUs.

## What it does

esm.cpp runs ESM-2 forward passes on CPU. The defensible niche is the intersection no existing project occupies: **production-grade CPU inference + ahead-of-time W8A8 quantization + variable-length packed-batch scheduling** for encoder-only PLMs.

Target workloads:

- **Deep mutational scanning** (10⁴–10⁷ variants per protein) — zero-shot fitness or masked-marginal scoring across millions of mutants.
- **Antibody developability screening** (10⁵–10⁶ candidates) — rank by PLM likelihood for aggregation / solubility / immunogenicity prefiltering.
- **Embedding extraction at corpus scale** — per-residue or per-sequence vectors for downstream classifiers, alignment, or retrieval.
- **On-prem / regulated environments** — runs on CPU without cloud GPU access. Useful for clinical, compliance-restricted, or air-gapped pipelines.

## Status: v0.2

ESM-2 at 8M, 35M, 150M, 650M, 3B. W8A8 INT8 with SmoothQuant ships for 150M and above; smaller models stay FP32. Both safetensors (HF native) and GGUF (esm.cpp native) load paths.

**Headline performance**:

- **Intel Xeon 8481C Sapphire Rapids** (AMX-INT8, 22 vCPUs, ESM-2-650M):

  | Workload | esm-cpp-int8 | hf-eager-fp32 | Speedup |
  |---|---:|---:|---:|
  | Variable-length 256-seq (OAS-shape) | **12.4 s** | 115.1 s | **9.31× HF** |
  | Variable-length + opt-in env vars   | 11.5 s | 115.0 s | **10.04× HF** |
  | Uniform 8-seq × 256-tokens          | 0.92 s | 3.42 s | 4.09× HF |

- **Apple M3 Pro** (ANE + GPU via CoreML, ESM-2-650M, **`ESM_APPLE_ANE_GRAPH=on`**):

  | Workload | esm-cpp-whole-graph-fp16 | hf-eager-fp32 | Speedup |
  |---|---:|---:|---:|
  | Uniform 8-seq × 256-tokens | **459 ms** | 4617 ms | **10.05× HF** |

The variable-length advantage on x86 comes from the `cu_seqlens` packed-batch forward: HuggingFace pads every sequence in a batch to `max(len)` and processes the resulting padded tensor uniformly; esm.cpp packs sequences back-to-back along the token axis and isolates per-sequence attention via `cu_seqlens`. On antibody-shaped data (mean ~120 residues, max ~250) that saves ~3× of HF's attention compute and ~2× of its FFN compute on top of the INT8/AMX baseline.

The Apple M3 uniform-shape win comes from compiling the entire ESM-2 forward (33 encoder layers + LM head) into ONE CoreML `.mlmodelc` at convert time and routing it through a small Obj-C++ MLModel bridge — keeping one op-fused fp16 graph hot on ANE/GPU instead of the per-Linear pattern that thrashes the ANE compiled-state cache. The path is opt-in (the default is unchanged) and only engages on uniform-length batches with a registered shape; varlen / mixed-length falls through to the Phase-11 AMX-fp16 path. See `docs/benchmarks.md` for the full path table and reproduction.

### ARM (Apple Silicon / Linux ARM)

esm.cpp also runs on AArch64 with a hand-written NEON kernel stack — FMLA FP32, SDOT INT8 (the VNNI analog), and an opt-in SMMLA/i8mm path (the AMX analog) — runtime-dispatched the same way as x86. No Apple Accelerate dependency in the default build, so it runs on Linux ARM / AWS Graviton too.

**Apple M3 Pro, ESM-2, esm-cpp vs HF eager FP32:**

| Workload | esm-cpp | hf-eager-fp32 | Speedup |
|---|---:|---:|---:|
| Uniform 8-seq × 256-tokens, 650M (with `esm-cpp-fetch-artifacts`) | **459 ms** | 4617 ms | **10.05× HF** |
| Uniform 8-seq × 256-tokens, 650M (NEON SDOT, no fetch) | 2.17 s | 3.88 s | 1.79× HF |
| Variable-length 256-seq (OAS-shape), 650M (SDOT, Phase-9) | **37.8 s** | 150.1 s | **3.97× HF** |
| Variable-length 256-seq (OAS-shape), 35M | **3.20 s** | 9.46 s | **2.95× HF** |
| Variable-length 256-seq (OAS-shape), 8M | **1.02 s** | 3.28 s | **3.21× HF** |
| Uniform 8-seq × 100-tokens, 35M | 78.5 ms | 99.8 ms | 1.27× HF |
| Uniform 8-seq × 100-tokens, 8M | 30.5 ms | 34.5 ms | 1.13× HF |

The Phase-13 **uniform-shape headline** comes from a whole-graph CoreML path: ONE compiled `.mlmodelc` for the entire ESM-2 forward (33 encoder layers + LM head), dispatched to the Neural Engine + GPU via CoreML through a small Objective-C++ MLModel bridge. As of Phase 14, **a single `esm-cpp-fetch-artifacts --model esm2_t33_650M` download is the only setup the user needs** — Model::Load* auto-discovers the artifacts and routes through them in the forward; no env vars, no manual register calls. 650M @ uniform 8×256 hits 459 ms p50 (**10.05× HF**), with quality vs FP32 strictly better than INT8 SDOT (corr 0.999998 vs 0.99956 on 650M; PPPL drift < 0.001 across the 25-protein holdout subset).

The path engages on Apple Silicon when (a) artifacts are present (sibling `<weights>.apple/` or `~/.cache/esm_cpp/<key>/`) and (b) the batch has uniform sequence length matching a registered shape. Mixed-length / varlen batches fall through to the Phase-11 AMX-fp16 path (per-Linear fp16 BNNSGraph artifacts; same fetch CLI installs them), which delivers 2.23× HF at 650M uniform and 4.53× HF at 650M varlen. Without any artifacts installed, the **NEON SDOT default** (Phase-10: branch-hoist + register-resident attention PV) gives 1.79× HF at 650M uniform / 3.97× HF at 650M varlen — and is the only path on Linux ARM / AWS Graviton (the whole AMX + ANE backends are `#ifdef`'d out on non-Apple builds — hand-written NEON / SDOT stays the default and only Linux-ARM path). The SMMLA/i8mm tier is opt-in (`ESM_NEON_I8MM=on`): on Apple M3 it does not out-throughput SDOT, but is expected to win on Graviton3-class cores. Set `ESM_APPLE_AMX=off` / `ESM_APPLE_ANE_GRAPH=off` to disable the auto-engage paths for debugging.

## Install

```bash
pip install esm-cpp
```

On Apple Silicon, one extra step gets you the 10× headline:

```bash
# Pulls pre-built whole-graph + AMX artifacts (~5 GB for 650M) from
# the GitHub release matching your esm-cpp version. Zero coremltools,
# torch, or transformers install required at user time.
esm-cpp-fetch-artifacts --model esm2_t33_650M
```

The artifacts land in `~/.cache/esm_cpp/<model>/` and `Model.load_*`
auto-discovers them on every load. Linux ARM / AWS Graviton skip this
step — the runtime uses the same wheel and gets the NEON SDOT path
(no Apple frameworks required).

## Quick start

```python
import esm_cpp

# Load FP32 weights from a HF safetensors file. On Apple Silicon, if
# you ran `esm-cpp-fetch-artifacts`, this auto-engages the whole-graph
# CoreML path — no env vars, no register calls.
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

# Diagnose which Apple paths auto-engaged (empty strings = nothing loaded):
print("whole-graph shapes :", model.whole_graph_shapes)
print("amx artifacts dir  :", model.amx_artifacts_path)
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