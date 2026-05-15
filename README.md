# esm.cpp

A CPU-first inference engine for ESM-2 protein language models. Built in C++ for production throughput on commodity x86 CPUs, with Python bindings for direct use in research and discovery pipelines.

## What it does

esm.cpp runs ESM-2 forward passes on CPU. It targets workloads where throughput-per-dollar across millions of sequences matters more than per-sequence GPU latency:

- **Deep mutational scanning and variant-effect prediction.** Score 10⁴–10⁷ mutants per protein for masked-marginal likelihood, zero-shot fitness landscapes, or alongside FoldX/Rosetta.
- **Antibody developability screening.** Rank 10⁵–10⁶ candidate sequences by PLM likelihood for aggregation, solubility, or immunogenicity prefiltering.
- **Embedding extraction at corpus scale.** Per-residue or per-sequence vectors for downstream classifiers, alignment, or retrieval.
- **On-prem and regulated environments.** Runs on CPU without cloud GPU access — useful for clinical, compliance-restricted, or air-gapped pipelines.

## Supported models

ESM-2 at 8M, 35M, 150M, 650M, and 3B parameters. Weights load directly from a HuggingFace model ID or a local `safetensors` path — no conversion step. ESM-2-architecture antibody models (AbLang, AntiBERTy, AntiBERTa, IgBERT, BALM) work through the same path.

## Features

- AVX-512 + VNNI primary code path; AMX `TDPBSSD` runtime-gated for Sapphire Rapids and newer
- INT8 W8A8 quantization with SmoothQuant calibration for 150M–3B; FP32 for smaller models
- `cu_seqlens`-packed batched attention — no wasted compute on padding in mixed-length workloads
- pip-installable Python package with a small, stable API
- HuggingFace weights load directly, no conversion step required

## Install

```bash
pip install esm-cpp
```

## Quick start

```python
import esm_cpp

model = esm_cpp.Model.load("facebook/esm2_t33_650M_UR50D")
tokenizer = esm_cpp.Tokenizer()

ids = tokenizer.encode("MKTGVAQRLELDSPMVLQKRSGE")
logits = model.forward(ids)            # [seq_len, 33]
```

Batched scoring on mixed-length inputs uses a `cu_seqlens` packed layout to avoid padding overhead:

```python
logits, cu_seqlens = model.forward_batch(sequences)
```