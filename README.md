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

ESM-2 at 8M, 35M, 150M, 650M, 3B. The W8A8 INT8 path with SmoothQuant ships for 150M and above; smaller models stay FP32. Both safetensors (HF native) and GGUF (esm.cpp native) load paths.

The Phase 3 ship includes the cu_seqlens packed-batch scheduler, the public benchmark harness, and the GGUF reader/writer. The headline VNNI/AMX SIMD microkernels are scaffolded; the production x86 perf numbers are a gate-machine hand-off — see [docs/benchmarks.md](docs/benchmarks.md) for the dev-host baseline and the reproduction commands for the gate run.

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

See [docs/benchmarks.md](docs/benchmarks.md) for the reproduction commands and the current results table.

## Documentation

- [docs/architecture.md](docs/architecture.md) — system overview, dispatch facade, packed cu_seqlens scheduler.
- [docs/quant-recipe.md](docs/quant-recipe.md) — W8A8 + SmoothQuant recipe, escape list, α-sweep procedure.
- [docs/benchmarks.md](docs/benchmarks.md) — reproduction commands and results.
- [SPEC.md](SPEC.md) — the original spec; phase gates and out-of-scope decisions.
- [CLAUDE.md](CLAUDE.md) — engineering practices (kernel discipline, comment policy, threading rules).

## Non-goals

- Training, fine-tuning, LoRA, gradients — [ESME](https://github.com/qingyuan-hou/esme) owns that on GPU.
- Decoder-only / generative models — encoder-only PLMs only.
- GPU backend — v2 or later.
- INT4 weight-only quant (GPTQ / AWQ) — v2 stretch goal for 3B / 15B.

## License

Apache-2.0.
