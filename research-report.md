# esm.cpp: A Comprehensive Research Report on Building a High-Performance C++ Inference Engine for ESM-2

## TL;DR
- **The project is viable and timely, but only narrowly novel.** ESME (Celik & Xie, *iScience* 28(10):113495, Sep 2025) already delivers 4–9× faster inference and 3–14× lower memory on GPU via FlashAttention + sequence packing; BioNeMo Framework (Bagusetty et al., arXiv:2411.10548) hits 59.2% model-FLOPs-utilization on A100 at batch 46 for ESM-2 650M; and llama.cpp's `llm_build_bert` graph plus the existing BERT-embeddings path (`bert 33M F16` at 26,629 tok/s pp128 on Apple Metal, Discussion #7712) show ggml can host encoder-only PLMs. There is **no production-grade CPU-first, INT8-quantized, continuously-batched C++ implementation of ESM-2** — that is the defensible niche.
- **The single largest engineering risks are (a) ESM-2's half-then-half RoPE convention paired with its `token_dropout` 0.88-rescaling quirk, (b) outlier activations in early transformer layers, and (c) padding waste of 50–80% on real antibody workloads.** Each has a named mitigation (Llama-style `rotate_half`, SmoothQuant α≈0.5, FlashAttention varlen with `cu_seqlens`).
- **A realistic target on a modern AVX-512+VNNI CPU is sub-PyTorch latency for ESM-2 ≤650M at batch 1–16**, especially for short antibody sequences (≤256 aa). For 3B and 15B the workload is memory-bandwidth-bound and INT4 weight-only quantization (AWQ/GPTQ) matters more than INT8 W8A8.

## Key Findings

### 1. ESM-2 Architecture — Exact Specs

Authoritative from `facebookresearch/esm` source (`esm/model/esm2.py`, `esm/modules.py`, `esm/rotary_embedding.py`) and matching HuggingFace `EsmConfig`. FFN is always 4× hidden, activation is **GELU**, architecture is **Pre-LN**, **LM head weights are tied** to input embeddings (`RobertaLMHead(..., weight=self.embed_tokens.weight)`), with a final LayerNorm after the last transformer block, and **no learned positional embeddings** (RoPE only):

| Model | Layers | hidden | heads | head_dim | FFN | Params |
|---|---|---|---|---|---|---|
| esm2_t6_8M  | 6  | 320  | 20 | 16  | 1280  | 8M |
| esm2_t12_35M | 12 | 480  | 20 | 24  | 1920  | 35M |
| esm2_t30_150M | 30 | 640  | 20 | 32  | 2560  | 150M |
| esm2_t33_650M | 33 | 1280 | 20 | 64  | 5120  | 650M |
| esm2_t36_3B | 36 | 2560 | 40 | 64  | 10240 | 3B |
| esm2_t48_15B | 48 | 5120 | 40 | 128 | 20480 | 15B |

**Tokenizer (vocab_size = 33), order matters:** `<cls>=0, <pad>=1, <eos>=2, <unk>=3, L=4, A=5, G=6, V=7, S=8, E=9, R=10, T=11, I=12, D=13, P=14, K=15, Q=16, N=17, F=18, Y=19, M=20, H=21, W=22, C=23, X=24, B=25, U=26, Z=27, O=28, .=29, -=30, <null_1>=31, <mask>=32`. The 20 canonical amino acids are listed in **UR50 frequency order**, not alphabetical — an easy bug. `prepend_bos=True`, `append_eos=True`, `model_max_length=1024` (1022 residues + `<cls>` + `<eos>`); HF config's `max_position_embeddings=1026` follows the RoBERTa convention.

**RoPE — three critical gotchas:**
1. **Rotation convention is half-then-half (Llama / GPT-NeoX style), NOT interleaved pairs (RoFormer / GPT-J).** From `esm/rotary_embedding.py`: `rotate_half(x): x1, x2 = x.chunk(2, dim=-1); return torch.cat((-x2, x1), dim=-1)`, with `freqs` concatenated as `torch.cat((freqs, freqs), dim=-1)`. Misreading this is the #1 silent-failure mode for ESM ports.
2. RoPE is applied **on every layer**, to Q and K independently, after the QKV projection but before the attention scores. Base θ = 10000, with `inv_freq[i] = 1/10000^(2i/dim)` for `i ∈ [0, head_dim/2)`.
3. **ESM-2 scales the query by `1/√head_dim` *before* RoPE**, not the score after — HuggingFace's port explicitly comments: *"ESM scales the query down by the same factor instead. Modulo numerical stability these are equivalent, but not when rotary embeddings get involved."* For bit-exact equivalence to the official checkpoint, this order matters.

**`token_dropout` quirk (non-standard, easily missed):** ESM-2 zeroes mask-token embeddings and rescales **all** embeddings by `(1 - 0.15·0.8) / (1 - observed_mask_fraction) = 0.88 / (1 - observed_mask_fraction)`. At inference with zero masks, every embedding is silently multiplied by **0.88**. Source: `esm/model/esm2.py` (`x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]`). Skipping this produces qualitatively correct but quantitatively wrong logits.

### 2. Prior Work — What Already Exists

- **Official Meta `facebookresearch/esm`** (archived August 2024, Apache-2.0): reference PyTorch; naive attention; FSDP CPU-offload only for 15B.
- **HuggingFace `EsmModel`** (`transformers/src/transformers/models/esm/modeling_esm.py`): the canonical port; supports FA-2 via `attn_implementation="flash_attention_2"`.
- **NVIDIA BioNeMo Framework** (Bagusetty et al., arXiv:2411.10548): converted checkpoints for 650M / 3B / 15B with TransformerEngine layers; reports **59.2% MFU on A100 at batch 46** for 650M and 96.85% strong-scaling linearity to 256 × A100 for 3B training. GPU-only.
- **ESME (Celik & Xie, *iScience* 28(10):113495, Sep 3 2025; doi:10.1016/j.isci.2025.113495)** — closest prior art: PyPI package `esm-efficient` (github.com/uci-cbcl/esm-efficient). Adds FlashAttention plus a novel **Partition-Attention** for variable-length proteins, plus LoRA fine-tuning. Reports **"4–9× faster inference and 3–14× lower memory usage"** with linear (vs. quadratic) scaling in sequence length; e.g., 650M on batch 16 × 300–400aa drops from ~0.21 s to ~0.07 s on A100. 8-bit / 4-bit quantization is shown to be useful only at the billion-parameter scale and degrades quality below 150M. **GPU-only**, PyTorch, no C++.
- **llama.cpp / ggml**: BERT encoder support landed in PR #5423; the `llm_build_bert` graph in `src/llama-model.cpp` is the reference for adding new encoder-only architectures (per `docs/development/HOWTO-add-model.md`). The embeddings tutorial (Discussion #7712) measured `bert 33M F16` at **26,629.51 ± 133.81 tok/s for pp128** and **30,843.16 ± 105.98 tok/s for pp256** on Apple Metal. **ESM-2 is not supported in mainline llama.cpp** as of May 2026 — adding it requires a new `LLM_ARCH_ESM` enum, a converter pass in `convert_hf_to_gguf.py`, and a graph builder that wires RoPE into the encoder Pre-LN path. T5 encoder-only was requested in Issue #8900 and only partially landed. This is a real, capturable contribution.
- **vLLM, TGI**: no native ESM-2 / encoder-only PLM support. vLLM is decoder-centric; PagedAttention's value is in KV-cache paging for autoregressive generation, which ESM-2 has none of.
- **Quantization for PLMs specifically**: sparse. "Scaling Up ESM2 Architectures for Long Protein Sequences" (arXiv:2501.07747) and the ESME paper are the main published references; both are PyTorch + bitsandbytes (W4/W8 weight-only) and report degraded quality below the 150M scale.

### 3. Antibody PLMs — Production Landscape

All BERT/RoBERTa-style encoder-only and structurally near-identical to ESM-2, so the same engine serves them all:
- **AbLang / AbLang2** (Olsen, Moal, Deane — OPIG Oxford, 2022): 12 layers, 768 hidden, 12 heads, learned positional embeddings, trained on 14M heavy + 187K light chains.
- **AntiBERTy** (Ruffolo et al., 2021): 26M params, trained on 558M OAS sequences.
- **AntiBERTa / AntiBERTa2 / AntiBERTa2-CSSP** (Leem et al., now hosted under `alchemab` on HuggingFace): 86M, structure-aware; powers **Alchemab Therapeutics**' commercial antibody discovery.
- **Sapiens** (Prihoda et al., Merck): two 569K-param models for humanization.
- **IgLM** (Ruffolo et al.): 13M GPT-style decoder for infilling (the only non-encoder here).
- **AbMAP** (Singh lab, PNAS 2024): adapter on top of ESM-2.
- **BALM, p-IgGen, S2ALM, IgT5, IgBERT, Ab-RoBERTa** (Mogam AI, 402M antibody sequences) — all encoder-only.

**Industry users that bottleneck on PLM inference**: Alchemab, Absci, Generate Biomedicines, Genentech (Sapiens lineage), Janssen (AntiBERTa lineage), Prescient/Schrödinger, OpenProtein.AI, Cradle Bio, Profluent, EvolutionaryScale, and most large pharma's internal pipelines. The user's own work at Aghazadeh Lab on antibody PLMs is squarely in this set. The dominant CPU-suitable bottleneck is **large-scale variant scoring** (DMS-style enumerations producing 10⁵–10⁷ sequences) where each call is a forward pass.

### 4. CPU GEMM Landscape & Realistic Peak

- **State of the art**: Intel **MKL/oneDNN** and **BLIS** hit **>90% of single-core peak FLOPS** for SGEMM/DGEMM on AVX-512 hardware (BLIS `docs/Performance.md`). **OpenBLAS lags on AVX-512** because of incomplete microkernel coverage. **libxsmm** is the choice for the small-GEMM regime that dominates ESM-2 attention at low batch.
- **Algorithm**: Goto-style packing — pack A into K_C × M_C in L2, B into K_C × N_C panels in L3, register-blocked microkernel (typically 6×16 for AVX-2 SP, 16×32 or 24×16 for AVX-512). Walkthroughs at `salykova.github.io/matmul-cpu` and `yzhaiustc/Optimizing-DGEMM-on-Intel-CPUs-with-AVX512F` reach ~90% of MKL in pure C/intrinsics.
- **Theoretical peak**: a single core at 4 GHz with 2× 512-bit FMA units = 2(FMA) × 2(units) × 16(FP32 lanes) × 4 GHz ≈ **256 GFLOPS FP32** (128 GFLOPS FP64). For INT8 with VNNI: one `VPDPBUSD` does 64 INT8 MACs/cycle/unit → **~1 TOPS/core**.
- **AVX-512 VNNI (`VPDPBUSD`)**: u8×s8 → s32 accumulator, single-cycle throughput on most ports, replaces the older 3-instruction `vpmaddubsw + vpmaddwd + vpaddd`. Available on Cascade Lake onwards and on **AMD Zen 6** (Znver6 ISA manual confirms AVX_VNNI_INT8 (256-bit) and AVX512_FP16 via GCC patches; Zen 6 expected late 2026/early 2027, codenames Medusa/Venice). Note Zen 6's AVX_VNNI_INT8 is 256-bit, not full AVX-512 VNNI.
- **Intel AMX (Sapphire Rapids, 2023+)**: 8 tile registers × 1 KB; `TDPBSSD` does up to **2048 INT8 ops/cycle** (256 J multiplications + 256 J additions in 16 cycles where J=64 for INT8) and 1024 BF16 ops/cycle — a 4–8× win over AVX-512 VNNI for INT8 matmul. Requires `syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`, 16-row max per tile, K-dim 64 INT8 or 32 BF16. Granite Rapids adds AMX-FP16. Single-core AMX BF16 peak ≈ 11.9 TFLOPS at 3.4 GHz; one socket Xeon Platinum 8480 ≈ 0.66 PFLOPS BF16 (theoretical).
- **ARM**: Apple Silicon exposes an Apple-private "AMX coprocessor" via Accelerate's `cblas_sgemm`. ARM **SME (Scalable Matrix Extension)** on Neoverse V3/V4 and the Apple M4 generation standardizes matrix tiles. For Graviton 3/4 and most antibody-lab Macs, the realistic baseline is **NEON dot-product (`sdot`/`udot`)** and **SVE2 BFMMLA**.

### 5. INT8 Quantization — Decision Matrix

- **Per-channel symmetric weight INT8** is the production standard; per-tensor activation INT8 fails on outliers.
- **LLM.int8() (Dettmers et al., NeurIPS 2022)**: keeps ~0.1% outlier dims in FP16, INT8 the rest. Memory only — *not faster* than FP16.
- **SmoothQuant (Xiao et al., ICML 2023, arXiv:2211.10438)**: offline migration of activation outliers into weights via diagonal scale `s` (α≈0.5 typical for OPT/BLOOM). Enables W8A8 with up to **1.56×** speedup and half memory in FasterTransformer — the right baseline for esm.cpp's static-quant path.
- **GPTQ (Frantar et al., ICLR 2023, arXiv:2210.17323)**: layer-wise weight-only with Hessian-guided rounding; dominant for INT4 weights × FP16 activations.
- **AWQ (Lin et al., MLSys 2024, arXiv:2306.00978)**: activation-aware salient-channel weight protection; slightly higher quality than GPTQ at INT4 and easier to fuse on CPU.
- **llama.cpp k-quants (Q4_K, Q5_K, Q6_K, Q8_0)**: 32-element block superblocks with mixed bit widths and importance-matrix calibration; battle-tested CPU paths in ggml's `vec_dot` kernels (AVX-2/AVX-512/NEON/AMX).
- **Protein-LM specific evidence**: ESME shows W8/W4 useful only at ≥150M; below that, INT4 collapses. This argues **W8A8 + SmoothQuant for the 35M–650M sweet spot** and **W4A16 (GPTQ/AWQ) for 3B/15B**. Outliers in early layers of large transformers (Bondarenko et al. 2021, Dettmers et al. 2022) appear in ESM-2 too; per-channel observers and α-tuning are essential.

### 6. Continuous Batching / Variable-Length Attention

- **vLLM PagedAttention (Kwon et al., SOSP 2023, arXiv:2309.06180)**: pages KV cache in 16-token blocks. The paper documents that **"only 20.4%–38.2% of the KV cache memory is used to store the actual token states in the existing systems"** — i.e., 62–80% memory waste from fragmentation/pre-allocation. For encoder-only ESM-2 there is no KV cache to page, but the *continuous-batching scheduler* concept (iteration-level admission) still applies to batched scoring services.
- **FlashAttention varlen (`flash_attn_varlen_func`)**: packed `q/k/v` 1-D tensors plus `cu_seqlens` and `max_seqlen`. This is the same API ESME uses for "Partition-Attention" and the shape esm.cpp's attention kernel should expose.
- **NestedTensor in PyTorch** + IBM's `DataCollatorWithFlattening` provide drop-in padding-free batching for MLMs. IBM Research's blog post on packing with FA-2 reports **~2× throughput** on Llama2-7B, Mistral-7B, and Granite-8B-code with FLAN data — directly transferable to OAS antibody batches.
- **Padding waste in protein workloads**: typical OAS antibody batches are 110–135 aa (heavy) and 105–115 aa (light) with occasional 250+ aa outliers. Naive padding-to-max in a 32-sequence batch spends **50–80% of FLOPs on `<pad>`**. Length-bucketed scheduling (sort by length, power-of-two bins) recovers most of this; full `cu_seqlens` packing recovers all.

### 7. Performance Targets & Validation

- **PyTorch eager baseline** (FP32 CPU, 650M, batch 1, 300aa): ~200–600 ms on a recent Xeon (no official number; internally observable).
- **ESME GPU baseline**: 650M, batch 16, 300–400aa → 0.07 s on A100 (≈220 seqs/s).
- **Realistic esm.cpp CPU targets** (Sapphire Rapids 1-socket, 32 cores, AMX-INT8):
  - 8M / 35M: hundreds of seqs/s at batch 1; compute-bound.
  - 150M / 650M: tens of seqs/s; mixed.
  - 3B: 1–5 seqs/s; memory-bandwidth-bound (3B × INT8 = 3 GB; at 200 GB/s DDR5 a single forward = ~15 ms minimum just for weight reads).
  - 15B: only viable with W4 quant (~8 GB) on a high-channel-count Xeon or Threadripper.
- **Quality validation**: replicate Lin et al. 2023's pseudo-perplexity (mask one residue at a time, sum NLLs, exponentiate the mean) on a 500–1000 held-out UniRef50 sample, and run **ProteinGym v1.3 (GitHub tag PG_v1.3, OATML-Markslab/ProteinGym; 217 DMS substitution assays, ~2.7M missense variants across 2,525 clinical proteins, 16 new baselines including ESM3 and ESM C)**, primary metric Spearman correlation, zero-shot masked-marginal scoring. Targets: PPPL drift < 0.1 vs FP32, Spearman drift < 0.01.

### 8. Industry Use Cases

- **Variant effect prediction / deep mutational scanning**: scoring 10⁴–10⁷ point mutants per protein. CPU inference wins when the throughput-per-dollar matters more than per-sequence latency, and when the workload runs alongside FoldX / Rosetta / structure relaxation. ProteinGym is the standard benchmark; ESM-2-650M is a top-3 zero-shot baseline.
- **Antibody developability screening**: ranking 10⁵–10⁶ candidates for aggregation, solubility, immunogenicity via PLM likelihood. Latency-sensitive in iterative library design.
- **Large-scale design pipelines**: Generate Biomedicines, EvolutionaryScale, Profluent, Cradle Bio, OpenProtein.AI all run ESM-family at scale and would benefit from a 2–5× CPU speedup that doesn't require A100s.
- **CPU-only consumers**: academic labs without GPU clusters (ESME's explicit motivation), regulated/clinical environments, and **on-prem at hedge funds / Millennium-style quant biotech** where compliance restricts cloud GPU use — directly relevant to the user's profile.

### 9. Implementation Risks & Pitfalls

- **RoPE convention**: #1 silent-bug source. Half-then-half vs interleaved gives equivalent attention only at position 0; for any longer sequence the logits diverge. Test with a 5-token sequence and bit-compare against `EsmModel` hidden states layer by layer.
- **`token_dropout` 0.88 scale**: forgetting it makes the model still look qualitatively right but gives ~10% wrong probabilities. Must be replicated.
- **Q-scale-before-RoPE**: scaling Q by `1/√d` *after* RoPE is numerically different in low-precision because RoPE is a rotation (norm-preserving only in exact arithmetic).
- **FP16 attention accumulation**: storing the softmax numerator/denominator in FP16 loses precision on long sequences. FlashAttention's recommendation (FP32 accumulator inside the inner block) is mandatory for INT8 attention.
- **Outliers in early layers**: ESM-2 inherits the transformer-outlier phenomenon (Dettmers 2022, Bondarenko 2021). Per-channel weight quant + SmoothQuant α≈0.5 on `fc1` is the baseline mitigation. Per-tensor activation quant on layer 0 will hurt PPPL by >0.5 points; static percentile-99.9 observers fix it.
- **Tokenizer pitfalls**: alphabet is **not** alphabetical; `<null_1>=31` is real; `B/U/Z/O/.` are valid (rare/special) tokens, not unknowns; `prepend_bos=True` and `append_eos=True` means the model never sees a raw `MKT...` — it sees `<cls> M K T ... <eos>`.
- **Memory-bandwidth vs compute analysis**:
  - 8M (16 MB FP32) — compute-bound at any batch ≥1.
  - 150M (300 MB FP32 / 150 MB INT8) — borderline.
  - 650M (1.3 GB FP32 / 650 MB INT8) — bandwidth-bound at batch 1, compute-bound at batch ≥16.
  - 3B and 15B — bandwidth-bound regardless of batch on commodity CPUs. INT4 weight-only is the only recovery.

### 10. Tooling & Ecosystem

- **GGUF**: works for encoder-only models (BERT and Nomic-embed prove it), but needs a new arch tag and converter (`convert_hf_to_gguf.py` extension). Block-quant superblocks (32 elements) play well with ESM-2 hidden dims (320/480/640/1280/2560/5120 — all multiples of 32 and 64).
- **Safetensors**: easiest source format; HuggingFace ships ESM-2 in safetensors natively.
- **ONNX**: `optimum-onnx` works for HF's `EsmModel`; ONNX Runtime with `OpenVINO EP` hits AMX automatically on SPR but locks you out of low-level kernel control — useful as a *baseline comparator*, not a target.
- **Benchmarking**: Google Benchmark (microkernels), nanobench (lower overhead), `perf stat` for cycles/IPC/cache misses, `llvm-mca` for static throughput analysis of microkernel asm.
- **Numerical-correctness testing**: golden tensors from `EsmModel(output_hidden_states=True)` at every layer + `allclose(rtol=1e-3, atol=1e-3)` for FP32; for INT8, validate PPPL drift and ProteinGym Spearman drift rather than element-wise tensors.

## Details

### A complete kernel inventory for esm.cpp
1. **embed_lookup** — gather rows from a `[33, d]` table; trivial.
2. **token_dropout_scale** — fused scalar multiply by `0.88 / (1 - observed_mask_frac)`.
3. **layernorm_fwd** (ε=1e-5, affine).
4. **qkv_gemm_packed** — one `[d, 3d]` matmul per layer; INT8 weight, FP32 accumulator.
5. **rope_apply_half** — fused on Q and K; precomputed cos/sin table per seqlen, half-then-half layout.
6. **attention_varlen** — FlashAttention-style block-streaming with `cu_seqlens`, FP32 softmax accumulator, INT8 K/V optional.
7. **out_proj_gemm** — `[d, d]`.
8. **ffn_fc1_gelu** — `[d, 4d]` matmul + fused GELU (tanh approximation matches PyTorch).
9. **ffn_fc2** — `[4d, d]`.
10. **final_layernorm**.
11. **lm_head** — `[d, d]` dense → GELU → LayerNorm → `[d, 33]` tied matmul + bias.

### Recommended INT8 quantization recipe
- Calibrate with 256–1024 random UniRef50 sequences (length 100–500).
- Per-channel symmetric INT8 on all Linear weights.
- Static per-tensor INT8 activations with **99.9th-percentile observer** *after* SmoothQuant migration (α=0.5; sweep α on PPPL).
- Keep `lm_head.dense` and `lm_head.layer_norm` in FP32 (cheap, sensitivity-critical).
- Keep **first transformer block's `fc1` input** in FP16 if PPPL drift > 0.2 — cheapest outlier mitigation if SmoothQuant alone is insufficient.

### Continuous-batching scheduler design
- Two queues: prefill-ready (new sequences) and in-flight (multi-step tasks like PPPL evaluators that need L masked passes).
- Pack up to `max_batched_tokens` (e.g. 8192) into one `cu_seqlens` batch.
- Length-bucket within a packed batch only when packed-work imbalance > 20% (stable microkernel tile sizes).
- For PPPL: a single length-L sequence generates L "virtual" sequences differing only in one masked position — these compress extremely well in a packed batch and are the killer demo for the project.

## Recommendations

### Stage 0 (Week 1–2): nail the reference
- Bit-equivalent FP32 forward of ESM-2-8M vs HF `EsmModel(output_hidden_states=True)` on 100 random sequences.
- **Threshold to advance: max abs diff < 1e-4 on final logits.** Catches RoPE convention, `token_dropout`, Q-scale-before-RoPE, and tokenizer-ordering bugs immediately.

### Stage 1 (Week 3–6): SIMD FP32 / BF16 baseline
- Goto-style packed SGEMM microkernel for AVX-512 (24×16 or 16×32). Target ≥80% of MKL on the four shapes that matter: `[B·L, d, 3d]`, `[B·L, d, d]`, `[B·L, d, 4d]`, `[B·L, 4d, d]` for `d ∈ {320, 640, 1280, 2560}`.
- FlashAttention-style packed-varlen kernel with FP32 accumulator.
- **Threshold to advance: ≥2× HuggingFace PyTorch FP32 on 650M, batch 16, 300aa, single socket.**

### Stage 2 (Week 7–10): INT8 + AMX
- SmoothQuant + per-channel W8A8.
- `VPDPBUSD` inner microkernel; AMX `TDPBSSD` path gated behind runtime detect (`cpuid -1 | grep AMX-INT8`).
- **Threshold to advance: PPPL drift < 0.1, ProteinGym Spearman drift < 0.01 vs FP32.**

### Stage 3 (Week 11–14): continuous batching + GGUF
- Implement `cu_seqlens` scheduler; benchmark PPPL throughput on real UniRef batches (mixed-length).
- Define `LLM_ARCH_ESM` in a llama.cpp fork or a standalone GGUF; provide `convert_esm_to_gguf.py`.
- **Threshold to ship: a public reproducible benchmark — ESM-2-650M PPPL throughput (seqs/sec) and ProteinGym leaderboard run — beating HF/PyTorch on single-socket CPU.**

### What to drop or punt
- Do NOT implement training, LoRA, or any backward pass — ESME owns that niche on GPU and it doesn't differentiate a C++ engine.
- Do NOT chase 15B in v1; it's bandwidth-bound and won't demo on commodity hardware without W4. Roadmap for v2 with AWQ.
- Do NOT hand-write GEMM from absolute scratch under deadline pressure — link **libxsmm** or **ruy** for small-shape microkernels and focus your value on schedulers, quantization recipe, and varlen attention.

## Caveats

- **ESME is published, recent, and very close in scope to the GPU half of esm.cpp.** Position esm.cpp as **CPU-first + ahead-of-time INT8 + continuous batching**, not as "ESME for CPU."
- **No vetted INT8 W8A8 recipe for ESM-2 exists in the literature.** ESME uses bitsandbytes (W8/W4 weight-only) and finds degraded quality below 150M — your SmoothQuant W8A8 path is genuinely uncharted and may need careful per-channel observers.
- **AMX availability is narrow** (Sapphire Rapids, Emerald Rapids, Granite Rapids; AMD Zen 6 has VNNI-INT8 but not AMX tiles). Most benchmarking machines a reader can reproduce on are still Ice Lake / Zen 3. Lead with AVX-512 VNNI numbers and treat AMX as a bonus.
- **The 15B model has tied embeddings but a 5120-dim hidden state**; `lm_head` output `[seq, 33]` is small — not a performance concern, but the embedding table (5120 × 33 × 4 B = 660 KB FP32) is trivial.
- **Pseudo-perplexity is L forward passes per sequence.** The "One Fell Swoop" approximation (Kantroo et al. 2024, bioRxiv 2024.07.09.602754) does it in one pass but is approximate; for a benchmark paper, use the exact form.
- **No prior C++ ESM port appears to exist in mainline open source** as of May 2026; the user should confirm by searching GitHub for `esm cpp`, `esm2 cpp`, and `protein cpp` immediately before announcing.