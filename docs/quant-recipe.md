# Quantization recipe

esm.cpp ships **W8A8 (per-channel symmetric INT8 weights, per-tensor symmetric INT8 activations) with SmoothQuant offline migration** for ESM-2 at 150M and above. Smaller models (8M, 35M) ship FP32 only — per the [ESME](https://github.com/qingyuan-hou/esme) empirical finding that INT8 collapses below 150M, we don't pretend otherwise.

## Recipe at a glance

| | Value | Source |
|---|---|---|
| Weight quant | per-channel symmetric INT8, `[-127, 127]` (no `-128`) | [src/quant/pack.cpp](../src/quant/pack.cpp) |
| Activation quant | per-tensor symmetric INT8, `u8 in [1, 255]` (zero-point 128) | [src/kernels/gemm_int8.cpp](../src/kernels/gemm_int8.cpp) |
| Activation observer | 99.9th-pctile reservoir, 65536 samples per site | [src/quant/observer.cpp](../src/quant/observer.cpp) |
| SmoothQuant α | 0.5 default; sweep `{0.3, 0.4, 0.5, 0.6, 0.7}` per model | [python/esm_cpp/alpha_sweep.py](../python/esm_cpp/alpha_sweep.py) |
| Escapes | `lm_head.dense`, `lm_head.layer_norm`, tied decoder, embed: FP32 | [src/model.cpp](../src/model.cpp) `QuantizeWeights` |
| Optional escape | Layer-0 `fc1` input rounded to FP16 if drift > 0.2 | `Model.set_first_block_fc1_fp16(True)` |
| K/V quant | Off by default; FP32 K/V even when weights are INT8 | [include/esm_cpp/kernels.h](../include/esm_cpp/kernels.h) |

## Why this recipe

**Per-channel weight + per-tensor activation** is the standard `LLM.int8()` / SmoothQuant configuration. Per-channel weight tolerates the large dynamic range of ESM's QKV projections (some channels have ~50× others); per-tensor activation keeps the GEMM kernels simple and lets us use VNNI's `VPDPBUSD` (u8 × s8 → s32) directly.

**SmoothQuant α=0.5 default** is the OPT/BLOOM default that research-report §5 + §9 say generalizes to ESM-2. The migration is identity-preserving for the FP32 forward at any α — it just trades activation dynamic range for weight dynamic range. After SmoothQuant, per-tensor activation quant tolerates the now-smoother distribution.

**Why we exclude `inter_gelu → fc2`** from SmoothQuant: the GELU non-linearity breaks the diagonal-rescale identity. Migrating across GELU would not preserve the FP32 forward. The plan accepted this trade-off — the layer-0 outliers that hurt INT8 most are pre-GELU.

**Why `lm_head` stays FP32**: the LM head reads from the post-encoder hidden state (which still has the residual stream's full magnitude), runs a small Linear (`d → d`), GELU, LN, and finally projects to vocab. Quantizing the `d → V` projection is unsafe because logit magnitudes set softmax temperature; small quant noise compounds catastrophically into argmax-level errors. The per-vocab decoder is also tied to the embedding (we want word_embeddings stored bit-exact, not approximated).

## The α sweep

For a new model size, sweep α and pick the lowest-drift recipe:

```bash
python -m esm_cpp.alpha_sweep \
    --model esm2_t30_150M \
    --calib data/uniref50_calib_v1.fasta \
    --pppl-holdout data/uniref50_pppl_holdout.fasta \
    --alphas 0.3,0.4,0.5,0.6,0.7
```

Report: per-α PPPL drift. If best drift > 0.2, enable `--first-block-fp16` and re-sweep. If still > 0.1, surface the failure — INT8 may not be viable at that scale.

Measured on 8M (3 calib + 2 PPPL seqs):

| α | PPPL drift vs FP32 |
|---|---:|
| 0.3 | 0.0354 |
| 0.5 | **0.0045** (best) |
| 0.7 | 0.0132 |

(8M doesn't ship INT8 per SPEC; these numbers are the recipe smoke test only.)

## End-to-end commands

Calibrate (run forward_with_observer on UniRef50 sequences):

```bash
python -m esm_cpp.quantize --calibrate \
    --model esm2_t30_150M \
    --calib data/uniref50_calib_v1.fasta \
    --out weights/esm2_t30_150M_calib.json
```

Apply SmoothQuant + quantize + emit a GGUF artifact:

```bash
python -m esm_cpp.quantize --apply-smoothquant \
    --model esm2_t30_150M \
    --stats weights/esm2_t30_150M_calib.json \
    --alpha 0.5 \
    --output weights/esm2_t30_150M_q8.gguf
```

Load the quantized artifact and run forward:

```python
import esm_cpp
m = esm_cpp.Model.load_from_gguf("weights/esm2_t30_150M_q8.gguf")
assert m.config.weights_quantized
ids = esm_cpp.Tokenizer().encode("MKTGVA...")
logits = m.forward(ids)
```

## SIMD paths (x86 hand-off)

The recipe is implemented; the production VNNI / AMX microkernels are pending hand-off to the gate machine. Current state in [src/kernels/gemm_int8.cpp](../src/kernels/gemm_int8.cpp):

- `LinearInt8Ref`: scalar W8A16 reference. Always present. Used by all hosts that don't have a faster path. INT8 weights, FP32 activations (the activation quantizer is unused on this path).
- `LinearVnni` (stub): forwards to `LinearInt8Ref`. The full intrinsics body (16×16 register block of s32 accumulators, `VPDPBUSD` inner loop, Goto packing, scale folding at C-write-out) is documented in source comments as the gate-machine deliverable.
- `LinearAmx` (planned): `TDPBSSD` with 16-row tiles, K=64 INT8 chunks. Needs `syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` at first use, guarded behind `std::call_once`.

For ARM, `LinearInt8` falls through to the scalar reference (no NEON INT8 microkernel; Accelerate's BLAS is FP32 only). ARM perf for INT8 is dev-only.

## Quality gate

Per [SPEC.md](../SPEC.md) Phase 2:

- `|PPPL_int8 − PPPL_fp32| < 0.1` on a 1000-sequence UniRef50 holdout (length 100–500), both 150M and 650M.
- `|Spearman_int8 − Spearman_fp32| < 0.01` averaged across the 217 ProteinGym v1.3 substitution assays, zero-shot masked-marginal scoring, both 150M and 650M.

These measurements are the Phase 3 Slice 7.7 gate-machine deliverable: the recipe is in place, the eval harness is in place, the run is a hand-off to the AVX-512+VNNI host.
