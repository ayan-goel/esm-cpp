#pragma once

#include <cstddef>

#include "esm_cpp/quant.h"

namespace esm::kernels {

// Public entry points dispatch on esm::CurrentIsa() to the registered
// implementation. Until Slice 3 lands SIMD paths, every dispatch falls
// through to the scalar reference (the *Ref symbols below). Tests can
// pin the path with ESM_FORCE_ISA=<ref|neon|avx512|...>.
//
// linear: C[m, n] = sum_k A[m, k] * W[n, k] + (bias ? bias[n] : 0)
// A: row-major [M, K]
// W: row-major [N, K]   (PyTorch convention: out_features, in_features)
// bias: [N] or nullptr
// C: row-major [M, N]
void Linear(const float* A, const float* W, const float* bias, float* C,
            int M, int N, int K);

void LayerNorm(const float* x, const float* gamma, const float* beta,
               float eps, float* out, int num_rows, int d);

void Gelu(const float* x, float* out, std::size_t n);

void RopeApplyInplace(float* x, const float* cos, const float* sin,
                      int num_heads, int seq_len, int head_dim);

void Attention(const float* Q, const float* K, const float* V,
               const int* attention_mask, float* out, int num_heads,
               int seq_len, int head_dim);

// Scalar reference implementations. Always linked in. Tests and the
// SIMD cross-check harness use these as the ground truth. Production
// callers should prefer the facade above so dispatch happens.

void LinearRef(const float* A, const float* W, const float* bias, float* C,
               int M, int N, int K);

// W8A16 Linear with per-channel symmetric INT8 weights and FP32
// activations. Slice 6 will add LinearVnni (W8A8) on top of this
// using the same per-channel scale convention.
void LinearInt8Ref(const float* A, const esm::quant::QuantizedTensor& W,
                   const float* bias, float* C, int M, int N, int K);
void LinearInt8(const float* A, const esm::quant::QuantizedTensor& W,
                const float* bias, float* C, int M, int N, int K);

void LayerNormRef(const float* x, const float* gamma, const float* beta,
                  float eps, float* out, int num_rows, int d);

// GELU using the exact erf form ESM-2 uses:
//   gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
// (HF EsmModel calls this `gelu()` and explicitly notes F.gelu yields
// "subtly wrong results"; the erf form is the canonical one for ESM.)
void GeluRef(const float* x, float* out, std::size_t n);

// Build cos/sin tables for RoPE.
//   inv_freq[i] = 1 / 10000^(2i / head_dim)  for i in [0, head_dim/2)
//   freqs[t, i] = t * inv_freq[i]                     (t in [0, seq_len))
//   cos/sin[t, i] = cos/sin(freqs[t, i % (head_dim/2)])  (half-duplicated)
// cos and sin are [seq_len, head_dim] row-major. Pure scalar; no dispatch.
void RopeBuildTables(int seq_len, int head_dim, float* cos, float* sin);

// Apply RoPE (half-then-half rotate) in place on x.
//   x: [num_heads, seq_len, head_dim]
//   cos/sin: [seq_len, head_dim]
void RopeApplyInplaceRef(float* x, const float* cos, const float* sin,
                         int num_heads, int seq_len, int head_dim);

// Packed-varlen RoPE in [T, H, head_dim] layout. Each sequence b in
// [0, batch_size) uses positions 0..seq_len_b - 1 against the shared
// cos/sin tables — positions restart at every cu_seqlens boundary.
//   x: [T, num_heads, head_dim]  (token-major; head inner)
//   cos/sin: [max_seqlen, head_dim] (max_seqlen >= max sequence length)
//   cu_seqlens: [batch_size + 1]
void RopeApplyVarlenRef(float* x, const float* cos, const float* sin,
                        const int* cu_seqlens, int batch_size, int num_heads,
                        int head_dim);

// Scaled-dot self-attention (no Q-scale here; caller scales Q ahead of
// RoPE per the ESM convention). Mask: -inf for padded positions, 0 for
// real positions. Softmax accumulator is FP32.
//   Q, K, V: [num_heads, seq_len, head_dim]
//   mask:    [seq_len] or nullptr (0 keep, 1 pad)
//   out:     [seq_len, num_heads * head_dim] (heads contiguous along last dim)
void AttentionRef(const float* Q, const float* K, const float* V,
                  const int* attention_mask, float* out, int num_heads,
                  int seq_len, int head_dim);

// Packed-varlen scaled-dot attention. Q must already be scaled (ESM-2
// scales Q by 1/sqrt(head_dim) before RoPE). Sequences are packed back-
// to-back along the T axis; cu_seqlens[b+1] - cu_seqlens[b] is the b-th
// sequence's length, with cu_seqlens[0] = 0 and cu_seqlens[B] = T total.
// Attention is per-sequence — KV from sequence j never attends to queries
// from sequence i.
//   q, k, v: [T, num_heads, head_dim] (token-major; head inner)
//   cu_seqlens: [batch_size + 1]
//   out: [T, num_heads * head_dim] (heads concatenated along last dim,
//        matches AttentionRef's output for cross-validation)
// Softmax accumulator is FP32. This is the interface the Phase 3
// cu_seqlens scheduler will consume.
void AttentionVarlenRef(const float* q, const float* k, const float* v,
                        const int* cu_seqlens, int batch_size, int num_heads,
                        int head_dim, float* out);
void AttentionVarlen(const float* q, const float* k, const float* v,
                     const int* cu_seqlens, int batch_size, int num_heads,
                     int head_dim, float* out);

}  // namespace esm::kernels
