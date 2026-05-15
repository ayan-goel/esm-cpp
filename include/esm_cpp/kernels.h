#pragma once

#include <cstddef>

namespace esm::kernels {

// All kernels here are FP32 scalar reference implementations (Phase 0).
// SIMD vectorized paths arrive in Phase 1 and will share the same signature.

// linear: C[m, n] = sum_k A[m, k] * W[n, k] + (bias ? bias[n] : 0)
// A: row-major [M, K]
// W: row-major [N, K]   (PyTorch convention: out_features, in_features)
// bias: [N] or nullptr
// C: row-major [M, N]
void LinearRef(const float* A, const float* W, const float* bias, float* C,
               int M, int N, int K);

// LayerNorm over the last dimension.
//   x:    [num_rows, d]
//   gamma, beta: [d]
//   out:  [num_rows, d]
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
// cos and sin are [seq_len, head_dim] row-major.
void RopeBuildTables(int seq_len, int head_dim, float* cos, float* sin);

// Apply RoPE (half-then-half rotate) in place on x.
//   x: [num_heads, seq_len, head_dim]
//   cos/sin: [seq_len, head_dim]
void RopeApplyInplace(float* x, const float* cos, const float* sin,
                      int num_heads, int seq_len, int head_dim);

// Scaled-dot self-attention (no Q-scale here; caller scales Q ahead of
// RoPE per the ESM convention). Mask: -inf for padded positions, 0 for
// real positions. Softmax accumulator is FP32.
//   Q, K, V: [num_heads, seq_len, head_dim]
//   mask:    [seq_len] or nullptr (0 keep, 1 pad)
//   out:     [seq_len, num_heads * head_dim] (heads contiguous along last dim)
void AttentionRef(const float* Q, const float* K, const float* V,
                  const int* attention_mask, float* out, int num_heads,
                  int seq_len, int head_dim);

}  // namespace esm::kernels
