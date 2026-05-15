#include "esm_cpp/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace esm::kernels {

#ifdef ESM_KERNEL_REFERENCE

// Scaled-dot self-attention; Q must already be scaled by 1/sqrt(head_dim)
// per the ESM convention (scale before RoPE, not the score after).
//   Q, K, V: [num_heads, seq_len, head_dim]
//   attention_mask: [seq_len] with 1 for real tokens and 0 for pad,
//                   or nullptr to treat everything as real.
//   out: [seq_len, num_heads * head_dim] with heads concatenated along
//        the last dimension (matches HF .transpose(1,2).reshape(L, -1)).
// Softmax accumulator is FP32 — kept in double for numerical safety on
// long sequences, since this is the reference path.
void AttentionRef(const float* Q, const float* K, const float* V,
                  const int* attention_mask, float* out, int num_heads,
                  int seq_len, int head_dim) {
  std::vector<float> scores(static_cast<std::size_t>(seq_len));
  for (int h = 0; h < num_heads; ++h) {
    const float* Qh = Q + static_cast<long>(h) * seq_len * head_dim;
    const float* Kh = K + static_cast<long>(h) * seq_len * head_dim;
    const float* Vh = V + static_cast<long>(h) * seq_len * head_dim;
    for (int i = 0; i < seq_len; ++i) {
      const float* qi = Qh + static_cast<long>(i) * head_dim;
      float max_score = -std::numeric_limits<float>::infinity();
      for (int j = 0; j < seq_len; ++j) {
        const float* kj = Kh + static_cast<long>(j) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
        if (attention_mask && attention_mask[j] == 0) {
          dot = -std::numeric_limits<float>::infinity();
        }
        scores[static_cast<std::size_t>(j)] = dot;
        if (dot > max_score) max_score = dot;
      }
      double sum = 0.0;
      for (int j = 0; j < seq_len; ++j) {
        float e = (scores[static_cast<std::size_t>(j)] == -std::numeric_limits<float>::infinity())
                      ? 0.0f
                      : std::exp(scores[static_cast<std::size_t>(j)] - max_score);
        scores[static_cast<std::size_t>(j)] = e;
        sum += e;
      }
      float inv_sum = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
      float* out_row = out + (static_cast<long>(i) * num_heads + h) * head_dim;
      for (int d = 0; d < head_dim; ++d) out_row[d] = 0.0f;
      for (int j = 0; j < seq_len; ++j) {
        float w = scores[static_cast<std::size_t>(j)] * inv_sum;
        if (w == 0.0f) continue;
        const float* vj = Vh + static_cast<long>(j) * head_dim;
        for (int d = 0; d < head_dim; ++d) out_row[d] += w * vj[d];
      }
    }
  }
}

// Packed-varlen scaled-dot attention with FP32 softmax accumulator.
// Layout differs from AttentionRef: Q/K/V are token-major [T, H, dh]
// rather than head-major [H, L, dh]; output is still [T, H*dh] so it
// drops in to model.cpp without further rearrangement. cu_seqlens
// isolates sequences — each query at token t in sequence b only attends
// to keys/values from positions [cu_seqlens[b], cu_seqlens[b+1]).
//
// This is the scalar reference; tile size is 1. Slice 4.3 will add the
// FlashAttention-style tiled AVX-512 path against this oracle.
void AttentionVarlenRef(const float* q, const float* k, const float* v,
                        const int* cu_seqlens, int batch_size, int num_heads,
                        int head_dim, float* out) {
  for (int b = 0; b < batch_size; ++b) {
    const int seq_start = cu_seqlens[b];
    const int seq_end = cu_seqlens[b + 1];
    const int seq_len = seq_end - seq_start;
    if (seq_len <= 0) continue;
    std::vector<float> scores(static_cast<std::size_t>(seq_len));
    for (int h = 0; h < num_heads; ++h) {
      for (int i = 0; i < seq_len; ++i) {
        const int t_q = seq_start + i;
        const float* qi =
            q + (static_cast<long>(t_q) * num_heads + h) * head_dim;
        float max_score = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < seq_len; ++j) {
          const int t_k = seq_start + j;
          const float* kj =
              k + (static_cast<long>(t_k) * num_heads + h) * head_dim;
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
          scores[static_cast<std::size_t>(j)] = dot;
          if (dot > max_score) max_score = dot;
        }
        double sum = 0.0;
        for (int j = 0; j < seq_len; ++j) {
          float e = std::exp(scores[static_cast<std::size_t>(j)] - max_score);
          scores[static_cast<std::size_t>(j)] = e;
          sum += e;
        }
        float inv_sum = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
        float* out_row =
            out + (static_cast<long>(t_q) * num_heads + h) * head_dim;
        for (int d = 0; d < head_dim; ++d) out_row[d] = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
          float w = scores[static_cast<std::size_t>(j)] * inv_sum;
          if (w == 0.0f) continue;
          const int t_k = seq_start + j;
          const float* vj =
              v + (static_cast<long>(t_k) * num_heads + h) * head_dim;
          for (int d = 0; d < head_dim; ++d) out_row[d] += w * vj[d];
        }
      }
    }
  }
}

#endif  // ESM_KERNEL_REFERENCE

}  // namespace esm::kernels
