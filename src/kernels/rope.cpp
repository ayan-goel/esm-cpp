#include "esm_cpp/kernels.h"

namespace esm::kernels {

#ifdef ESM_KERNEL_REFERENCE

// rotate_half(x) = cat(-x_second_half, x_first_half) along the last dim.
// apply_rotary_pos_emb(x, cos, sin) = x * cos + rotate_half(x) * sin
// We do this in-place on x for all heads.
void RopeApplyInplaceRef(float* x, const float* cos, const float* sin,
                         int num_heads, int seq_len, int head_dim) {
  const int half = head_dim / 2;
  for (int h = 0; h < num_heads; ++h) {
    for (int t = 0; t < seq_len; ++t) {
      float* xrow = x + (static_cast<long>(h) * seq_len + t) * head_dim;
      const float* crow = cos + static_cast<long>(t) * head_dim;
      const float* srow = sin + static_cast<long>(t) * head_dim;
      for (int i = 0; i < half; ++i) {
        float x1 = xrow[i];
        float x2 = xrow[i + half];
        xrow[i] = x1 * crow[i] + (-x2) * srow[i];
        xrow[i + half] = x2 * crow[i + half] + x1 * srow[i + half];
      }
    }
  }
}

// Same RoPE math as above but over the token-major [T, H, head_dim]
// layout that AttentionVarlen consumes. Positions restart at every
// cu_seqlens boundary so each sequence sees rotations from 0..seq_len-1
// regardless of where in the packed buffer it lives. cos/sin tables
// are shared across the batch.
void RopeApplyVarlenRef(float* x, const float* cos, const float* sin,
                        const int* cu_seqlens, int batch_size, int num_heads,
                        int head_dim) {
  const int half = head_dim / 2;
  for (int b = 0; b < batch_size; ++b) {
    const int seq_start = cu_seqlens[b];
    const int seq_end = cu_seqlens[b + 1];
    for (int p = 0; p < seq_end - seq_start; ++p) {
      const int t_global = seq_start + p;
      const float* crow = cos + static_cast<long>(p) * head_dim;
      const float* srow = sin + static_cast<long>(p) * head_dim;
      for (int h = 0; h < num_heads; ++h) {
        float* xrow =
            x + (static_cast<long>(t_global) * num_heads + h) * head_dim;
        for (int i = 0; i < half; ++i) {
          float x1 = xrow[i];
          float x2 = xrow[i + half];
          xrow[i] = x1 * crow[i] + (-x2) * srow[i];
          xrow[i + half] = x2 * crow[i + half] + x1 * srow[i + half];
        }
      }
    }
  }
}

#endif  // ESM_KERNEL_REFERENCE

}  // namespace esm::kernels
