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

#endif  // ESM_KERNEL_REFERENCE

}  // namespace esm::kernels
