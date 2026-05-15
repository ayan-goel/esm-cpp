#include "esm_cpp/kernels.h"

#include <cmath>

namespace esm::kernels {

// Table builder is pure scalar and always linked (not behind ESM_KERNEL_REFERENCE).
// SIMD impls reuse the same cos/sin tables.
//
// inv_freq[i] = 1 / 10000^(2i/head_dim) for i in [0, head_dim/2)
// Tables are duplicated half/half so cos[t, j] = cos[t, j + half] for
// j < half. This matches HF's torch.cat((freqs, freqs), dim=-1).
void RopeBuildTables(int seq_len, int head_dim, float* cos, float* sin) {
  const int half = head_dim / 2;
  for (int i = 0; i < half; ++i) {
    float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
    float inv_freq = std::pow(10000.0f, -exponent);
    for (int t = 0; t < seq_len; ++t) {
      float angle = static_cast<float>(t) * inv_freq;
      float c = std::cos(angle);
      float s = std::sin(angle);
      cos[t * head_dim + i] = c;
      cos[t * head_dim + i + half] = c;
      sin[t * head_dim + i] = s;
      sin[t * head_dim + i + half] = s;
    }
  }
}

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
