#include "esm_cpp/quant.h"

#include <algorithm>
#include <cmath>

namespace esm::quant {

void Quantize(const float* W_fp32, int N, int K, QuantizedTensor* out) {
  out->N = N;
  out->K = K;
  out->packed.assign(static_cast<std::size_t>(N) * K, 0);
  out->per_channel_scales.assign(N, 0.0f);
  for (int n = 0; n < N; ++n) {
    float row_max = 0.0f;
    const float* row = W_fp32 + static_cast<long>(n) * K;
    for (int k = 0; k < K; ++k) {
      row_max = std::max(row_max, std::fabs(row[k]));
    }
    if (row_max == 0.0f) {
      // Already zero-initialized; nothing to round.
      out->per_channel_scales[n] = 0.0f;
      continue;
    }
    const float scale = row_max / 127.0f;
    out->per_channel_scales[n] = scale;
    const float inv_scale = 1.0f / scale;
    for (int k = 0; k < K; ++k) {
      float q = std::nearbyint(row[k] * inv_scale);
      if (q > 127.0f) q = 127.0f;
      if (q < -127.0f) q = -127.0f;
      out->packed[static_cast<std::size_t>(n) * K + k] =
          static_cast<std::int8_t>(q);
    }
  }
}

}  // namespace esm::quant
