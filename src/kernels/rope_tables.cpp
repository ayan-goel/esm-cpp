#include "esm_cpp/kernels.h"

#include <cmath>

namespace esm::kernels {

// Pure scalar helper, ISA-independent. Lives in esm_cpp_core directly
// (not in any per-ISA OBJECT lib) so it's defined exactly once even when
// multiple per-ISA kernel TUs are linked in.
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

}  // namespace esm::kernels
