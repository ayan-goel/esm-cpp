#include "esm_cpp/kernels.h"

#include <cmath>

namespace esm::kernels {

// ESM uses the exact erf form: x * 0.5 * (1 + erf(x / sqrt(2))).
// HF EsmModel ships its own gelu() rather than F.gelu() because F.gelu
// yields "subtly wrong results" relative to the original ESM repo.
void GeluRef(const float* x, float* out, std::size_t n) {
  static constexpr float kInvSqrt2 = 0.70710678118654752440f;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = x[i] * 0.5f * (1.0f + std::erf(x[i] * kInvSqrt2));
  }
}

}  // namespace esm::kernels
