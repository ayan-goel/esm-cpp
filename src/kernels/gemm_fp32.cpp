#include "esm_cpp/kernels.h"

namespace esm::kernels {

#ifdef ESM_KERNEL_REFERENCE

void LinearRef(const float* A, const float* W, const float* bias, float* C,
               int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = bias ? bias[n] : 0.0f;
      const float* a_row = A + static_cast<long>(m) * K;
      const float* w_row = W + static_cast<long>(n) * K;
      for (int k = 0; k < K; ++k) {
        acc += a_row[k] * w_row[k];
      }
      C[static_cast<long>(m) * N + n] = acc;
    }
  }
}

#endif  // ESM_KERNEL_REFERENCE

}  // namespace esm::kernels
