#include "esm_cpp/kernels.h"

#ifdef ESM_KERNEL_NEON
#include <Accelerate/Accelerate.h>
#endif

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

#ifdef ESM_KERNEL_NEON

// Dev fallback: wrap Apple Accelerate's cblas_sgemm. NEON is documented
// as a dev-iteration backend only (CLAUDE.md); the canonical Phase 1
// SIMD path is AVX-512+VNNI on x86. We're not staffing a hand-tuned
// NEON microkernel here.
//
// Our Linear contract: C[m, n] = sum_k A[m, k] * W[n, k] + bias[n]
//   A: row-major [M, K], W: row-major [N, K] (PyTorch out_features x in_features).
// In cblas terms that's C = A * W^T (NoTrans for A, Trans for W).
void LinearNeon(const float* A, const float* W, const float* bias, float* C,
                int M, int N, int K) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, W,
              K, 0.0f, C, N);
  if (bias) {
    for (int m = 0; m < M; ++m) {
      float* C_row = C + static_cast<long>(m) * N;
      for (int n = 0; n < N; ++n) C_row[n] += bias[n];
    }
  }
}

#endif  // ESM_KERNEL_NEON

#ifdef ESM_KERNEL_AVX512

// STUB. Slice 3.1-3.3 will replace the body with a Goto-packed 16x32
// AVX-512 microkernel + macrokernel using _mm512_* intrinsics. Until
// then dispatching to AVX-512 just runs the scalar reference — the
// stub exists so dispatch.cpp's switch arm has a defined symbol to
// call when CMAKE_SYSTEM_PROCESSOR=x86_64.
void LinearRef(const float* A, const float* W, const float* bias, float* C,
               int M, int N, int K);
void LinearAvx512(const float* A, const float* W, const float* bias, float* C,
                  int M, int N, int K) {
  LinearRef(A, W, bias, C, M, N, K);
}

#endif  // ESM_KERNEL_AVX512

}  // namespace esm::kernels
