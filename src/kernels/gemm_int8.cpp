#include "esm_cpp/kernels.h"
#include "esm_cpp/quant.h"

namespace esm::kernels {

#ifdef ESM_KERNEL_REFERENCE

// Scalar reference for W8A16: per-channel symmetric INT8 weights,
// activations stay FP32 here. Slice 4 will quantize activations too
// (W8A8) by adding a per-tensor activation scale to the inner loop.
// Slice 6 replaces this body's inner loop with VPDPBUSD on x86.
//
// C[m, n] = sum_k A[m, k] * (packed[n, k] * scale[n]) + bias[n]
// Factor scale[n] out of the k-loop: do the integer accumulation in
// FP32 (the int8 promotes to float for the multiply) then multiply
// by scale[n] at the end. This is bit-equivalent to dequant-then-
// matmul but avoids the per-element scale multiply.
void LinearInt8Ref(const float* A, const esm::quant::QuantizedTensor& W,
                   const float* bias, float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    const float* a_row = A + static_cast<long>(m) * K;
    for (int n = 0; n < N; ++n) {
      const float scale = W.per_channel_scales[n];
      const std::int8_t* w_row =
          W.packed.data() + static_cast<long>(n) * K;
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += a_row[k] * static_cast<float>(w_row[k]);
      }
      C[static_cast<long>(m) * N + n] = acc * scale + (bias ? bias[n] : 0.0f);
    }
  }
}

#endif  // ESM_KERNEL_REFERENCE

}  // namespace esm::kernels
