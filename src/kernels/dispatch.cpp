#include "esm_cpp/cpu_features.h"
#include "esm_cpp/kernels.h"

// Dispatch facade. Until Slice 3 lands SIMD paths, every facade falls
// through to the *Ref implementation. SIMD kernels register here by
// extending the switch arms; the Ref path is the load-bearing fallback
// for any ISA we don't have a specialized impl for.

namespace esm::kernels {

void Linear(const float* A, const float* W, const float* bias, float* C,
            int M, int N, int K) {
  switch (esm::CurrentIsa()) {
    default:
      return LinearRef(A, W, bias, C, M, N, K);
  }
}

void LayerNorm(const float* x, const float* gamma, const float* beta,
               float eps, float* out, int num_rows, int d) {
  switch (esm::CurrentIsa()) {
    default:
      return LayerNormRef(x, gamma, beta, eps, out, num_rows, d);
  }
}

void Gelu(const float* x, float* out, std::size_t n) {
  switch (esm::CurrentIsa()) {
    default:
      return GeluRef(x, out, n);
  }
}

void RopeApplyInplace(float* x, const float* cos, const float* sin,
                      int num_heads, int seq_len, int head_dim) {
  switch (esm::CurrentIsa()) {
    default:
      return RopeApplyInplaceRef(x, cos, sin, num_heads, seq_len, head_dim);
  }
}

void Attention(const float* Q, const float* K, const float* V,
               const int* attention_mask, float* out, int num_heads,
               int seq_len, int head_dim) {
  switch (esm::CurrentIsa()) {
    default:
      return AttentionRef(Q, K, V, attention_mask, out, num_heads, seq_len,
                          head_dim);
  }
}

}  // namespace esm::kernels
