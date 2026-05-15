#include "esm_cpp/cpu_features.h"
#include "esm_cpp/kernels.h"

// Dispatch facade. The *Ref impls are always present (Ref TU); SIMD impls
// register themselves below via per-arch forward declarations. Any ISA we
// don't have a specialized impl for falls through to *Ref so the dispatch
// never silently runs nothing.

namespace esm::kernels {

// Forward declarations for per-ISA impls. CMake links the matching OBJECT
// library based on CMAKE_SYSTEM_PROCESSOR, so the symbols referenced here
// are only resolved when the matching arch path is in play.
#if defined(__aarch64__) || defined(_M_ARM64)
void LinearNeon(const float* A, const float* W, const float* bias, float* C,
                int M, int N, int K);
#endif

#if defined(__x86_64__) || defined(_M_X64)
void LinearAvx512(const float* A, const float* W, const float* bias, float* C,
                  int M, int N, int K);
#endif

void Linear(const float* A, const float* W, const float* bias, float* C,
            int M, int N, int K) {
  switch (esm::CurrentIsa()) {
#if defined(__aarch64__) || defined(_M_ARM64)
    case Isa::Neon:
      return LinearNeon(A, W, bias, C, M, N, K);
#endif
#if defined(__x86_64__) || defined(_M_X64)
    case Isa::Avx512:
    case Isa::Avx512Vnni:
    case Isa::Amx:
      return LinearAvx512(A, W, bias, C, M, N, K);
#endif
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
