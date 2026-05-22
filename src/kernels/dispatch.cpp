#include "esm_cpp/cpu_features.h"
#include "esm_cpp/kernels.h"
#include "esm_cpp/thread_pool.h"

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
    // All ARM tiers share the FP32 NEON kernel; the DotProd/i8mm tiers only
    // change the INT8 path (LinearInt8), not FP32 GEMM.
    case Isa::Neon:
    case Isa::NeonDotProd:
    case Isa::NeonI8mm:
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

#if defined(__x86_64__) || defined(_M_X64)
void LinearVnni(const float* A, const esm::quant::QuantizedTensor& W,
                const float* bias, float* C, int M, int N, int K);
void LinearAmx(const float* A, const esm::quant::QuantizedTensor& W,
                const float* bias, float* C, int M, int N, int K);
void GeluAvx512(const float* x, float* out, std::size_t n);
void LayerNormAvx512(const float* x, const float* gamma, const float* beta,
                     float eps, float* out, int num_rows, int d);
void AttentionVarlenAvx512(const float* q, const float* k, const float* v,
                            const int* cu_seqlens, int batch_size,
                            int num_heads, int head_dim, float* out);
void ResidualAddInplaceAvx512(float* y, const float* x, std::size_t n);
void ScaleInplaceAvx512(float* x, std::size_t n, float scale);
void RopeApplyVarlenAvx512(float* x, const float* cos, const float* sin,
                            const int* cu_seqlens, int batch_size,
                            int num_heads, int head_dim);
#endif

void LinearInt8(const float* A, const esm::quant::QuantizedTensor& W,
                const float* bias, float* C, int M, int N, int K) {
  switch (esm::CurrentIsa()) {
#if defined(__x86_64__) || defined(_M_X64)
    case Isa::Amx:
      // LinearAmx handles shape-gating + XSAVE-permission detection
      // internally; it falls back to LinearVnni when AMX can't be used.
      return LinearAmx(A, W, bias, C, M, N, K);
    case Isa::Avx512Vnni:
      return LinearVnni(A, W, bias, C, M, N, K);
#endif
    default:
      return LinearInt8Ref(A, W, bias, C, M, N, K);
  }
}

void LayerNorm(const float* x, const float* gamma, const float* beta,
               float eps, float* out, int num_rows, int d) {
  switch (esm::CurrentIsa()) {
#if defined(__x86_64__) || defined(_M_X64)
    case Isa::Avx512:
    case Isa::Avx512Vnni:
    case Isa::Amx:
      return LayerNormAvx512(x, gamma, beta, eps, out, num_rows, d);
#endif
    default:
      return LayerNormRef(x, gamma, beta, eps, out, num_rows, d);
  }
}

void Gelu(const float* x, float* out, std::size_t n) {
  switch (esm::CurrentIsa()) {
#if defined(__x86_64__) || defined(_M_X64)
    case Isa::Avx512:
    case Isa::Avx512Vnni:
    case Isa::Amx:
      return GeluAvx512(x, out, n);
#endif
    default:
      return GeluRef(x, out, n);
  }
}

void ResidualAddInplace(float* y, const float* x, std::size_t n) {
  switch (esm::CurrentIsa()) {
#if defined(__x86_64__) || defined(_M_X64)
    case Isa::Avx512:
    case Isa::Avx512Vnni:
    case Isa::Amx:
      return ResidualAddInplaceAvx512(y, x, n);
#endif
    default:
      return ResidualAddInplaceRef(y, x, n);
  }
}

void ScaleInplace(float* x, std::size_t n, float scale) {
  switch (esm::CurrentIsa()) {
#if defined(__x86_64__) || defined(_M_X64)
    case Isa::Avx512:
    case Isa::Avx512Vnni:
    case Isa::Amx:
      return ScaleInplaceAvx512(x, n, scale);
#endif
    default:
      return ScaleInplaceRef(x, n, scale);
  }
}

void RopeApplyInplace(float* x, const float* cos, const float* sin,
                      int num_heads, int seq_len, int head_dim) {
  switch (esm::CurrentIsa()) {
    default:
      return RopeApplyInplaceRef(x, cos, sin, num_heads, seq_len, head_dim);
  }
}

void RopeApplyVarlen(float* x, const float* cos, const float* sin,
                     const int* cu_seqlens, int batch_size, int num_heads,
                     int head_dim) {
  switch (esm::CurrentIsa()) {
#if defined(__x86_64__) || defined(_M_X64)
    case Isa::Avx512:
    case Isa::Avx512Vnni:
    case Isa::Amx:
      return RopeApplyVarlenAvx512(x, cos, sin, cu_seqlens, batch_size,
                                    num_heads, head_dim);
#endif
    default:
      return RopeApplyVarlenRef(x, cos, sin, cu_seqlens, batch_size,
                                num_heads, head_dim);
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

namespace {
// ISA-dispatched single-call body. Used both by the parallel_for shim
// below (one chunk per worker) and by the direct serial path.
inline void AttentionVarlenBody(const float* q, const float* k,
                                 const float* v, const int* cu_seqlens,
                                 int batch_size, int num_heads, int head_dim,
                                 float* out) {
  switch (esm::CurrentIsa()) {
#if defined(__x86_64__) || defined(_M_X64)
    case Isa::Avx512:
    case Isa::Avx512Vnni:
    case Isa::Amx:
      return AttentionVarlenAvx512(q, k, v, cu_seqlens, batch_size,
                                    num_heads, head_dim, out);
#endif
    default:
      return AttentionVarlenRef(q, k, v, cu_seqlens, batch_size, num_heads,
                                head_dim, out);
  }
}
}  // namespace

void AttentionVarlen(const float* q, const float* k, const float* v,
                     const int* cu_seqlens, int batch_size, int num_heads,
                     int head_dim, float* out) {
  // AVX-512 / VNNI / AMX kernels self-parallelize across (B × H) inside
  // AttentionVarlenAvx512 — exposes 160 chunks at B=8 H=20 instead of
  // the 8 we got from batch-only fan-out. Ref path stays serial; only
  // tests exercise it.
  AttentionVarlenBody(q, k, v, cu_seqlens, batch_size, num_heads, head_dim,
                       out);
}

}  // namespace esm::kernels
