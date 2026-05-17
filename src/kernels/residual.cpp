#include "esm_cpp/kernels.h"

#include <cstddef>

#ifdef ESM_KERNEL_AVX512
#include <immintrin.h>

#include "esm_cpp/thread_pool.h"
#endif

namespace esm::kernels {

#ifdef ESM_KERNEL_REFERENCE

void ResidualAddInplaceRef(float* y, const float* x, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) y[i] += x[i];
}

void ScaleInplaceRef(float* x, std::size_t n, float scale) {
  for (std::size_t i = 0; i < n; ++i) x[i] *= scale;
}

#endif  // ESM_KERNEL_REFERENCE

#ifdef ESM_KERNEL_AVX512

// Elementwise FP32 ops on dense buffers. The per-worker chunk size is
// 4096 elements (= 256 zmm-wide loads) — same threshold as GELU. The
// inner loop unrolls 4× so 4 add/mul ops issue per cycle on SPR's load
// + FMA ports, hiding the load latency.
namespace {

constexpr std::size_t kElemsPerChunk = 4096;

void ResidualAddChunk(float* y, const float* x, std::size_t start,
                       std::size_t end) {
  std::size_t i = start;
  for (; i + 64 <= end; i += 64) {
    __m512 y0 = _mm512_loadu_ps(y + i);
    __m512 y1 = _mm512_loadu_ps(y + i + 16);
    __m512 y2 = _mm512_loadu_ps(y + i + 32);
    __m512 y3 = _mm512_loadu_ps(y + i + 48);
    __m512 x0 = _mm512_loadu_ps(x + i);
    __m512 x1 = _mm512_loadu_ps(x + i + 16);
    __m512 x2 = _mm512_loadu_ps(x + i + 32);
    __m512 x3 = _mm512_loadu_ps(x + i + 48);
    _mm512_storeu_ps(y + i, _mm512_add_ps(y0, x0));
    _mm512_storeu_ps(y + i + 16, _mm512_add_ps(y1, x1));
    _mm512_storeu_ps(y + i + 32, _mm512_add_ps(y2, x2));
    _mm512_storeu_ps(y + i + 48, _mm512_add_ps(y3, x3));
  }
  for (; i + 16 <= end; i += 16) {
    _mm512_storeu_ps(y + i, _mm512_add_ps(_mm512_loadu_ps(y + i),
                                            _mm512_loadu_ps(x + i)));
  }
  if (i < end) {
    const std::size_t tail = end - i;
    const __mmask16 mask = static_cast<__mmask16>((1u << tail) - 1u);
    __m512 y_v = _mm512_maskz_loadu_ps(mask, y + i);
    __m512 x_v = _mm512_maskz_loadu_ps(mask, x + i);
    _mm512_mask_storeu_ps(y + i, mask, _mm512_add_ps(y_v, x_v));
  }
}

void ScaleChunk(float* x, std::size_t start, std::size_t end, __m512 s) {
  std::size_t i = start;
  for (; i + 64 <= end; i += 64) {
    _mm512_storeu_ps(x + i,      _mm512_mul_ps(s, _mm512_loadu_ps(x + i)));
    _mm512_storeu_ps(x + i + 16, _mm512_mul_ps(s, _mm512_loadu_ps(x + i + 16)));
    _mm512_storeu_ps(x + i + 32, _mm512_mul_ps(s, _mm512_loadu_ps(x + i + 32)));
    _mm512_storeu_ps(x + i + 48, _mm512_mul_ps(s, _mm512_loadu_ps(x + i + 48)));
  }
  for (; i + 16 <= end; i += 16) {
    _mm512_storeu_ps(x + i, _mm512_mul_ps(s, _mm512_loadu_ps(x + i)));
  }
  if (i < end) {
    const std::size_t tail = end - i;
    const __mmask16 mask = static_cast<__mmask16>((1u << tail) - 1u);
    __m512 v = _mm512_maskz_loadu_ps(mask, x + i);
    _mm512_mask_storeu_ps(x + i, mask, _mm512_mul_ps(s, v));
  }
}

}  // namespace

void ResidualAddInplaceAvx512(float* y, const float* x, std::size_t n) {
  const int total_chunks = static_cast<int>(
      (n + kElemsPerChunk - 1) / kElemsPerChunk);
  if (total_chunks > 1 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(
        0, total_chunks, /*grain=*/1, [&](int begin, int end) {
          for (int c = begin; c < end; ++c) {
            const std::size_t s =
                static_cast<std::size_t>(c) * kElemsPerChunk;
            const std::size_t e =
                std::min(s + kElemsPerChunk, n);
            ResidualAddChunk(y, x, s, e);
          }
        });
  } else {
    ResidualAddChunk(y, x, 0, n);
  }
}

void ScaleInplaceAvx512(float* x, std::size_t n, float scale) {
  const __m512 s = _mm512_set1_ps(scale);
  const int total_chunks = static_cast<int>(
      (n + kElemsPerChunk - 1) / kElemsPerChunk);
  if (total_chunks > 1 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(
        0, total_chunks, /*grain=*/1, [&](int begin, int end) {
          for (int c = begin; c < end; ++c) {
            const std::size_t s_off =
                static_cast<std::size_t>(c) * kElemsPerChunk;
            const std::size_t e_off =
                std::min(s_off + kElemsPerChunk, n);
            ScaleChunk(x, s_off, e_off, s);
          }
        });
  } else {
    ScaleChunk(x, 0, n, s);
  }
}

#endif  // ESM_KERNEL_AVX512

}  // namespace esm::kernels
