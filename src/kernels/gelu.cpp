#include "esm_cpp/kernels.h"

#include <cmath>
#include <cstddef>

// AVX-512 path needs intrinsics. Per gemm_int8.cpp's hard-learned lesson,
// keep system headers at file scope — never inside namespace esm::kernels.
#ifdef ESM_KERNEL_AVX512
#include <immintrin.h>

#include "esm_cpp/thread_pool.h"
#endif

#ifdef ESM_KERNEL_NEON
#include <arm_neon.h>

#include <algorithm>

#include "esm_cpp/thread_pool.h"
#include "simd_neon.h"
#endif

namespace esm::kernels {

#ifdef ESM_KERNEL_REFERENCE

// ESM uses the exact erf form: x * 0.5 * (1 + erf(x / sqrt(2))).
// HF EsmModel ships its own gelu() rather than F.gelu() because F.gelu
// yields "subtly wrong results" relative to the original ESM repo.
void GeluRef(const float* x, float* out, std::size_t n) {
  static constexpr float kInvSqrt2 = 0.70710678118654752440f;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = x[i] * 0.5f * (1.0f + std::erf(x[i] * kInvSqrt2));
  }
}

#endif  // ESM_KERNEL_REFERENCE

#ifdef ESM_KERNEL_AVX512

namespace {

// AVX-512 polynomial exp(x). Standard range-reduction: x = n*ln(2) + r
// with r in [-ln(2)/2, ln(2)/2], then exp(x) = 2^n * exp(r). The 2^n
// piece is built via the integer-cast bit-hack on the FP32 exponent; the
// exp(r) piece is a 5-term Horner of the Taylor series, which fits FP32
// to ~3e-7 over the reduced range.
inline __m512 ExpAvx512(__m512 x) {
  const __m512 log2e = _mm512_set1_ps(1.44269504088896340736f);
  const __m512 ln2_hi = _mm512_set1_ps(6.93145752e-1f);
  const __m512 ln2_lo = _mm512_set1_ps(1.42860677e-6f);
  __m512 n_f = _mm512_roundscale_ps(
      _mm512_mul_ps(x, log2e),
      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_sub_ps(_mm512_sub_ps(x, _mm512_mul_ps(n_f, ln2_hi)),
                            _mm512_mul_ps(n_f, ln2_lo));
  __m512 p = _mm512_set1_ps(1.0f / 120.0f);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
  __m512i ni = _mm512_cvtps_epi32(n_f);
  __m512i e_bits = _mm512_slli_epi32(
      _mm512_add_epi32(ni, _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(p, _mm512_castsi512_ps(e_bits));
}

// AVX-512 polynomial erf(x) via Abramowitz–Stegun 7.1.26. Max error
// ~1.5e-7 across all x; tight enough that gelu(x) lands inside FP32
// round-off for the [-5, 5] range we actually feed in. The reference
// gelu uses std::erf which is libm-precision (~1e-9); the cross-check
// test in tests/cpp/test_kernels.cpp asserts agreement to 1e-5.
inline __m512 ErfAvx512(__m512 x) {
  const __m512 sign_mask =
      _mm512_castsi512_ps(_mm512_set1_epi32(static_cast<int>(0x80000000)));
  const __m512 abs_mask =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
  const __m512 one = _mm512_set1_ps(1.0f);
  __m512 sign = _mm512_and_ps(x, sign_mask);
  __m512 ax = _mm512_and_ps(x, abs_mask);
  __m512 t = _mm512_div_ps(
      one, _mm512_fmadd_ps(_mm512_set1_ps(0.3275911f), ax, one));
  __m512 poly = _mm512_set1_ps(1.061405429f);
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(-1.453152027f));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(1.421413741f));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(-0.284496736f));
  poly = _mm512_fmadd_ps(poly, t, _mm512_set1_ps(0.254829592f));
  poly = _mm512_mul_ps(poly, t);
  __m512 neg_x2 = _mm512_sub_ps(_mm512_setzero_ps(), _mm512_mul_ps(x, x));
  __m512 e = ExpAvx512(neg_x2);
  __m512 abs_erf = _mm512_fnmadd_ps(poly, e, one);  // 1 - poly * e
  return _mm512_or_ps(abs_erf, sign);
}

}  // namespace

void GeluAvx512(const float* x, float* out, std::size_t n) {
  // Per-chunk worker: process [start_elem, end_elem) where end_elem may
  // equal n (last chunk owns the masked tail). The chunk granularity is
  // ZmmStride elements so most chunks fit a clean 16-wide vector loop.
  constexpr std::size_t kZmmStride = 16;
  auto worker = [&](std::size_t start_elem, std::size_t end_elem) {
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 inv_sqrt2 = _mm512_set1_ps(0.70710678118654752440f);
    const __m512 one = _mm512_set1_ps(1.0f);
    std::size_t i = start_elem;
    for (; i + 16 <= end_elem; i += 16) {
      __m512 v = _mm512_loadu_ps(x + i);
      __m512 e = ErfAvx512(_mm512_mul_ps(v, inv_sqrt2));
      __m512 y = _mm512_mul_ps(_mm512_mul_ps(v, half),
                                _mm512_add_ps(one, e));
      _mm512_storeu_ps(out + i, y);
    }
    if (i < end_elem) {
      const std::size_t tail = end_elem - i;
      const __mmask16 mask = static_cast<__mmask16>((1u << tail) - 1u);
      __m512 v = _mm512_maskz_loadu_ps(mask, x + i);
      __m512 e = ErfAvx512(_mm512_mul_ps(v, inv_sqrt2));
      __m512 y = _mm512_mul_ps(_mm512_mul_ps(v, half),
                                _mm512_add_ps(one, e));
      _mm512_mask_storeu_ps(out + i, mask, y);
    }
  };
  // Chunk along the element axis. For ESM-2's largest layer (650M ffn =
  // 5120, T = 2048 → 10.5 M elements), 22 cores yield ~480 K elements
  // per worker. We use a grain of 1024 elements to keep the parallel_for
  // overhead amortized for very short calls (e.g. lm_head with d = 320).
  // Chunks are aligned to kZmmStride so only the LAST chunk hits the
  // masked tail path.
  constexpr std::size_t kElemsPerChunk = 4096;  // 256 zmm vectors / chunk
  const int total_chunks = static_cast<int>(
      (n + kElemsPerChunk - 1) / kElemsPerChunk);
  if (total_chunks > 1 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(
        0, total_chunks, /*grain=*/1, [&](int begin, int end) {
          for (int c = begin; c < end; ++c) {
            const std::size_t s =
                static_cast<std::size_t>(c) * kElemsPerChunk;
            const std::size_t e = std::min(s + kElemsPerChunk, n);
            worker(s, e);
          }
        });
  } else {
    worker(0, n);
  }
  (void)kZmmStride;
}

#endif  // ESM_KERNEL_AVX512

#ifdef ESM_KERNEL_NEON

namespace {

using esm::kernels::simd::ExpNeon;

// NEON polynomial erf(x) via Abramowitz-Stegun 7.1.26 (max error ~1.5e-7);
// matches the AVX-512 path so gelu lands inside the 1e-5 cross-check.
inline float32x4_t ErfNeon(float32x4_t x) {
  const uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);
  const float32x4_t one = vdupq_n_f32(1.0f);
  float32x4_t sign = vreinterpretq_f32_u32(
      vandq_u32(vreinterpretq_u32_f32(x), sign_bit));
  float32x4_t ax = vabsq_f32(x);
  float32x4_t t = vdivq_f32(one, vfmaq_f32(one, vdupq_n_f32(0.3275911f), ax));
  float32x4_t poly = vdupq_n_f32(1.061405429f);
  poly = vfmaq_f32(vdupq_n_f32(-1.453152027f), poly, t);
  poly = vfmaq_f32(vdupq_n_f32(1.421413741f), poly, t);
  poly = vfmaq_f32(vdupq_n_f32(-0.284496736f), poly, t);
  poly = vfmaq_f32(vdupq_n_f32(0.254829592f), poly, t);
  poly = vmulq_f32(poly, t);
  float32x4_t e = ExpNeon(vnegq_f32(vmulq_f32(x, x)));
  float32x4_t abs_erf = vfmsq_f32(one, poly, e);  // 1 - poly * e
  return vreinterpretq_f32_u32(
      vorrq_u32(vreinterpretq_u32_f32(abs_erf), vreinterpretq_u32_f32(sign)));
}

constexpr std::size_t kElemsPerChunk = 4096;

}  // namespace

void GeluNeon(const float* x, float* out, std::size_t n) {
  static constexpr float kInvSqrt2 = 0.70710678118654752440f;
  auto worker = [&](std::size_t start, std::size_t end) {
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t inv_sqrt2 = vdupq_n_f32(kInvSqrt2);
    const float32x4_t one = vdupq_n_f32(1.0f);
    std::size_t i = start;
    for (; i + 4 <= end; i += 4) {
      float32x4_t v = vld1q_f32(x + i);
      float32x4_t e = ErfNeon(vmulq_f32(v, inv_sqrt2));
      vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, half), vaddq_f32(one, e)));
    }
    for (; i < end; ++i) {
      out[i] = x[i] * 0.5f * (1.0f + std::erf(x[i] * kInvSqrt2));
    }
  };
  const int total_chunks =
      static_cast<int>((n + kElemsPerChunk - 1) / kElemsPerChunk);
  if (total_chunks > 1 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(
        0, total_chunks, /*grain=*/1, [&](int begin, int end) {
          for (int c = begin; c < end; ++c) {
            const std::size_t s = static_cast<std::size_t>(c) * kElemsPerChunk;
            worker(s, std::min(s + kElemsPerChunk, n));
          }
        });
  } else {
    worker(0, n);
  }
}

#endif  // ESM_KERNEL_NEON

}  // namespace esm::kernels
