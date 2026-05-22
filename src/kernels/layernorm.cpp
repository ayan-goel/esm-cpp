#include "esm_cpp/kernels.h"

#include <cmath>
#include <cstddef>

// AVX-512 path; system headers at file scope (per the gemm_int8.cpp lesson).
#ifdef ESM_KERNEL_AVX512
#include <immintrin.h>

#include "esm_cpp/thread_pool.h"
#endif

#ifdef ESM_KERNEL_NEON
#include <arm_neon.h>

#include "esm_cpp/thread_pool.h"
#endif

namespace esm::kernels {

#ifdef ESM_KERNEL_REFERENCE

void LayerNormRef(const float* x, const float* gamma, const float* beta,
                  float eps, float* out, int num_rows, int d) {
  for (int r = 0; r < num_rows; ++r) {
    const float* xr = x + static_cast<long>(r) * d;
    float* yr = out + static_cast<long>(r) * d;
    double mean = 0.0;
    for (int i = 0; i < d; ++i) mean += xr[i];
    mean /= d;
    double var = 0.0;
    for (int i = 0; i < d; ++i) {
      double diff = xr[i] - mean;
      var += diff * diff;
    }
    var /= d;
    float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);
    for (int i = 0; i < d; ++i) {
      float normed = (xr[i] - static_cast<float>(mean)) * inv_std;
      yr[i] = normed * gamma[i] + beta[i];
    }
  }
}

#endif  // ESM_KERNEL_REFERENCE

#ifdef ESM_KERNEL_AVX512

// Two-pass numerically-stable LayerNorm. Pass 1 sums into 4 independent
// FP32 zmm accumulators (hides FMA latency, keeps per-lane partial sums
// short — for d <= 5120 each lane holds at most d/64 ≈ 80 values, so
// FP32 round-off stays well below the 1e-5 cross-check tolerance). Pass
// 2 computes the variance via FMA of (x - mean). Pass 3 emits the
// normalized + scaled + shifted output.
//
// ESM-2 d values (320 / 480 / 640 / 1280 / 2560) are all multiples of 16
// so the masked-tail path almost never runs in production; we keep it
// for the test sweep + future model dims.
void LayerNormAvx512(const float* x, const float* gamma, const float* beta,
                     float eps, float* out, int num_rows, int d) {
  const float inv_d = 1.0f / static_cast<float>(d);
  auto compute_row = [&](int r) {
    const float* xr = x + static_cast<long>(r) * d;
    float* yr = out + static_cast<long>(r) * d;

    // Pass 1: mean.
    __m512 s0 = _mm512_setzero_ps();
    __m512 s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps();
    __m512 s3 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 64 <= d; i += 64) {
      s0 = _mm512_add_ps(s0, _mm512_loadu_ps(xr + i));
      s1 = _mm512_add_ps(s1, _mm512_loadu_ps(xr + i + 16));
      s2 = _mm512_add_ps(s2, _mm512_loadu_ps(xr + i + 32));
      s3 = _mm512_add_ps(s3, _mm512_loadu_ps(xr + i + 48));
    }
    for (; i + 16 <= d; i += 16) {
      s0 = _mm512_add_ps(s0, _mm512_loadu_ps(xr + i));
    }
    float scalar_sum_tail = 0.0f;
    for (; i < d; ++i) scalar_sum_tail += xr[i];
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    const float mean =
        (_mm512_reduce_add_ps(s) + scalar_sum_tail) * inv_d;

    // Pass 2: variance of (x - mean).
    const __m512 mv = _mm512_set1_ps(mean);
    __m512 v0 = _mm512_setzero_ps();
    __m512 v1 = _mm512_setzero_ps();
    __m512 v2 = _mm512_setzero_ps();
    __m512 v3 = _mm512_setzero_ps();
    i = 0;
    for (; i + 64 <= d; i += 64) {
      __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(xr + i), mv);
      __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(xr + i + 16), mv);
      __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(xr + i + 32), mv);
      __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(xr + i + 48), mv);
      v0 = _mm512_fmadd_ps(d0, d0, v0);
      v1 = _mm512_fmadd_ps(d1, d1, v1);
      v2 = _mm512_fmadd_ps(d2, d2, v2);
      v3 = _mm512_fmadd_ps(d3, d3, v3);
    }
    for (; i + 16 <= d; i += 16) {
      __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(xr + i), mv);
      v0 = _mm512_fmadd_ps(d0, d0, v0);
    }
    float scalar_var_tail = 0.0f;
    for (; i < d; ++i) {
      const float diff = xr[i] - mean;
      scalar_var_tail += diff * diff;
    }
    __m512 vv = _mm512_add_ps(_mm512_add_ps(v0, v1), _mm512_add_ps(v2, v3));
    const float var =
        (_mm512_reduce_add_ps(vv) + scalar_var_tail) * inv_d;
    const float inv_std = 1.0f / std::sqrt(var + eps);

    // Pass 3: (x - mean) * inv_std * gamma + beta.
    const __m512 iv = _mm512_set1_ps(inv_std);
    i = 0;
    for (; i + 16 <= d; i += 16) {
      __m512 xv = _mm512_loadu_ps(xr + i);
      __m512 normed = _mm512_mul_ps(_mm512_sub_ps(xv, mv), iv);
      __m512 g = _mm512_loadu_ps(gamma + i);
      __m512 b = _mm512_loadu_ps(beta + i);
      _mm512_storeu_ps(yr + i, _mm512_fmadd_ps(normed, g, b));
    }
    if (i < d) {
      const int tail = d - i;
      const __mmask16 mask = static_cast<__mmask16>((1u << tail) - 1u);
      __m512 xv = _mm512_maskz_loadu_ps(mask, xr + i);
      __m512 normed = _mm512_mul_ps(_mm512_sub_ps(xv, mv), iv);
      __m512 g = _mm512_maskz_loadu_ps(mask, gamma + i);
      __m512 b = _mm512_maskz_loadu_ps(mask, beta + i);
      _mm512_mask_storeu_ps(yr + i, mask, _mm512_fmadd_ps(normed, g, b));
    }
  };
  // Each row is independent; parallel_for across rows when called from
  // outside a worker (TransformerBlock dispatches LN on the main thread
  // sequentially, so the inner parallel_for is safe).
  if (num_rows > 1 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(
        0, num_rows, /*grain=*/1, [&](int begin, int end) {
          for (int r = begin; r < end; ++r) compute_row(r);
        });
  } else {
    for (int r = 0; r < num_rows; ++r) compute_row(r);
  }
}

#endif  // ESM_KERNEL_AVX512

#ifdef ESM_KERNEL_NEON

// Three-pass NEON LayerNorm mirroring the AVX-512 path: 4 independent
// float32x4 accumulators per pass keep partial-sum round-off well under the
// 1e-5 cross-check tolerance for d up to 5120. Parallelized across rows.
void LayerNormNeon(const float* x, const float* gamma, const float* beta,
                   float eps, float* out, int num_rows, int d) {
  const float inv_d = 1.0f / static_cast<float>(d);
  auto compute_row = [&](int r) {
    const float* xr = x + static_cast<long>(r) * d;
    float* yr = out + static_cast<long>(r) * d;

    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= d; i += 16) {
      s0 = vaddq_f32(s0, vld1q_f32(xr + i));
      s1 = vaddq_f32(s1, vld1q_f32(xr + i + 4));
      s2 = vaddq_f32(s2, vld1q_f32(xr + i + 8));
      s3 = vaddq_f32(s3, vld1q_f32(xr + i + 12));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < d; ++i) sum += xr[i];
    const float mean = sum * inv_d;

    const float32x4_t mv = vdupq_n_f32(mean);
    float32x4_t v0 = vdupq_n_f32(0.0f), v1 = v0, v2 = v0, v3 = v0;
    i = 0;
    for (; i + 16 <= d; i += 16) {
      float32x4_t d0 = vsubq_f32(vld1q_f32(xr + i), mv);
      float32x4_t d1 = vsubq_f32(vld1q_f32(xr + i + 4), mv);
      float32x4_t d2 = vsubq_f32(vld1q_f32(xr + i + 8), mv);
      float32x4_t d3 = vsubq_f32(vld1q_f32(xr + i + 12), mv);
      v0 = vfmaq_f32(v0, d0, d0);
      v1 = vfmaq_f32(v1, d1, d1);
      v2 = vfmaq_f32(v2, d2, d2);
      v3 = vfmaq_f32(v3, d3, d3);
    }
    float var_sum =
        vaddvq_f32(vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)));
    for (; i < d; ++i) {
      const float diff = xr[i] - mean;
      var_sum += diff * diff;
    }
    const float inv_std = 1.0f / std::sqrt(var_sum * inv_d + eps);

    const float32x4_t iv = vdupq_n_f32(inv_std);
    i = 0;
    for (; i + 4 <= d; i += 4) {
      float32x4_t normed = vmulq_f32(vsubq_f32(vld1q_f32(xr + i), mv), iv);
      vst1q_f32(yr + i,
                vfmaq_f32(vld1q_f32(beta + i), normed, vld1q_f32(gamma + i)));
    }
    for (; i < d; ++i) {
      yr[i] = (xr[i] - mean) * inv_std * gamma[i] + beta[i];
    }
  };
  if (num_rows > 1 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(
        0, num_rows, /*grain=*/1, [&](int begin, int end) {
          for (int r = begin; r < end; ++r) compute_row(r);
        });
  } else {
    for (int r = 0; r < num_rows; ++r) compute_row(r);
  }
}

#endif  // ESM_KERNEL_NEON

}  // namespace esm::kernels
