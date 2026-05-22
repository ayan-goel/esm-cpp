#pragma once

// Shared NEON SIMD helpers for the AArch64 kernel TUs. Included only from
// within an `#ifdef ESM_KERNEL_NEON` block (the same .cpp files also compile a
// scalar-reference variant where this header must not be pulled in).

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

namespace esm::kernels::simd {

// NEON polynomial exp(x): range-reduce x = n*ln2 + r, then 2^n * exp(r) with a
// 5-term series. Same form as the AVX-512 path; FP32-accurate to ~3e-7 over
// the reduced range. Used by GELU's erf and by the attention softmax.
inline float32x4_t ExpNeon(float32x4_t x) {
  const float32x4_t log2e = vdupq_n_f32(1.44269504088896340736f);
  const float32x4_t ln2_hi = vdupq_n_f32(6.93145752e-1f);
  const float32x4_t ln2_lo = vdupq_n_f32(1.42860677e-6f);
  float32x4_t n_f = vrndnq_f32(vmulq_f32(x, log2e));
  float32x4_t r = vsubq_f32(vsubq_f32(x, vmulq_f32(n_f, ln2_hi)),
                            vmulq_f32(n_f, ln2_lo));
  float32x4_t p = vdupq_n_f32(1.0f / 120.0f);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  int32x4_t ni = vcvtq_s32_f32(n_f);
  int32x4_t e_bits = vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(e_bits));
}

}  // namespace esm::kernels::simd

#endif  // aarch64
