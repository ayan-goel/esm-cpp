#include "esm_cpp/kernels.h"

#include <cstddef>

#ifdef ESM_KERNEL_AVX512
#include <immintrin.h>

#include "esm_cpp/thread_pool.h"
#endif

namespace esm::kernels {

#ifdef ESM_KERNEL_REFERENCE

// rotate_half(x) = cat(-x_second_half, x_first_half) along the last dim.
// apply_rotary_pos_emb(x, cos, sin) = x * cos + rotate_half(x) * sin
// We do this in-place on x for all heads.
void RopeApplyInplaceRef(float* x, const float* cos, const float* sin,
                         int num_heads, int seq_len, int head_dim) {
  const int half = head_dim / 2;
  for (int h = 0; h < num_heads; ++h) {
    for (int t = 0; t < seq_len; ++t) {
      float* xrow = x + (static_cast<long>(h) * seq_len + t) * head_dim;
      const float* crow = cos + static_cast<long>(t) * head_dim;
      const float* srow = sin + static_cast<long>(t) * head_dim;
      for (int i = 0; i < half; ++i) {
        float x1 = xrow[i];
        float x2 = xrow[i + half];
        xrow[i] = x1 * crow[i] + (-x2) * srow[i];
        xrow[i + half] = x2 * crow[i + half] + x1 * srow[i + half];
      }
    }
  }
}

// Same RoPE math as above but over the token-major [T, H, head_dim]
// layout that AttentionVarlen consumes. Positions restart at every
// cu_seqlens boundary so each sequence sees rotations from 0..seq_len-1
// regardless of where in the packed buffer it lives. cos/sin tables
// are shared across the batch.
void RopeApplyVarlenRef(float* x, const float* cos, const float* sin,
                        const int* cu_seqlens, int batch_size, int num_heads,
                        int head_dim) {
  const int half = head_dim / 2;
  for (int b = 0; b < batch_size; ++b) {
    const int seq_start = cu_seqlens[b];
    const int seq_end = cu_seqlens[b + 1];
    for (int p = 0; p < seq_end - seq_start; ++p) {
      const int t_global = seq_start + p;
      const float* crow = cos + static_cast<long>(p) * head_dim;
      const float* srow = sin + static_cast<long>(p) * head_dim;
      for (int h = 0; h < num_heads; ++h) {
        float* xrow =
            x + (static_cast<long>(t_global) * num_heads + h) * head_dim;
        for (int i = 0; i < half; ++i) {
          float x1 = xrow[i];
          float x2 = xrow[i + half];
          xrow[i] = x1 * crow[i] + (-x2) * srow[i];
          xrow[i + half] = x2 * crow[i + half] + x1 * srow[i + half];
        }
      }
    }
  }
}

#endif  // ESM_KERNEL_REFERENCE

#ifdef ESM_KERNEL_AVX512

// Packed-varlen RoPE in [T, H, dh] layout. The half-then-half pattern
// is load-bearing — Llama/GPT-NeoX style, NOT interleaved (see CLAUDE.md
// and esm/rotary_embedding.py). We rely on RopeBuildTables's invariant
// that crow[i] == crow[i + half] and srow[i] == srow[i + half], so the
// same SIMD load of cos/sin serves both halves of the rotation.
//
// Per-position math (per CLAUDE.md):
//   new_lo = x_lo * c - x_hi * s
//   new_hi = x_hi * c + x_lo * s
//
// Vectorized over the first-half index i in chunks of 16. Parallelized
// across (batch × num_heads) — same fan-out as the AVX-512 attention
// kernel (S6 phase 6). Skips dispatch under InGlobalPoolWorker() to
// avoid the nested-pool deadlock pattern.
void RopeApplyVarlenAvx512(float* x, const float* cos, const float* sin,
                            const int* cu_seqlens, int batch_size,
                            int num_heads, int head_dim) {
  const int half = head_dim / 2;

  auto run_one_pos = [&](int t_global, int p, int h) {
    const float* crow =
        cos + static_cast<long>(p) * head_dim;
    const float* srow =
        sin + static_cast<long>(p) * head_dim;
    float* xrow =
        x + (static_cast<long>(t_global) * num_heads + h) * head_dim;

    int i = 0;
    for (; i + 16 <= half; i += 16) {
      __m512 x_lo = _mm512_loadu_ps(xrow + i);
      __m512 x_hi = _mm512_loadu_ps(xrow + i + half);
      __m512 c = _mm512_loadu_ps(crow + i);
      __m512 s = _mm512_loadu_ps(srow + i);
      __m512 lo_c_term = _mm512_mul_ps(x_lo, c);
      __m512 hi_s_term = _mm512_mul_ps(x_lo, s);
      // new_lo = x_lo * c - x_hi * s = -(x_hi * s) + (x_lo * c)
      __m512 new_lo = _mm512_fnmadd_ps(x_hi, s, lo_c_term);
      // new_hi = x_hi * c + x_lo * s
      __m512 new_hi = _mm512_fmadd_ps(x_hi, c, hi_s_term);
      _mm512_storeu_ps(xrow + i, new_lo);
      _mm512_storeu_ps(xrow + i + half, new_hi);
    }
    if (i < half) {
      const int tail = half - i;
      const __mmask16 mask = static_cast<__mmask16>((1u << tail) - 1u);
      __m512 x_lo = _mm512_maskz_loadu_ps(mask, xrow + i);
      __m512 x_hi = _mm512_maskz_loadu_ps(mask, xrow + i + half);
      __m512 c = _mm512_maskz_loadu_ps(mask, crow + i);
      __m512 s = _mm512_maskz_loadu_ps(mask, srow + i);
      __m512 lo_c_term = _mm512_mul_ps(x_lo, c);
      __m512 hi_s_term = _mm512_mul_ps(x_lo, s);
      __m512 new_lo = _mm512_fnmadd_ps(x_hi, s, lo_c_term);
      __m512 new_hi = _mm512_fmadd_ps(x_hi, c, hi_s_term);
      _mm512_mask_storeu_ps(xrow + i, mask, new_lo);
      _mm512_mask_storeu_ps(xrow + i + half, mask, new_hi);
    }
  };

  const int total_bh = batch_size * num_heads;
  auto worker = [&](int bh_begin, int bh_end) {
    for (int bh = bh_begin; bh < bh_end; ++bh) {
      const int b = bh / num_heads;
      const int h = bh % num_heads;
      const int seq_start = cu_seqlens[b];
      const int seq_end = cu_seqlens[b + 1];
      const int seq_len = seq_end - seq_start;
      for (int p = 0; p < seq_len; ++p) {
        run_one_pos(seq_start + p, p, h);
      }
    }
  };
  if (total_bh > 1 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(0, total_bh, /*grain=*/1, worker);
  } else {
    worker(0, total_bh);
  }
}

#endif  // ESM_KERNEL_AVX512

}  // namespace esm::kernels
