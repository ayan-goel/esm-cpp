#include "esm_cpp/kernels.h"
#include "esm_cpp/quant.h"

// AVX-512 path needs intrinsics + a few STL helpers. These must live at
// file scope, NOT inside namespace esm::kernels — system headers nested
// in a namespace leak C-stdlib symbols (::abs, ::malloc, ::div_t, ...)
// into that namespace and break libstdc++'s <cstdlib> using-declarations.
#ifdef ESM_KERNEL_AVX512
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "esm_cpp/thread_pool.h"
#endif

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

#ifdef ESM_KERNEL_AVX512

// AVX-512 VNNI W8A8 microkernel. Production INT8 GEMM path for x86
// with AVX-512+VNNI (Ice Lake, Sapphire Rapids, etc.). Inner loop uses
// VPDPBUSD: u8 (activation) x s8 (weight) -> s32 accumulator, 4 K
// values per lane, 16 N lanes per zmm register.
//
// Layout:
//   A: row-major [M, K] FP32 input. Quantized on the fly to u8 with
//      per-tensor scale = max(|A|) / 127 and zero-point 128. Symmetric
//      INT8 in [-127, 127] then bias-shifted to u8 in [1, 255].
//   W: per-channel symmetric INT8, range [-127, 127]. Original layout
//      is row-major [N, K]. We repack into [N/16 panels, K/4 tiles,
//      16 N x 4 K bytes per tile] = 64-byte VPDPBUSD-friendly chunks.
//   C: row-major [M, N] FP32 output. Bias is FP32.
//
// Zero-point correction: VPDPBUSD computes sum_k(u8[k] * s8[k]) where
// u8[k] = q[k] + 128. To recover sum_k(q*w) we subtract 128 * sum_k(w)
// per output column. We precompute col_sum[N] once per call.
//
// Scale folding: final output is
//   C[m,n] = (raw_acc - 128 * col_sum[n]) * w_scale[n] * act_scale + bias[n]
//
// Threading: parallel_for across M when called from the main thread.
// Skips parallel dispatch when InGlobalPoolWorker() to avoid nested
// parallel_for deadlock (same pattern as AttentionVarlen).

void LinearInt8Ref(const float* A, const esm::quant::QuantizedTensor& W,
                   const float* bias, float* C, int M, int N, int K);

namespace {

// Per-thread scratch buffers reused across LinearVnni calls. Sized on
// demand; never shrunk. Avoids per-call malloc on the hot path.
thread_local std::vector<std::uint8_t> g_a_u8;
thread_local std::vector<std::int8_t> g_w_packed;
thread_local std::vector<std::int32_t> g_col_sum;

// Pack W from row-major [N, K] into VPDPBUSD tile order. Output layout:
// for each N-block of 16, for each K-block of 4: one 64-byte tile
// containing [N0:k0,k1,k2,k3, N1:k0,k1,k2,k3, ..., N15:k0,k1,k2,k3].
// N tail (< 16) zero-padded; K tail (< 4) zero-padded.
void PackWeight(const std::int8_t* w, int N, int K, int K_pad,
                std::int8_t* packed) {
  std::memset(packed,
              0, static_cast<std::size_t>(N) * static_cast<std::size_t>(K_pad));
  for (int nb = 0; nb < N; nb += 16) {
    const int n_block = std::min(16, N - nb);
    std::int8_t* panel = packed +
        static_cast<long>(nb) * static_cast<long>(K_pad);
    for (int kb = 0; kb < K; kb += 4) {
      std::int8_t* tile = panel + static_cast<long>(kb) * 16;
      const int k_step = std::min(4, K - kb);
      for (int nn = 0; nn < n_block; ++nn) {
        for (int kk = 0; kk < k_step; ++kk) {
          tile[nn * 4 + kk] =
              w[(nb + nn) * static_cast<long>(K) + (kb + kk)];
        }
      }
    }
  }
}

// One-row finalize: take an s32 accumulator zmm (16 lanes for n..n+15),
// apply zero-point correction, scale by per-channel + activation scales,
// add bias, store as FP32.
inline void FinalizeStore16(__m512i acc_raw, const std::int32_t* col_sum,
                             const float* w_scale, const float* bias,
                             float act_scale, float* out_row) {
  // raw_acc - 128 * col_sum  (all s32)
  __m512i cs = _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(col_sum));
  __m512i corrected =
      _mm512_sub_epi32(acc_raw, _mm512_slli_epi32(cs, 7));  // *128
  // s32 -> f32
  __m512 fp = _mm512_cvtepi32_ps(corrected);
  // scale[n] * act_scale
  __m512 ws = _mm512_loadu_ps(w_scale);
  __m512 combined = _mm512_mul_ps(ws, _mm512_set1_ps(act_scale));
  // out = corrected_fp * combined + bias
  __m512 b = _mm512_loadu_ps(bias);
  __m512 out = _mm512_fmadd_ps(fp, combined, b);
  _mm512_storeu_ps(out_row, out);
}

// One-row, partial-N (n_tail < 16) finalize via masked store.
inline void FinalizeStoreTail(__m512i acc_raw, const std::int32_t* col_sum,
                               const float* w_scale, const float* bias,
                               float act_scale, float* out_row, int n_tail) {
  __mmask16 mask = static_cast<__mmask16>((1u << n_tail) - 1);
  __m512i cs = _mm512_maskz_loadu_epi32(mask, col_sum);
  __m512i corrected =
      _mm512_sub_epi32(acc_raw, _mm512_slli_epi32(cs, 7));
  __m512 fp = _mm512_cvtepi32_ps(corrected);
  __m512 ws = _mm512_maskz_loadu_ps(mask, w_scale);
  __m512 combined = _mm512_mul_ps(ws, _mm512_set1_ps(act_scale));
  __m512 b = _mm512_maskz_loadu_ps(mask, bias);
  __m512 out = _mm512_fmadd_ps(fp, combined, b);
  _mm512_mask_storeu_ps(out_row, mask, out);
}

// Process rows [m_begin, m_end) of the output. Activations a_u8 are
// already quantized [M, K], weights are packed into 64-byte tiles per
// (N-block, K-block). Each row of C is produced independently.
void ComputeRows(const std::uint8_t* a_u8, const std::int8_t* w_packed,
                  const float* w_scale, const std::int32_t* col_sum,
                  const float* bias, float act_scale, int K, int K_pad,
                  int N, float* C, int m_begin, int m_end) {
  // Zero-bias path needs a small zero buffer for the FinalizeStore16 path
  // since it always loads 16 floats from `bias`. Tiny stack alloc per task.
  alignas(64) float zero_bias[16] = {0};
  for (int m = m_begin; m < m_end; ++m) {
    const std::uint8_t* a_row =
        a_u8 + static_cast<long>(m) * static_cast<long>(K);
    float* c_row = C + static_cast<long>(m) * static_cast<long>(N);
    int nb = 0;
    for (; nb + 16 <= N; nb += 16) {
      __m512i acc = _mm512_setzero_si512();
      const std::int8_t* panel =
          w_packed + static_cast<long>(nb) * static_cast<long>(K_pad);
      int kb = 0;
      for (; kb + 4 <= K; kb += 4) {
        std::uint32_t a4;
        std::memcpy(&a4, &a_row[kb], 4);
        __m512i a_bcast = _mm512_set1_epi32(static_cast<int>(a4));
        __m512i w_tile = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(panel + kb * 16));
        acc = _mm512_dpbusd_epi32(acc, a_bcast, w_tile);
      }
      if (kb < K) {
        // K tail: load partial 4-byte chunk with zero-point bytes (128)
        // for the missing positions so the contribution is zeroed out
        // after the zero-point correction.
        alignas(4) std::uint8_t tmp[4] = {128, 128, 128, 128};
        for (int kk = 0; kk < K - kb; ++kk) tmp[kk] = a_row[kb + kk];
        std::uint32_t a4;
        std::memcpy(&a4, tmp, 4);
        __m512i a_bcast = _mm512_set1_epi32(static_cast<int>(a4));
        __m512i w_tile = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(panel + kb * 16));
        acc = _mm512_dpbusd_epi32(acc, a_bcast, w_tile);
      }
      FinalizeStore16(acc, col_sum + nb, w_scale + nb,
                       bias ? (bias + nb) : zero_bias,
                       act_scale, c_row + nb);
    }
    if (nb < N) {
      const int n_tail = N - nb;
      __m512i acc = _mm512_setzero_si512();
      const std::int8_t* panel =
          w_packed + static_cast<long>(nb) * static_cast<long>(K_pad);
      int kb = 0;
      for (; kb + 4 <= K; kb += 4) {
        std::uint32_t a4;
        std::memcpy(&a4, &a_row[kb], 4);
        __m512i a_bcast = _mm512_set1_epi32(static_cast<int>(a4));
        __m512i w_tile = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(panel + kb * 16));
        acc = _mm512_dpbusd_epi32(acc, a_bcast, w_tile);
      }
      if (kb < K) {
        alignas(4) std::uint8_t tmp[4] = {128, 128, 128, 128};
        for (int kk = 0; kk < K - kb; ++kk) tmp[kk] = a_row[kb + kk];
        std::uint32_t a4;
        std::memcpy(&a4, tmp, 4);
        __m512i a_bcast = _mm512_set1_epi32(static_cast<int>(a4));
        __m512i w_tile = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(panel + kb * 16));
        acc = _mm512_dpbusd_epi32(acc, a_bcast, w_tile);
      }
      FinalizeStoreTail(acc, col_sum + nb, w_scale + nb,
                         bias ? (bias + nb) : zero_bias,
                         act_scale, c_row + nb, n_tail);
    }
  }
}

}  // namespace

void LinearVnni(const float* A, const esm::quant::QuantizedTensor& W,
                const float* bias, float* C, int M, int N, int K) {
  // K must be at least 1; if it's zero just zero the output.
  if (K <= 0 || M <= 0 || N <= 0) {
    if (M > 0 && N > 0 && C) {
      std::memset(C, 0,
                  static_cast<std::size_t>(M) * static_cast<std::size_t>(N) *
                      sizeof(float));
    }
    return;
  }

  // Step 1: per-tensor activation scale = max(|A|) / 127.
  float a_max = 0.0f;
  const std::size_t MK = static_cast<std::size_t>(M) *
                          static_cast<std::size_t>(K);
  for (std::size_t i = 0; i < MK; ++i) {
    const float v = std::fabs(A[i]);
    if (v > a_max) a_max = v;
  }
  const float act_scale = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
  const float inv_act_scale = 1.0f / act_scale;

  // Step 2: quantize A to u8 with zero-point 128 (q in [-127, 127] then
  // shifted by +128 -> u8 in [1, 255]).
  if (g_a_u8.size() < MK) g_a_u8.resize(MK);
  std::uint8_t* a_u8 = g_a_u8.data();
  for (std::size_t i = 0; i < MK; ++i) {
    int q = static_cast<int>(std::lround(A[i] * inv_act_scale));
    if (q < -127) q = -127;
    if (q > 127) q = 127;
    a_u8[i] = static_cast<std::uint8_t>(q + 128);
  }

  // Step 3: pack W into 64-byte VPDPBUSD tiles. K_pad rounds K up to a
  // multiple of 4 (production ESM shapes are always divisible by 4).
  const int K_pad = (K + 3) & ~3;
  const std::size_t packed_size =
      static_cast<std::size_t>(N) * static_cast<std::size_t>(K_pad);
  if (g_w_packed.size() < packed_size) g_w_packed.resize(packed_size);
  std::int8_t* w_packed = g_w_packed.data();
  PackWeight(W.packed.data(), N, K, K_pad, w_packed);

  // Step 4: column sums for zero-point correction.
  if (g_col_sum.size() < static_cast<std::size_t>(N)) {
    g_col_sum.resize(static_cast<std::size_t>(N));
  }
  std::int32_t* col_sum = g_col_sum.data();
  for (int n = 0; n < N; ++n) {
    std::int32_t s = 0;
    const std::int8_t* w_row =
        W.packed.data() + static_cast<long>(n) * static_cast<long>(K);
    for (int k = 0; k < K; ++k) s += w_row[k];
    col_sum[n] = s;
  }

  // Step 5: GEMM. Parallelize across M when we're not already inside
  // a pool worker (would nested-deadlock if we are).
  const float* w_scale = W.per_channel_scales.data();
  if (M > 1 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(
        0, M, /*grain=*/1, [&](int begin, int end) {
          ComputeRows(a_u8, w_packed, w_scale, col_sum, bias, act_scale, K,
                      K_pad, N, C, begin, end);
        });
  } else {
    ComputeRows(a_u8, w_packed, w_scale, col_sum, bias, act_scale, K, K_pad,
                N, C, 0, M);
  }
}

#endif  // ESM_KERNEL_AVX512

}  // namespace esm::kernels
