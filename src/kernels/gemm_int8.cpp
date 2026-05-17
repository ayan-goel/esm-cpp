#include "esm_cpp/kernels.h"
#include "esm_cpp/quant.h"

// AVX-512 path needs intrinsics + a few STL helpers. These must live at
// file scope, NOT inside namespace esm::kernels — system headers nested
// in a namespace leak C-stdlib symbols (::abs, ::malloc, ::div_t, ...)
// into that namespace and break libstdc++'s <cstdlib> using-declarations.
#ifdef ESM_KERNEL_AVX512
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

// Per-thread scratch buffer reused across LinearVnni calls. Sized on
// demand; never shrunk. Avoids per-call malloc on the hot path. Weight
// packing + col_sum live on the QuantizedTensor itself (BuildVnniCache);
// only the activation u8 staging buffer is per-call.
thread_local std::vector<std::uint8_t> g_a_u8;

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

// Goto-style 6 rows × 32 N microkernel. 12 independent zmm s32
// accumulators (6 rows × 2 N-tiles of 16 lanes each). Two weight loads +
// 6 broadcasts feed 12 VPDPBUSDs per K=4 step. Sapphire Rapids issues
// 2 VPDPBUSDs/cycle, so 12 deps spread across 6 cycles fills the FMA
// pipe — roughly 2× the single-accumulator kernel's throughput on the
// 650M shapes.
//
// M-tail (rows past the last full 6-block) and N-tail (cols past the
// last full 32-block, including any 16-N remainder) fall through to the
// 1-row × 16-N kernel above. The 16-N tile layout in packed_vnni is
// shared between both kernels — we just process two adjacent panels per
// microkernel invocation.
inline void Kernel6x32(const std::uint8_t* a_block, const std::int8_t* w_panel0,
                       const std::int8_t* w_panel1, const float* w_scale,
                       const std::int32_t* col_sum, const float* bias,
                       float act_scale, int K, int K_main, float* c_base,
                       int N) {
  __m512i a00 = _mm512_setzero_si512(), a01 = _mm512_setzero_si512();
  __m512i a10 = _mm512_setzero_si512(), a11 = _mm512_setzero_si512();
  __m512i a20 = _mm512_setzero_si512(), a21 = _mm512_setzero_si512();
  __m512i a30 = _mm512_setzero_si512(), a31 = _mm512_setzero_si512();
  __m512i a40 = _mm512_setzero_si512(), a41 = _mm512_setzero_si512();
  __m512i a50 = _mm512_setzero_si512(), a51 = _mm512_setzero_si512();
  for (int kb = 0; kb < K_main; kb += 4) {
    const __m512i w0 = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(w_panel0 + kb * 16));
    const __m512i w1 = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(w_panel1 + kb * 16));
    std::uint32_t a0w, a1w, a2w, a3w, a4w, a5w;
    std::memcpy(&a0w, a_block + 0 * K + kb, 4);
    std::memcpy(&a1w, a_block + 1 * K + kb, 4);
    std::memcpy(&a2w, a_block + 2 * K + kb, 4);
    std::memcpy(&a3w, a_block + 3 * K + kb, 4);
    std::memcpy(&a4w, a_block + 4 * K + kb, 4);
    std::memcpy(&a5w, a_block + 5 * K + kb, 4);
    const __m512i a0 = _mm512_set1_epi32(static_cast<int>(a0w));
    const __m512i a1 = _mm512_set1_epi32(static_cast<int>(a1w));
    const __m512i a2 = _mm512_set1_epi32(static_cast<int>(a2w));
    const __m512i a3 = _mm512_set1_epi32(static_cast<int>(a3w));
    const __m512i a4 = _mm512_set1_epi32(static_cast<int>(a4w));
    const __m512i a5 = _mm512_set1_epi32(static_cast<int>(a5w));
    a00 = _mm512_dpbusd_epi32(a00, a0, w0);
    a01 = _mm512_dpbusd_epi32(a01, a0, w1);
    a10 = _mm512_dpbusd_epi32(a10, a1, w0);
    a11 = _mm512_dpbusd_epi32(a11, a1, w1);
    a20 = _mm512_dpbusd_epi32(a20, a2, w0);
    a21 = _mm512_dpbusd_epi32(a21, a2, w1);
    a30 = _mm512_dpbusd_epi32(a30, a3, w0);
    a31 = _mm512_dpbusd_epi32(a31, a3, w1);
    a40 = _mm512_dpbusd_epi32(a40, a4, w0);
    a41 = _mm512_dpbusd_epi32(a41, a4, w1);
    a50 = _mm512_dpbusd_epi32(a50, a5, w0);
    a51 = _mm512_dpbusd_epi32(a51, a5, w1);
  }
  // K-tail (K % 4 != 0): one extra VPDPBUSD with zero-point-padded A
  // bytes so the missing positions contribute nothing after the
  // -128 * col_sum correction at finalize. Weight panel padding bytes
  // are already zero from BuildVnniCache.
  if (K_main < K) {
    const __m512i w0 = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(w_panel0 + K_main * 16));
    const __m512i w1 = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(w_panel1 + K_main * 16));
    alignas(4) std::uint8_t tmp[6][4] = {
        {128, 128, 128, 128}, {128, 128, 128, 128}, {128, 128, 128, 128},
        {128, 128, 128, 128}, {128, 128, 128, 128}, {128, 128, 128, 128},
    };
    const int k_step = K - K_main;
    for (int r = 0; r < 6; ++r) {
      for (int kk = 0; kk < k_step; ++kk) {
        tmp[r][kk] = a_block[r * K + K_main + kk];
      }
    }
    std::uint32_t a0w, a1w, a2w, a3w, a4w, a5w;
    std::memcpy(&a0w, tmp[0], 4);
    std::memcpy(&a1w, tmp[1], 4);
    std::memcpy(&a2w, tmp[2], 4);
    std::memcpy(&a3w, tmp[3], 4);
    std::memcpy(&a4w, tmp[4], 4);
    std::memcpy(&a5w, tmp[5], 4);
    const __m512i a0 = _mm512_set1_epi32(static_cast<int>(a0w));
    const __m512i a1 = _mm512_set1_epi32(static_cast<int>(a1w));
    const __m512i a2 = _mm512_set1_epi32(static_cast<int>(a2w));
    const __m512i a3 = _mm512_set1_epi32(static_cast<int>(a3w));
    const __m512i a4 = _mm512_set1_epi32(static_cast<int>(a4w));
    const __m512i a5 = _mm512_set1_epi32(static_cast<int>(a5w));
    a00 = _mm512_dpbusd_epi32(a00, a0, w0);
    a01 = _mm512_dpbusd_epi32(a01, a0, w1);
    a10 = _mm512_dpbusd_epi32(a10, a1, w0);
    a11 = _mm512_dpbusd_epi32(a11, a1, w1);
    a20 = _mm512_dpbusd_epi32(a20, a2, w0);
    a21 = _mm512_dpbusd_epi32(a21, a2, w1);
    a30 = _mm512_dpbusd_epi32(a30, a3, w0);
    a31 = _mm512_dpbusd_epi32(a31, a3, w1);
    a40 = _mm512_dpbusd_epi32(a40, a4, w0);
    a41 = _mm512_dpbusd_epi32(a41, a4, w1);
    a50 = _mm512_dpbusd_epi32(a50, a5, w0);
    a51 = _mm512_dpbusd_epi32(a51, a5, w1);
  }
  alignas(64) static const float kZeroBias[16] = {0};
  const float* b0 = bias ? bias : kZeroBias;
  const float* b1 = bias ? bias + 16 : kZeroBias;
  FinalizeStore16(a00, col_sum, w_scale, b0, act_scale, c_base + 0 * N);
  FinalizeStore16(a01, col_sum + 16, w_scale + 16, b1, act_scale,
                   c_base + 0 * N + 16);
  FinalizeStore16(a10, col_sum, w_scale, b0, act_scale, c_base + 1 * N);
  FinalizeStore16(a11, col_sum + 16, w_scale + 16, b1, act_scale,
                   c_base + 1 * N + 16);
  FinalizeStore16(a20, col_sum, w_scale, b0, act_scale, c_base + 2 * N);
  FinalizeStore16(a21, col_sum + 16, w_scale + 16, b1, act_scale,
                   c_base + 2 * N + 16);
  FinalizeStore16(a30, col_sum, w_scale, b0, act_scale, c_base + 3 * N);
  FinalizeStore16(a31, col_sum + 16, w_scale + 16, b1, act_scale,
                   c_base + 3 * N + 16);
  FinalizeStore16(a40, col_sum, w_scale, b0, act_scale, c_base + 4 * N);
  FinalizeStore16(a41, col_sum + 16, w_scale + 16, b1, act_scale,
                   c_base + 4 * N + 16);
  FinalizeStore16(a50, col_sum, w_scale, b0, act_scale, c_base + 5 * N);
  FinalizeStore16(a51, col_sum + 16, w_scale + 16, b1, act_scale,
                   c_base + 5 * N + 16);
}

// Compute output rows [m_begin, m_end) using the Goto 6×32 microkernel
// where the row block aligns, falling back to the 1×16 kernel for M-tail
// (< 6 rows remaining). N is required to be a multiple of 32 for the
// microkernel path; production ESM dims (320 / 480 / 640 / 1280 / 1920 /
// 2560 / 5120) are all multiples of 32 so this is always taken at
// inference time. Test shapes that violate it fall through to the legacy
// kernel for the whole range.
void ComputeRowsGoto(const std::uint8_t* a_u8, const std::int8_t* w_packed,
                     const float* w_scale, const std::int32_t* col_sum,
                     const float* bias, float act_scale, int K, int K_pad,
                     int N, float* C, int m_begin, int m_end) {
  if ((N & 31) != 0) {
    ComputeRows(a_u8, w_packed, w_scale, col_sum, bias, act_scale, K, K_pad,
                N, C, m_begin, m_end);
    return;
  }
  const int K_main = K & ~3;
  int m = m_begin;
  for (; m + 6 <= m_end; m += 6) {
    const std::uint8_t* a_block =
        a_u8 + static_cast<long>(m) * static_cast<long>(K);
    float* c_row = C + static_cast<long>(m) * static_cast<long>(N);
    for (int nb = 0; nb < N; nb += 32) {
      const std::int8_t* w_panel0 =
          w_packed + static_cast<long>(nb) * static_cast<long>(K_pad);
      const std::int8_t* w_panel1 =
          w_packed + static_cast<long>(nb + 16) * static_cast<long>(K_pad);
      Kernel6x32(a_block, w_panel0, w_panel1, w_scale + nb, col_sum + nb,
                 bias ? bias + nb : nullptr, act_scale, K, K_main,
                 c_row + nb, N);
    }
  }
  if (m < m_end) {
    ComputeRows(a_u8, w_packed, w_scale, col_sum, bias, act_scale, K, K_pad,
                N, C, m, m_end);
  }
}

}  // namespace

// Activation prefix: compute the per-tensor activation scale (max|A|/127)
// and quantize A to u8 with zero-point 128. Parallelized over MK in
// 16-element-aligned chunks; the absmax pass reduces via a CAS on the
// IEEE-754 bit pattern (non-negative float bits are monotonic in float
// magnitude, so the bit-pattern max is the float max).
//
// Called from LinearVnni and from LinearAmx (forward-declared in
// gemm_amx.cpp) so both INT8 paths share the same parallel prefix.
// Threshold avoids pool-dispatch overhead on tiny MK.
void QuantizeActPrefixAvx512(const float* A, std::size_t MK,
                              std::uint8_t* a_u8, float* act_scale_out) {
  constexpr std::size_t kParallelThreshold = 64 * 1024;
  auto run_serial_absmax = [&]() -> float {
    __m512 v_max = _mm512_setzero_ps();
    const __m512 abs_mask =
        _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    std::size_t i = 0;
    for (; i + 16 <= MK; i += 16) {
      v_max = _mm512_max_ps(v_max,
                            _mm512_and_ps(_mm512_loadu_ps(A + i), abs_mask));
    }
    float local = _mm512_reduce_max_ps(v_max);
    for (; i < MK; ++i) {
      const float v = std::fabs(A[i]);
      if (v > local) local = v;
    }
    return local;
  };
  auto run_serial_quantize = [&](float inv_act_scale) {
    const __m512 scale_v = _mm512_set1_ps(inv_act_scale);
    const __m512i hi127 = _mm512_set1_epi32(127);
    const __m512i lo127 = _mm512_set1_epi32(-127);
    const __m512i shift128 = _mm512_set1_epi32(128);
    std::size_t i = 0;
    for (; i + 16 <= MK; i += 16) {
      __m512 scaled = _mm512_mul_ps(_mm512_loadu_ps(A + i), scale_v);
      __m512i q = _mm512_cvtps_epi32(scaled);
      q = _mm512_min_epi32(_mm512_max_epi32(q, lo127), hi127);
      q = _mm512_add_epi32(q, shift128);
      __m128i u8 = _mm512_cvtepi32_epi8(q);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(a_u8 + i), u8);
    }
    for (; i < MK; ++i) {
      int q = static_cast<int>(std::lround(A[i] * inv_act_scale));
      if (q < -127) q = -127;
      if (q > 127) q = 127;
      a_u8[i] = static_cast<std::uint8_t>(q + 128);
    }
  };

  if (MK < kParallelThreshold || esm::InGlobalPoolWorker()) {
    const float a_max = run_serial_absmax();
    const float act_scale = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
    *act_scale_out = act_scale;
    run_serial_quantize(1.0f / act_scale);
    return;
  }

  // Parallel absmax. Chunk by 16-element blocks so each chunk's inner
  // loop is a clean 16-wide vector loop with at most one masked-tail
  // block at the very end of the array.
  const int n_blocks = static_cast<int>((MK + 15) / 16);
  std::atomic<std::uint32_t> shared_bits{0};
  esm::GlobalPool().parallel_for(
      0, n_blocks, /*grain=*/256, [&](int block_begin, int block_end) {
        __m512 v_max = _mm512_setzero_ps();
        const __m512 abs_mask =
            _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
        for (int b = block_begin; b < block_end; ++b) {
          const std::size_t i = static_cast<std::size_t>(b) * 16;
          if (i + 16 <= MK) {
            __m512 v = _mm512_and_ps(_mm512_loadu_ps(A + i), abs_mask);
            v_max = _mm512_max_ps(v_max, v);
          } else {
            const int tail = static_cast<int>(MK - i);
            const __mmask16 mask = static_cast<__mmask16>((1u << tail) - 1u);
            __m512 v = _mm512_maskz_loadu_ps(mask, A + i);
            v = _mm512_and_ps(v, abs_mask);
            v_max = _mm512_max_ps(v_max, v);
          }
        }
        const float local = _mm512_reduce_max_ps(v_max);
        std::uint32_t local_bits;
        std::memcpy(&local_bits, &local, sizeof(float));
        std::uint32_t prev = shared_bits.load(std::memory_order_relaxed);
        while (local_bits > prev &&
               !shared_bits.compare_exchange_weak(
                   prev, local_bits, std::memory_order_relaxed,
                   std::memory_order_relaxed)) {}
      });
  const std::uint32_t a_max_bits =
      shared_bits.load(std::memory_order_relaxed);
  float a_max;
  std::memcpy(&a_max, &a_max_bits, sizeof(float));
  const float act_scale = (a_max > 0.0f) ? (a_max / 127.0f) : 1.0f;
  const float inv_act_scale = 1.0f / act_scale;
  *act_scale_out = act_scale;

  // Parallel quantize. Same 16-block chunking; the masked-store path
  // covers MK not divisible by 16.
  esm::GlobalPool().parallel_for(
      0, n_blocks, /*grain=*/256, [&](int block_begin, int block_end) {
        const __m512 scale_v = _mm512_set1_ps(inv_act_scale);
        const __m512i hi127 = _mm512_set1_epi32(127);
        const __m512i lo127 = _mm512_set1_epi32(-127);
        const __m512i shift128 = _mm512_set1_epi32(128);
        for (int b = block_begin; b < block_end; ++b) {
          const std::size_t i = static_cast<std::size_t>(b) * 16;
          if (i + 16 <= MK) {
            __m512 scaled = _mm512_mul_ps(_mm512_loadu_ps(A + i), scale_v);
            __m512i q = _mm512_cvtps_epi32(scaled);
            q = _mm512_min_epi32(_mm512_max_epi32(q, lo127), hi127);
            q = _mm512_add_epi32(q, shift128);
            __m128i u8 = _mm512_cvtepi32_epi8(q);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(a_u8 + i), u8);
          } else {
            const int tail = static_cast<int>(MK - i);
            for (int kk = 0; kk < tail; ++kk) {
              int q = static_cast<int>(std::lround(A[i + kk] * inv_act_scale));
              if (q < -127) q = -127;
              if (q > 127) q = 127;
              a_u8[i + kk] = static_cast<std::uint8_t>(q + 128);
            }
          }
        }
      });
}

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

  // Steps 1 + 2: activation absmax + quantize, parallelized across MK.
  const std::size_t MK = static_cast<std::size_t>(M) *
                          static_cast<std::size_t>(K);
  if (g_a_u8.size() < MK) g_a_u8.resize(MK);
  std::uint8_t* a_u8 = g_a_u8.data();
  float act_scale = 1.0f;
  QuantizeActPrefixAvx512(A, MK, a_u8, &act_scale);

  // Steps 3 + 4: VPDPBUSD-tiled weight layout and zero-point col_sum live
  // on the QuantizedTensor itself (populated by BuildVnniCache at load/
  // quantize time; idempotent and weights-only). The kernel just reads
  // them — no per-call repacking or reductions.
  const int K_pad = (K + 3) & ~3;
  const std::int8_t* w_packed = W.packed_vnni.data();
  const std::int32_t* col_sum = W.col_sum.data();

  // Step 5: GEMM through the Goto 6×32 microkernel (fills the VPDPBUSD
  // pipe with 12 independent accumulators). ComputeRowsGoto falls back
  // to the legacy 1×16 path internally for the M-tail (M % 6) and the
  // N-tail (N % 32) — see lines 342 + 363.
  const float* w_scale = W.per_channel_scales.data();
  auto run = [&](int begin, int end) {
    ComputeRowsGoto(a_u8, w_packed, w_scale, col_sum, bias, act_scale, K,
                    K_pad, N, C, begin, end);
  };
  if (M > 1 && !esm::InGlobalPoolWorker()) {
    // Grain of 6 keeps full 6-row blocks together in each worker's chunk;
    // M-tail (M % 6) lands in exactly one worker's chunk via the legacy
    // fallback inside ComputeRowsGoto.
    esm::GlobalPool().parallel_for(0, M, /*grain=*/6, run);
  } else {
    run(0, M);
  }
}

#endif  // ESM_KERNEL_AVX512

}  // namespace esm::kernels
