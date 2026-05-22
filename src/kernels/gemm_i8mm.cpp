#include "esm_cpp/kernels.h"
#include "esm_cpp/quant.h"

// NEON i8mm (SMMLA) W8A8 GEMM — the ARM analog of the x86 AMX tier. Compiled
// into its own TU with `+i8mm` so the SMMLA intrinsic is isolated from the
// ARMv8.0 baseline NEON TU; runtime-gated (NeonI8mm host only), falling back
// to the SDOT kernel on hosts without FEAT_I8MM (handled by dispatch, which
// routes a NeonDotProd host to LinearNeonDotProd instead).
//
// vmmlaq_s32(acc, a, b): a and b each pack a 2x8 int8 matrix (row0 = bytes
// [0,8), row1 = bytes [8,16)); the result is the 2x2 int32 product
//   [a_row0.b_row0, a_row0.b_row1, a_row1.b_row0, a_row1.b_row1].
// So one instruction produces a 2-row x 2-col output tile, accumulating 8 K.
#ifdef ESM_KERNEL_I8MM
#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "esm_cpp/thread_pool.h"
#endif

namespace esm::kernels {

#ifdef ESM_KERNEL_I8MM

// Shared NEON activation prefix (absmax + symmetric s8 quantize), defined in
// the baseline NEON TU (gemm_int8.cpp) and reused here.
void QuantizeActPrefixNeon(const float* A, std::size_t MK, std::int8_t* a_s8,
                           float* act_scale_out);

namespace {

thread_local std::vector<std::int8_t> g_a_s8_i8mm;

// Load a 2x8 activation row-pair (rows m, m+1) for k-block kb into an
// int8x16 [row0:k0..k7, row1:k0..k7], zero-padding the K-tail and a missing
// second row. The fast (full 8 K, both rows) case avoids the temp copy.
inline int8x16_t LoadArowPair(const std::int8_t* a_s8, int m, int rp, int kb,
                              int K) {
  if (rp == 2 && kb + 8 <= K) {
    int8x8_t r0 = vld1_s8(a_s8 + static_cast<long>(m) * K + kb);
    int8x8_t r1 = vld1_s8(a_s8 + static_cast<long>(m + 1) * K + kb);
    return vcombine_s8(r0, r1);
  }
  std::int8_t tmp[16] = {0};
  const int k_step = std::min(8, K - kb);
  for (int kk = 0; kk < k_step; ++kk)
    tmp[kk] = a_s8[static_cast<long>(m) * K + kb + kk];
  if (rp == 2) {
    for (int kk = 0; kk < k_step; ++kk)
      tmp[8 + kk] = a_s8[static_cast<long>(m + 1) * K + kb + kk];
  }
  return vld1q_s8(tmp);
}

inline float Finalize(float acc_lane, float act_scale, float w_scale,
                      const float* bias, int n) {
  return acc_lane * act_scale * w_scale + (bias ? bias[n] : 0.0f);
}

// Hot path: 4 rows x 4 cols (2 row-pairs x 2 col-pairs = 4 SMMLA tiles).
// Caller guarantees m+4 <= M and n+4 <= N.
inline void Kernel4x4(const std::int8_t* a_s8, const std::int8_t* packed,
                      const float* w_scale, const float* bias, float act_scale,
                      int N, int K, int K_pad8, float* C, int m, int n) {
  const std::int8_t* pair0 = packed + static_cast<long>(n) * K_pad8;
  const std::int8_t* pair1 = packed + static_cast<long>(n + 2) * K_pad8;
  int32x4_t acc00 = vdupq_n_s32(0), acc01 = vdupq_n_s32(0);
  int32x4_t acc10 = vdupq_n_s32(0), acc11 = vdupq_n_s32(0);
  for (int kb = 0; kb < K; kb += 8) {
    int8x16_t av0 = LoadArowPair(a_s8, m, 2, kb, K);
    int8x16_t av1 = LoadArowPair(a_s8, m + 2, 2, kb, K);
    int8x16_t wv0 = vld1q_s8(pair0 + kb * 2);
    int8x16_t wv1 = vld1q_s8(pair1 + kb * 2);
    acc00 = vmmlaq_s32(acc00, av0, wv0);
    acc01 = vmmlaq_s32(acc01, av0, wv1);
    acc10 = vmmlaq_s32(acc10, av1, wv0);
    acc11 = vmmlaq_s32(acc11, av1, wv1);
  }
  float l[4];
  const struct {
    int32x4_t acc;
    int row, col;
  } tiles[4] = {{acc00, m, n}, {acc01, m, n + 2}, {acc10, m + 2, n},
                {acc11, m + 2, n + 2}};
  for (const auto& t : tiles) {
    vst1q_f32(l, vcvtq_f32_s32(t.acc));
    C[static_cast<long>(t.row) * N + t.col] =
        Finalize(l[0], act_scale, w_scale[t.col], bias, t.col);
    C[static_cast<long>(t.row) * N + t.col + 1] =
        Finalize(l[1], act_scale, w_scale[t.col + 1], bias, t.col + 1);
    C[static_cast<long>(t.row + 1) * N + t.col] =
        Finalize(l[2], act_scale, w_scale[t.col], bias, t.col);
    C[static_cast<long>(t.row + 1) * N + t.col + 1] =
        Finalize(l[3], act_scale, w_scale[t.col + 1], bias, t.col + 1);
  }
}

// Flexible mop-up: `rows` rows (1..4) x one col-pair of `ncols` cols (1..2)
// at column n. Used for the M-tail and N-tail (incl. the partial last pair).
inline void GeneralBlock(const std::int8_t* a_s8, const std::int8_t* packed,
                         const float* w_scale, const float* bias,
                         float act_scale, int N, int K, int K_pad8, float* C,
                         int m, int rows, int n, int ncols) {
  const std::int8_t* pair = packed + static_cast<long>(n) * K_pad8;
  for (int r = 0; r < rows; r += 2) {
    const int rp = std::min(2, rows - r);
    int32x4_t acc = vdupq_n_s32(0);
    for (int kb = 0; kb < K; kb += 8) {
      int8x16_t av = LoadArowPair(a_s8, m + r, rp, kb, K);
      int8x16_t wv = vld1q_s8(pair + kb * 2);
      acc = vmmlaq_s32(acc, av, wv);
    }
    float l[4];
    vst1q_f32(l, vcvtq_f32_s32(acc));
    C[static_cast<long>(m + r) * N + n] =
        Finalize(l[0], act_scale, w_scale[n], bias, n);
    if (ncols > 1)
      C[static_cast<long>(m + r) * N + n + 1] =
          Finalize(l[1], act_scale, w_scale[n + 1], bias, n + 1);
    if (rp > 1) {
      C[static_cast<long>(m + r + 1) * N + n] =
          Finalize(l[2], act_scale, w_scale[n], bias, n);
      if (ncols > 1)
        C[static_cast<long>(m + r + 1) * N + n + 1] =
            Finalize(l[3], act_scale, w_scale[n + 1], bias, n + 1);
    }
  }
}

void ComputeRowsI8mm(const std::int8_t* a_s8, const std::int8_t* packed,
                     const float* w_scale, const float* bias, float act_scale,
                     int N, int K, int K_pad8, float* C, int m_begin,
                     int m_end) {
  int m = m_begin;
  for (; m + 4 <= m_end; m += 4) {
    int n = 0;
    for (; n + 4 <= N; n += 4)
      Kernel4x4(a_s8, packed, w_scale, bias, act_scale, N, K, K_pad8, C, m, n);
    for (; n < N; n += 2)
      GeneralBlock(a_s8, packed, w_scale, bias, act_scale, N, K, K_pad8, C, m, 4,
                   n, std::min(2, N - n));
  }
  if (m < m_end) {
    const int rows = m_end - m;
    for (int n = 0; n < N; n += 2)
      GeneralBlock(a_s8, packed, w_scale, bias, act_scale, N, K, K_pad8, C, m,
                   rows, n, std::min(2, N - n));
  }
}

}  // namespace

void LinearNeonI8mm(const float* A, const esm::quant::QuantizedTensor& W,
                    const float* bias, float* C, int M, int N, int K) {
  if (K <= 0 || M <= 0 || N <= 0) {
    if (M > 0 && N > 0 && C) {
      std::memset(C, 0,
                  static_cast<std::size_t>(M) * static_cast<std::size_t>(N) *
                      sizeof(float));
    }
    return;
  }
  const std::size_t MK =
      static_cast<std::size_t>(M) * static_cast<std::size_t>(K);
  if (g_a_s8_i8mm.size() < MK) g_a_s8_i8mm.resize(MK);
  std::int8_t* a_s8 = g_a_s8_i8mm.data();
  float act_scale = 1.0f;
  QuantizeActPrefixNeon(A, MK, a_s8, &act_scale);

  const int K_pad8 = (K + 7) & ~7;
  const std::int8_t* packed = W.packed_arm_i8mm.data();
  const float* w_scale = W.per_channel_scales.data();
  auto run = [&](int begin, int end) {
    ComputeRowsI8mm(a_s8, packed, w_scale, bias, act_scale, N, K, K_pad8, C,
                    begin, end);
  };
  if (M > 4 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(0, M, /*grain=*/4, run);
  } else {
    run(0, M);
  }
}

#endif  // ESM_KERNEL_I8MM

}  // namespace esm::kernels
