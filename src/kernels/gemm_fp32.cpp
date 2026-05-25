#include "esm_cpp/kernels.h"

#include <cstddef>

#ifdef ESM_KERNEL_NEON
#include <arm_neon.h>

#include "esm_cpp/thread_pool.h"

// On Apple we can route plain FP32 GEMM through Accelerate's cblas (which
// uses the AMX coprocessor under the hood). This is independent of the
// BNNSGraph path in apple_amx.cpp — even SDKs without <Accelerate/BNNSGraph.h>
// (older Xcode on the macos-14 GHA runner) still have cblas. Gate on
// __APPLE__ rather than ESM_APPLE_AMX_AVAILABLE so the cblas path
// survives the BNNSGraph SDK check in CMakeLists.txt.
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>

#include "esm_cpp/cpu_features.h"
#endif
#endif

#ifdef ESM_KERNEL_AVX512
#include <immintrin.h>

#include "esm_cpp/thread_pool.h"
#endif

namespace esm::kernels {

#ifdef ESM_KERNEL_REFERENCE

void LinearRef(const float* A, const float* W, const float* bias, float* C,
               int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = bias ? bias[n] : 0.0f;
      const float* a_row = A + static_cast<long>(m) * K;
      const float* w_row = W + static_cast<long>(n) * K;
      for (int k = 0; k < K; ++k) {
        acc += a_row[k] * w_row[k];
      }
      C[static_cast<long>(m) * N + n] = acc;
    }
  }
}

#endif  // ESM_KERNEL_REFERENCE

#ifdef ESM_KERNEL_NEON

// Hand-written NEON FMLA SGEMM. Linear contract:
//   C[m, n] = sum_k A[m, k] * W[n, k] + bias[n]
//   A: row-major [M, K], W: row-major [N, K] (PyTorch out_features x in_features).
// W rows are contiguous, so the computation is M·N independent length-K dot
// products (C = A·Wᵀ). Apple Accelerate (which taps the AMX coprocessor) is
// available behind ESM_ARM_USE_ACCELERATE as a dev backend, but the default
// path is this microkernel so the engine also runs fast on Linux ARM/Graviton
// where Accelerate does not exist.

namespace {

// 4-wide-accumulated dot product, used for the M/N tail rows/cols.
inline float DotNeon(const float* a, const float* w, int K) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  int k = 0;
  for (; k + 4 <= K; k += 4) {
    acc = vfmaq_f32(acc, vld1q_f32(a + k), vld1q_f32(w + k));
  }
  float s = vaddvq_f32(acc);
  for (; k < K; ++k) s += a[k] * w[k];
  return s;
}

// 4 rows × 4 cols register-blocked microkernel: 16 float32x4_t accumulators
// hold the K-lane partial sums for C[m..m+3][n..n+3]; horizontally reduced at
// the end. 16 acc + 8 load registers fit the 32 AArch64 vector registers.
inline void Micro4x4(const float* A, const float* W, const float* bias,
                     float* C, int N, int K, int m, int n) {
  const float* a[4] = {A + static_cast<long>(m + 0) * K,
                       A + static_cast<long>(m + 1) * K,
                       A + static_cast<long>(m + 2) * K,
                       A + static_cast<long>(m + 3) * K};
  const float* w[4] = {W + static_cast<long>(n + 0) * K,
                       W + static_cast<long>(n + 1) * K,
                       W + static_cast<long>(n + 2) * K,
                       W + static_cast<long>(n + 3) * K};
  float32x4_t acc[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) acc[i][j] = vdupq_n_f32(0.0f);
  int k = 0;
  for (; k + 4 <= K; k += 4) {
    float32x4_t av[4], wv[4];
    for (int i = 0; i < 4; ++i) av[i] = vld1q_f32(a[i] + k);
    for (int j = 0; j < 4; ++j) wv[j] = vld1q_f32(w[j] + k);
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) acc[i][j] = vfmaq_f32(acc[i][j], av[i], wv[j]);
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      float s = vaddvq_f32(acc[i][j]);
      for (int kk = k; kk < K; ++kk) s += a[i][kk] * w[j][kk];
      C[static_cast<long>(m + i) * N + (n + j)] = s + (bias ? bias[n + j] : 0.0f);
    }
  }
}

// Compute output rows [m_begin, m_end). 4×4 blocks where they fit; scalar
// (4-wide) dot products for the N-tail (N % 4) and the M-tail (< 4 rows).
void ComputeRowsNeon(const float* A, const float* W, const float* bias,
                     float* C, int N, int K, int m_begin, int m_end) {
  int m = m_begin;
  for (; m + 4 <= m_end; m += 4) {
    int n = 0;
    for (; n + 4 <= N; n += 4) Micro4x4(A, W, bias, C, N, K, m, n);
    for (; n < N; ++n) {
      for (int i = 0; i < 4; ++i) {
        float s = DotNeon(A + static_cast<long>(m + i) * K,
                          W + static_cast<long>(n) * K, K);
        C[static_cast<long>(m + i) * N + n] = s + (bias ? bias[n] : 0.0f);
      }
    }
  }
  for (; m < m_end; ++m) {
    for (int n = 0; n < N; ++n) {
      float s = DotNeon(A + static_cast<long>(m) * K,
                        W + static_cast<long>(n) * K, K);
      C[static_cast<long>(m) * N + n] = s + (bias ? bias[n] : 0.0f);
    }
  }
}

}  // namespace

void LinearNeon(const float* A, const float* W, const float* bias, float* C,
                int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) return;
#ifdef __APPLE__
  // Apple-only Accelerate cblas FP32 GEMM — ~3.75-6.2x the hand-written
  // NEON FMLA on M-series. Opt-in via ESM_APPLE_AMX flag — same env knob
  // used by the BNNSGraph fp16 path; semantically "use the Accelerate
  // implementation when on Apple."
  if (esm::ArmUseAppleAmx()) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, W,
                K, 0.0f, C, N);
    if (bias) {
      for (int m = 0; m < M; ++m) {
        float* C_row = C + static_cast<long>(m) * N;
        for (int n = 0; n < N; ++n) C_row[n] += bias[n];
      }
    }
    return;
  }
#endif
  // Default portable path. Parallelize across M-row-blocks; grain of 4 keeps
  // full 4-row microkernel blocks together; the InGlobalPoolWorker guard
  // prevents a nested parallel_for deadlock when called from inside a task.
  auto run = [&](int begin, int end) {
    ComputeRowsNeon(A, W, bias, C, N, K, begin, end);
  };
  constexpr long kParallelThreshold = 64L * 1024L;
  if (M > 4 && !esm::InGlobalPoolWorker() &&
      static_cast<long>(M) * N * K >= kParallelThreshold) {
    esm::GlobalPool().parallel_for(0, M, /*grain=*/4, run);
  } else {
    run(0, M);
  }
}

#endif  // ESM_KERNEL_NEON

#ifdef ESM_KERNEL_AVX512

// FP32 AVX-512 GEMM. C[m, n] = sum_k A[m, k] * W[n, k] + bias[n].
// W is row-major [N, K] — PyTorch's nn.Linear layout (out_features ×
// in_features). Both A and W rows are contiguous along K, so we
// accumulate the dot-product across K inside zmm registers and
// reduce-add at the end. Eight independent accumulators per (m, N-block)
// give the FMA pipeline enough parallelism to stay busy (SPR can issue
// 2 FMAs/cycle on ports 0+5, and the dependency chain on a single zmm
// would otherwise bottleneck at the 4-cycle FMA latency).
//
// Parallelism: across M-rows. lm_head's two FP32 GEMMs (lm_dense at
// [2048, d, d] and lm_decoder at [2048, V, d]) dominated the 650M
// forward at 53 % + 1 % of wall time before this kernel landed — the
// previous LinearAvx512 was a stub that delegated to LinearRef, so
// every FP32 GEMM ran scalar single-threaded.
namespace {

inline __m512 MaskedLoad(const float* p, std::size_t tail) {
  const __mmask16 mask = static_cast<__mmask16>((1u << tail) - 1u);
  return _mm512_maskz_loadu_ps(mask, p);
}

inline void Dot1Row8N(const float* a_row, const float* W, int K, int N_stride,
                       int n_base, const float* bias, float* c_row) {
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps();
  __m512 acc3 = _mm512_setzero_ps();
  __m512 acc4 = _mm512_setzero_ps();
  __m512 acc5 = _mm512_setzero_ps();
  __m512 acc6 = _mm512_setzero_ps();
  __m512 acc7 = _mm512_setzero_ps();
  const float* w0 = W + static_cast<long>(n_base + 0) * N_stride;
  const float* w1 = W + static_cast<long>(n_base + 1) * N_stride;
  const float* w2 = W + static_cast<long>(n_base + 2) * N_stride;
  const float* w3 = W + static_cast<long>(n_base + 3) * N_stride;
  const float* w4 = W + static_cast<long>(n_base + 4) * N_stride;
  const float* w5 = W + static_cast<long>(n_base + 5) * N_stride;
  const float* w6 = W + static_cast<long>(n_base + 6) * N_stride;
  const float* w7 = W + static_cast<long>(n_base + 7) * N_stride;
  int k = 0;
  for (; k + 16 <= K; k += 16) {
    __m512 a = _mm512_loadu_ps(a_row + k);
    acc0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(w0 + k), acc0);
    acc1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(w1 + k), acc1);
    acc2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(w2 + k), acc2);
    acc3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(w3 + k), acc3);
    acc4 = _mm512_fmadd_ps(a, _mm512_loadu_ps(w4 + k), acc4);
    acc5 = _mm512_fmadd_ps(a, _mm512_loadu_ps(w5 + k), acc5);
    acc6 = _mm512_fmadd_ps(a, _mm512_loadu_ps(w6 + k), acc6);
    acc7 = _mm512_fmadd_ps(a, _mm512_loadu_ps(w7 + k), acc7);
  }
  if (k < K) {
    const std::size_t tail = static_cast<std::size_t>(K - k);
    __m512 a = MaskedLoad(a_row + k, tail);
    acc0 = _mm512_fmadd_ps(a, MaskedLoad(w0 + k, tail), acc0);
    acc1 = _mm512_fmadd_ps(a, MaskedLoad(w1 + k, tail), acc1);
    acc2 = _mm512_fmadd_ps(a, MaskedLoad(w2 + k, tail), acc2);
    acc3 = _mm512_fmadd_ps(a, MaskedLoad(w3 + k, tail), acc3);
    acc4 = _mm512_fmadd_ps(a, MaskedLoad(w4 + k, tail), acc4);
    acc5 = _mm512_fmadd_ps(a, MaskedLoad(w5 + k, tail), acc5);
    acc6 = _mm512_fmadd_ps(a, MaskedLoad(w6 + k, tail), acc6);
    acc7 = _mm512_fmadd_ps(a, MaskedLoad(w7 + k, tail), acc7);
  }
  const float b0 = bias ? bias[n_base + 0] : 0.0f;
  const float b1 = bias ? bias[n_base + 1] : 0.0f;
  const float b2 = bias ? bias[n_base + 2] : 0.0f;
  const float b3 = bias ? bias[n_base + 3] : 0.0f;
  const float b4 = bias ? bias[n_base + 4] : 0.0f;
  const float b5 = bias ? bias[n_base + 5] : 0.0f;
  const float b6 = bias ? bias[n_base + 6] : 0.0f;
  const float b7 = bias ? bias[n_base + 7] : 0.0f;
  c_row[n_base + 0] = b0 + _mm512_reduce_add_ps(acc0);
  c_row[n_base + 1] = b1 + _mm512_reduce_add_ps(acc1);
  c_row[n_base + 2] = b2 + _mm512_reduce_add_ps(acc2);
  c_row[n_base + 3] = b3 + _mm512_reduce_add_ps(acc3);
  c_row[n_base + 4] = b4 + _mm512_reduce_add_ps(acc4);
  c_row[n_base + 5] = b5 + _mm512_reduce_add_ps(acc5);
  c_row[n_base + 6] = b6 + _mm512_reduce_add_ps(acc6);
  c_row[n_base + 7] = b7 + _mm512_reduce_add_ps(acc7);
}

inline float Dot1Row1N(const float* a_row, const float* w_row, int K) {
  __m512 acc = _mm512_setzero_ps();
  int k = 0;
  for (; k + 16 <= K; k += 16) {
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a_row + k),
                           _mm512_loadu_ps(w_row + k), acc);
  }
  if (k < K) {
    const std::size_t tail = static_cast<std::size_t>(K - k);
    acc = _mm512_fmadd_ps(MaskedLoad(a_row + k, tail),
                           MaskedLoad(w_row + k, tail), acc);
  }
  return _mm512_reduce_add_ps(acc);
}

}  // namespace

void LinearAvx512(const float* A, const float* W, const float* bias, float* C,
                  int M, int N, int K) {
  auto work_rows = [&](int m_begin, int m_end) {
    constexpr int kNR = 8;
    for (int m = m_begin; m < m_end; ++m) {
      const float* a_row = A + static_cast<long>(m) * K;
      float* c_row = C + static_cast<long>(m) * N;
      int n = 0;
      for (; n + kNR <= N; n += kNR) {
        Dot1Row8N(a_row, W, K, K, n, bias, c_row);
      }
      for (; n < N; ++n) {
        const float* w_row = W + static_cast<long>(n) * K;
        c_row[n] = (bias ? bias[n] : 0.0f) + Dot1Row1N(a_row, w_row, K);
      }
    }
  };
  if (M > 1 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(0, M, /*grain=*/1, work_rows);
  } else {
    work_rows(0, M);
  }
}

#endif  // ESM_KERNEL_AVX512

}  // namespace esm::kernels
