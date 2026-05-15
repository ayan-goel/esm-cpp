#include "esm_cpp/kernels.h"

#ifdef ESM_KERNEL_NEON
#include <Accelerate/Accelerate.h>
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

// Dev fallback: wrap Apple Accelerate's cblas_sgemm. NEON is documented
// as a dev-iteration backend only (CLAUDE.md); the canonical Phase 1
// SIMD path is AVX-512+VNNI on x86. We're not staffing a hand-tuned
// NEON microkernel here.
//
// Our Linear contract: C[m, n] = sum_k A[m, k] * W[n, k] + bias[n]
//   A: row-major [M, K], W: row-major [N, K] (PyTorch out_features x in_features).
// In cblas terms that's C = A * W^T (NoTrans for A, Trans for W).
void LinearNeon(const float* A, const float* W, const float* bias, float* C,
                int M, int N, int K) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, W,
              K, 0.0f, C, N);
  if (bias) {
    for (int m = 0; m < M; ++m) {
      float* C_row = C + static_cast<long>(m) * N;
      for (int n = 0; n < N; ++n) C_row[n] += bias[n];
    }
  }
}

#endif  // ESM_KERNEL_NEON

#ifdef ESM_KERNEL_AVX512

// STUB — the canonical Phase 1 SIMD path. Slice 3.1-3.3 hand-off (needs
// an x86 AVX-512+VNNI instance to validate; see notes/phase1-handoff.md
// for the rationale of why this isn't written here).
//
// Target design (decision locked in tasks/plan.md §S3):
//
//   Microkernel: 16 rows x 32 cols of FP32 C accumulated in 16 zmm
//   registers. Each K-step issues two vfmadd231ps (one per output-col
//   half) against a broadcast-loaded A column. K_C ~ 512, M_C ~ 256,
//   N_C ~ 4096 with the standard 6-loop Goto packing nest:
//     for jc in [0, N) step N_C:         pack B panel [K, N_C] -> Bp
//       for kc in [0, K) step K_C:        pack A panel [M, K_C] -> Ap
//         for ic in [0, M) step M_C:
//           for jr in [0, N_C) step 32:
//             for ir in [0, M_C) step 16:
//               kernel_16x32(Ap, Bp, &C[ic+ir, jc+jr], K_C, ldc, beta)
//   Bias is folded into the kernel's beta=0 init (load bias[jc..jc+32]
//   into the 16 zmm accs at K=0). Tail M%16 / N%32 / K%K_C handled by
//   masked-load microkernels (use _mm512_mask_loadu_ps with the rem mask).
//
// References worth re-reading at implementation time:
//   - salykova.github.io/matmul-cpu — the clearest walkthrough we know of
//   - yzhaiustc/Optimizing-DGEMM-on-Intel-CPUs-with-AVX512F — production
//   - BLIS docs/Performance.md — confirms the >90% MKL target
//
// libxsmm hookup (Slice 3.6): libxsmm_smmdispatch on the four critical
// (M, N, K) tuples at Model::load time and cache the function pointers
// on the Model. Use libxsmm for shapes where M<32 or N<32 (lm_head's
// [L, 33], future small-batch service shapes). The hand-written
// microkernel above is the primary path for the four production shapes;
// libxsmm covers the small-shape tail.
//
// Verification ordering when this lands:
//   1) tests/cpp/test_gemm_shapes.cpp::Avx512DispatchMatchesRef
//      — add an #if defined(__x86_64__) sibling to the NEON test; same
//      shape sweep, same 1e-4 relative tolerance.
//   2) bench_gemm in Release on the gate machine; compare against MKL's
//      cblas_sgemm via a wrapper script (tasks/plan.md S6.2).
//   3) Re-run HF parity; layer envelope should match or improve on the
//      NEON-Accelerate envelope captured in commit 5766242.
void LinearRef(const float* A, const float* W, const float* bias, float* C,
               int M, int N, int K);
void LinearAvx512(const float* A, const float* W, const float* bias, float* C,
                  int M, int N, int K) {
  LinearRef(A, W, bias, C, M, N, K);
}

#endif  // ESM_KERNEL_AVX512

}  // namespace esm::kernels
