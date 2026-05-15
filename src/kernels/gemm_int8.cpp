#include "esm_cpp/kernels.h"
#include "esm_cpp/quant.h"

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

// STUB — AVX-512 VNNI INT8 microkernel hand-off. Slice 6 of Phase 2's
// production INT8 path. Until the intrinsics body lands on the x86
// gate machine, dispatching to VNNI just routes through LinearInt8Ref.
//
// Target design (locked alongside the FP32 Goto microkernel; see
// gemm_fp32.cpp's AVX-512 stub for the matching FP32 register block):
//
//   Microkernel: 16x16 register block of s32 C accumulators (8 zmm
//   registers — leaves the other 24 zmm for VPDPBUSD operands plus
//   prefetch). Each K-step loads 4 INT8 values per channel (4-wide
//   u8 input chunk via _mm512_set1_epi32 broadcast), 4 INT8 weight
//   columns via _mm512_loadu_si512 (64 byte block), and issues one
//   VPDPBUSD per row of C accumulators (16 total per K-step).
//
//   Macrokernel: same Goto packing nest as FP32, K_C ~ 512 (so K_C * 4
//   per dispatch sits in L2), M_C ~ 256. Pack u8 input into K_C / 4
//   panels of 4-byte rows; pack s8 weights into K_C / 4 transposed
//   panels of 64-byte columns (4 channels x 16 columns each).
//
//   Scale folding: at C-write-out, multiply the s32 accumulator by
//   per-output-channel weight_scale[n] x per-tensor act_scale and
//   add FP32 bias[n]. One vfmadd231ps per output column at write-out.
//
// Tail handling: M % 16 / N % 16 via masked stores; K % (K_C * 4)
// via leftover loop with VPDPBUSD on partial loads.
//
// AMX path (Slice 6.2): TDPBSSD with 16-row tiles, K=64 INT8 chunks
// per dispatch. Requires syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
// XFEATURE_XTILEDATA) at first use — guard behind std::call_once.
// Tile config (ldtilecfg) sets TILES_DATA to 16 rows x 64 INT8 bytes
// for inputs and 16 rows x 64 bytes (= 16 s32 cols) for output. AMX
// gives 4-8x VNNI on Sapphire Rapids + later.
void LinearInt8Ref(const float* A, const esm::quant::QuantizedTensor& W,
                   const float* bias, float* C, int M, int N, int K);
void LinearVnni(const float* A, const esm::quant::QuantizedTensor& W,
                const float* bias, float* C, int M, int N, int K) {
  LinearInt8Ref(A, W, bias, C, M, N, K);
}

#endif  // ESM_KERNEL_AVX512

}  // namespace esm::kernels
