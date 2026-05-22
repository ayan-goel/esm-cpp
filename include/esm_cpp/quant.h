#pragma once

#include <cstdint>
#include <vector>

namespace esm::quant {

// Per-channel symmetric INT8 weight tensor.
//   packed[n, k]                ∈ [-127, 127]   (no -128; symmetric range)
//   per_channel_scales[n]       = max(|W[n, :]|) / 127    (>= 0)
//   weight_fp32[n, k]           ≈ packed[n, k] * per_channel_scales[n]
//
// Bias is stored separately as FP32 (not part of the QuantizedTensor).
// "Output" axis (N) is the channel axis — matches PyTorch's Linear
// out_features convention and is the natural axis for VPDPBUSD's
// per-output-channel scale folding.
struct QuantizedTensor {
  // Row-major [N, K] int8 weights.
  std::vector<std::int8_t> packed;
  // [N] FP32 per-output-channel scales.
  std::vector<float> per_channel_scales;
  int N = 0;
  int K = 0;

  // Derived data for the AVX-512 VPDPBUSD path. Populated by BuildVnniCache
  // (called from Quantize and from the GGUF Q8_ESM loader); LinearVnni
  // reads these instead of recomputing them per call.
  //   packed_vnni  = `packed` repacked into 64-byte VPDPBUSD tiles. For each
  //                  16-N panel and each 4-K tile, layout is
  //                  [N0:k0,k1,k2,k3, N1:k0,...,N15:k0,k1,k2,k3]. K_pad rounds
  //                  K up to a multiple of 4; the K-tail bytes (if any) are
  //                  zero. N-tail (< 16) tiles are zero-padded too.
  //   col_sum[n]   = sum_k packed[n, k], int32, for zero-point correction.
  std::vector<std::int8_t> packed_vnni;
  std::vector<std::int32_t> col_sum;
};

// Quantize an FP32 row-major [N, K] weight tensor into a per-channel
// symmetric INT8 QuantizedTensor. An all-zero row gets scale=0 and
// packed=0 (the dequant matmul handles this correctly). Also populates
// the VPDPBUSD-tiled `packed_vnni` and `col_sum` derived caches via
// BuildVnniCache so the hot kernel path can read them directly.
void Quantize(const float* W_fp32, int N, int K, QuantizedTensor* out);

// Populate (or repopulate) the derived `packed_vnni` and `col_sum` fields
// from `out->packed` + `out->N` + `out->K`. Idempotent; called by Quantize
// and by the GGUF loader after it materializes the int8 block.
void BuildVnniCache(QuantizedTensor* out);

// Per-tensor symmetric activation quantizer (FP32 -> u8 with zero-point 128).
//   q[i] = clamp(round(x[i] / scale), -127, 127) + 128
// scale == 0 produces all-zero-point output. The activation scale is a
// single FP32 number per tensor, typically derived from a 99.9-percentile
// observer (see ActivationObserver) and SmoothQuant migration. The VNNI
// VPDPBUSD path consumes the u8 output; the s32 accumulator is later
// rescaled by (act_scale * weight_scale[n]) at C-write-out. Slice 6
// AVX-512 version is x86 hand-off.
void QuantizeActivationRef(const float* x, std::size_t n, float scale,
                            std::uint8_t* q);

// Symmetric per-tensor activation quantizer (FP32 -> s8, no zero-point).
//   q[i] = clamp(round(x[i] / scale), -127, 127)
// scale == 0 produces all-zero output. This is the ARM (NEON SDOT / i8mm)
// activation form: SDOT and SMMLA are signed x signed, so activations stay
// symmetric s8 and the x86 zero-point-128 + col_sum correction is unnecessary.
// The s32 accumulator is rescaled by (act_scale * weight_scale[n]) at
// C-write-out, same as the VNNI path minus the zero-point term.
void QuantizeActivationSymmetricRef(const float* x, std::size_t n, float scale,
                                    std::int8_t* q);

}  // namespace esm::quant
