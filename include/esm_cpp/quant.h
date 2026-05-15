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
};

// Quantize an FP32 row-major [N, K] weight tensor into a per-channel
// symmetric INT8 QuantizedTensor. An all-zero row gets scale=0 and
// packed=0 (the dequant matmul handles this correctly).
void Quantize(const float* W_fp32, int N, int K, QuantizedTensor* out);

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

}  // namespace esm::quant
