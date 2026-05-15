// Phase 2 Slice 6: per-tensor symmetric activation quantizer (FP32 -> u8).
//
// For VPDPBUSD's u8 x s8 -> s32 inner product the activation has to land
// in [0, 255]. Symmetric per-tensor quant gives:
//   q[i] = clamp(round(x[i] / scale), -127, 127) + 128
//   reconstruct x[i] ~= (q[i] - 128) * scale
// Slice 6 ships the scalar reference; AVX-512 (single vfmadd + cvtps)
// version is x86 hand-off alongside the VNNI microkernel.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "esm_cpp/quant.h"

namespace {

std::vector<float> RandomVec(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

}  // namespace

TEST(ActivationQuantizeRef, RoundTripErrorBoundedByHalfScale) {
  const std::size_t N = 256;
  auto x = RandomVec(N, 0xa11c);
  float scale = 0.0f;
  for (auto v : x) scale = std::max(scale, std::fabs(v));
  scale /= 127.0f;
  std::vector<std::uint8_t> q(N);
  esm::quant::QuantizeActivationRef(x.data(), N, scale, q.data());
  for (std::size_t i = 0; i < N; ++i) {
    float reconstructed = (static_cast<int>(q[i]) - 128) * scale;
    EXPECT_LE(std::fabs(reconstructed - x[i]), scale * 0.51f) << "i=" << i;
  }
}

TEST(ActivationQuantizeRef, RangeClampedTo0_255) {
  // Inputs at extremes get clamped; q stays in [0, 255].
  std::vector<float> x = {1000.0f, -1000.0f, 0.0f, 0.01f, -0.01f};
  std::vector<std::uint8_t> q(x.size());
  const float scale = 0.1f;
  esm::quant::QuantizeActivationRef(x.data(), x.size(), scale, q.data());
  for (auto v : q) {
    // u8 is implicitly 0..255; verify clamp at the extremes specifically.
    EXPECT_GE(static_cast<int>(v), 0);
    EXPECT_LE(static_cast<int>(v), 255);
  }
  // Symmetric quant uses [-127, 127] (avoiding -128) so the bias-shifted
  // u8 range is [1, 255]. +1000 saturates at +127 -> 255; -1000 saturates
  // at -127 -> 1. 0 maps to the zero-point 128.
  EXPECT_EQ(static_cast<int>(q[0]), 255);
  EXPECT_EQ(static_cast<int>(q[1]), 1);
  EXPECT_EQ(static_cast<int>(q[2]), 128);
}

TEST(ActivationQuantizeRef, ZeroScaleProducesAllZeroPoint) {
  // Degenerate scale=0 case: every input quantizes to the zero point (128).
  std::vector<float> x = {1.0f, -1.0f, 0.0f};
  std::vector<std::uint8_t> q(x.size());
  esm::quant::QuantizeActivationRef(x.data(), x.size(), 0.0f, q.data());
  for (auto v : q) EXPECT_EQ(static_cast<int>(v), 128);
}
