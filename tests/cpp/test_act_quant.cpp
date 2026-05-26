// Phase 2 Slice 6: per-tensor symmetric activation quantizer (FP32 -> u8).
//
// For VPDPBUSD's u8 x s8 -> s32 inner product the activation has to land
// in [0, 255]. Symmetric per-tensor quant gives:
//   q[i] = clamp(round(x[i] / scale), -127, 127) + 128
//   reconstruct x[i] ~= (q[i] - 128) * scale
// Slice 6 ships the scalar reference; AVX-512 (single vfmadd + cvtps)
// version is x86 hand-off alongside the VNNI microkernel.

#include <gtest/gtest.h>

#include <algorithm>
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

// Symmetric s8 activation quant (the ARM SDOT/i8mm form: no zero-point).
TEST(ActivationQuantizeSymmetricRef, RoundTripErrorBoundedByHalfScale) {
  const std::size_t N = 256;
  auto x = RandomVec(N, 0xb22d);
  float scale = 0.0f;
  for (auto v : x) scale = std::max(scale, std::fabs(v));
  scale /= 127.0f;
  std::vector<std::int8_t> q(N);
  esm::quant::QuantizeActivationSymmetricRef(x.data(), N, scale, q.data());
  for (std::size_t i = 0; i < N; ++i) {
    float reconstructed = static_cast<int>(q[i]) * scale;
    EXPECT_LE(std::fabs(reconstructed - x[i]), scale * 0.51f) << "i=" << i;
  }
}

TEST(ActivationQuantizeSymmetricRef, RangeClampedToPlusMinus127) {
  std::vector<float> x = {1000.0f, -1000.0f, 0.0f};
  std::vector<std::int8_t> q(x.size());
  esm::quant::QuantizeActivationSymmetricRef(x.data(), x.size(), 0.1f, q.data());
  EXPECT_EQ(static_cast<int>(q[0]), 127);
  EXPECT_EQ(static_cast<int>(q[1]), -127);
  EXPECT_EQ(static_cast<int>(q[2]), 0);
}

TEST(ActivationQuantizeSymmetricRef, ZeroScaleProducesAllZero) {
  std::vector<float> x = {1.0f, -1.0f, 0.0f};
  std::vector<std::int8_t> q(x.size());
  esm::quant::QuantizeActivationSymmetricRef(x.data(), x.size(), 0.0f, q.data());
  for (auto v : q) EXPECT_EQ(static_cast<int>(v), 0);
}

#if defined(__aarch64__) || defined(_M_ARM64)
namespace esm::kernels {
// Internal NEON activation prefix (absmax + symmetric s8 quantize), defined
// in the NEON TU of gemm_int8.cpp. Forward-declared here for a direct
// cross-check against the scalar symmetric reference.
void QuantizeActPrefixNeon(const float* A, std::size_t MK, std::int8_t* a_s8,
                           float* act_scale_out);
}  // namespace esm::kernels

TEST(ActivationQuantizePrefixNeon, MatchesSymmetricRefAndComputesScale) {
  const std::size_t N = 1031;  // odd, exercises the scalar tail
  auto x = RandomVec(N, 0xc33e);
  float expect_absmax = 0.0f;
  for (auto v : x) expect_absmax = std::max(expect_absmax, std::fabs(v));
  const float expect_scale = expect_absmax / 127.0f;

  std::vector<std::int8_t> neon(N);
  float scale = 0.0f;
  esm::kernels::QuantizeActPrefixNeon(x.data(), N, neon.data(), &scale);
  EXPECT_NEAR(scale, expect_scale, 1e-6f * (1.0f + expect_scale));

  std::vector<std::int8_t> ref(N);
  esm::quant::QuantizeActivationSymmetricRef(x.data(), N, scale, ref.data());
  for (std::size_t i = 0; i < N; ++i) {
    EXPECT_LE(std::abs(static_cast<int>(neon[i]) - static_cast<int>(ref[i])), 1)
        << "i=" << i;
  }
}
#endif
