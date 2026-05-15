// Phase 2 Slice 4: SmoothQuant migration math.
//
// The migration is a diagonal rescale that's identity-preserving for
// the FP32 forward — moving activation outliers into weights does not
// change what the model computes, only how it computes it. These tests
// exercise the math at the kernel level; the full FP32-identity check
// across a real Model is in tests/python/test_smoothquant.py.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "esm_cpp/kernels.h"
#include "esm_cpp/model.h"
#include "esm_cpp/smoothquant.h"

namespace {

std::vector<float> RandomVec(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

}  // namespace

TEST(SmoothQuantScale, AlphaZeroProducesUnitScales) {
  // alpha=0 => s = max(|W|)^(-1) which... actually the formula at alpha=0
  // is s = 1 / max(|W|), which scales weights down. The Identity check
  // still holds. The unit-scale property is alpha-specific to weight max.
  std::vector<float> P_X = {2.0f, 3.0f, 5.0f};
  std::vector<float> P_W = {1.0f, 1.0f, 1.0f};
  auto s = esm::quant::SmoothQuantScales(P_X, P_W, 0.0f);
  ASSERT_EQ(s.size(), 3u);
  // s[k] = P_X[k]^0 / P_W[k]^1 = 1 / 1 = 1
  for (float v : s) EXPECT_NEAR(v, 1.0f, 1e-6f);
}

TEST(SmoothQuantScale, AlphaHalfIsGeometricMean) {
  // s[k] = sqrt(P_X[k] / P_W[k]).
  std::vector<float> P_X = {4.0f, 9.0f, 16.0f};
  std::vector<float> P_W = {1.0f, 1.0f, 1.0f};
  auto s = esm::quant::SmoothQuantScales(P_X, P_W, 0.5f);
  EXPECT_NEAR(s[0], 2.0f, 1e-6f);  // sqrt(4)
  EXPECT_NEAR(s[1], 3.0f, 1e-6f);  // sqrt(9)
  EXPECT_NEAR(s[2], 4.0f, 1e-6f);  // sqrt(16)
}

TEST(SmoothQuantScale, ZeroActivationOrZeroWeightProducesUnitScale) {
  // If P_X[k] is 0 or P_W[k] is 0, the rescale would blow up; the
  // implementation must fall back to s[k] = 1 (no-op for that channel).
  std::vector<float> P_X = {0.0f, 3.0f, 0.0f, 1.0f};
  std::vector<float> P_W = {1.0f, 0.0f, 0.0f, 1.0f};
  auto s = esm::quant::SmoothQuantScales(P_X, P_W, 0.5f);
  EXPECT_NEAR(s[0], 1.0f, 1e-6f);
  EXPECT_NEAR(s[1], 1.0f, 1e-6f);
  EXPECT_NEAR(s[2], 1.0f, 1e-6f);
  // Non-zero pair: sqrt(1/1) = 1.
  EXPECT_NEAR(s[3], 1.0f, 1e-6f);
}

TEST(SmoothQuantScale, ClampedToSaneRange) {
  // s[k] is clamped to [1e-3, 1e3] so a single pathological channel
  // doesn't make the activation rescale numerically unsafe.
  std::vector<float> P_X = {1e10f};
  std::vector<float> P_W = {1e-10f};
  auto s = esm::quant::SmoothQuantScales(P_X, P_W, 1.0f);
  EXPECT_LE(s[0], 1e3f + 1.0f);
  EXPECT_GE(s[0], 1.0f);  // upper-bounded but still positive
}

TEST(SmoothQuant, LayerNormToLinearIdentityHoldsAtAlphaHalf) {
  // (LayerNorm -> Linear) with synthetic gamma/beta and W:
  // original_y[i, n] = LinearRef(LayerNorm(x), W, b)[i, n]
  // migrated_y[i, n] = LinearRef(LayerNorm(x with rescaled gamma/beta),
  //                              W with rescaled cols, b)[i, n]
  // The two must match within FP32 round-off.
  const int rows = 4, d_in = 8, d_out = 6;
  auto x = RandomVec(rows * d_in, 0x4001);
  auto gamma = RandomVec(d_in, 0x4002);
  auto beta = RandomVec(d_in, 0x4003);
  auto w = RandomVec(d_out * d_in, 0x4004);
  auto b = RandomVec(d_out, 0x4005);
  // Make gamma positive so dividing by it is well-defined.
  for (auto& g : gamma) g = std::fabs(g) + 0.1f;

  // Choose synthetic per-channel scales by hand.
  std::vector<float> s(d_in);
  for (int k = 0; k < d_in; ++k) s[k] = 0.5f + 0.1f * static_cast<float>(k);

  // Forward: original.
  std::vector<float> ln_orig(rows * d_in);
  esm::kernels::LayerNormRef(x.data(), gamma.data(), beta.data(), 1e-5f,
                              ln_orig.data(), rows, d_in);
  std::vector<float> y_orig(rows * d_out);
  esm::kernels::LinearRef(ln_orig.data(), w.data(), b.data(), y_orig.data(),
                           rows, d_out, d_in);

  // Forward: migrated. gamma/beta divided by s; w[n, k] multiplied by s[k].
  std::vector<float> gamma_m(d_in), beta_m(d_in), w_m(w);
  for (int k = 0; k < d_in; ++k) {
    gamma_m[k] = gamma[k] / s[k];
    beta_m[k] = beta[k] / s[k];
  }
  for (int n = 0; n < d_out; ++n) {
    for (int k = 0; k < d_in; ++k) {
      w_m[n * d_in + k] = w[n * d_in + k] * s[k];
    }
  }
  std::vector<float> ln_mig(rows * d_in);
  esm::kernels::LayerNormRef(x.data(), gamma_m.data(), beta_m.data(), 1e-5f,
                              ln_mig.data(), rows, d_in);
  std::vector<float> y_mig(rows * d_out);
  esm::kernels::LinearRef(ln_mig.data(), w_m.data(), b.data(), y_mig.data(),
                           rows, d_out, d_in);

  for (int i = 0; i < rows * d_out; ++i) {
    EXPECT_NEAR(y_orig[i], y_mig[i], 1e-4f * (std::fabs(y_orig[i]) + 1.0f))
        << "i=" << i;
  }
}
