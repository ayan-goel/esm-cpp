#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "esm_cpp/kernels.h"

namespace {
constexpr float kFp32Tol = 1e-5f;
}  // namespace

TEST(LinearRef, MatchesHandComputed_2x3_times_4x3_plus_bias) {
  // A: [[1, 2, 3], [4, 5, 6]]            (M=2, K=3)
  // W: [[1, 0, -1], [0, 1, 0], [1, 1, 1], [2, -1, 0]]  (N=4, K=3)
  // bias: [10, 20, 30, 40]
  // Expected (out[m,n] = sum_k A[m,k]*W[n,k] + bias[n]):
  //   row 0: [1-3+10, 2+20, 6+30, 0+40]    = [8, 22, 36, 40]
  //   row 1: [4-6+10, 5+20, 15+30, 3+40]   = [8, 25, 45, 43]
  std::vector<float> A = {1, 2, 3, 4, 5, 6};
  std::vector<float> W = {1, 0, -1, 0, 1, 0, 1, 1, 1, 2, -1, 0};
  std::vector<float> bias = {10, 20, 30, 40};
  std::vector<float> C(8);
  esm::kernels::LinearRef(A.data(), W.data(), bias.data(), C.data(), 2, 4, 3);
  std::vector<float> expected = {8, 22, 36, 40, 8, 25, 45, 43};
  for (size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(C[i], expected[i], kFp32Tol);
}

TEST(LinearRef, NoBias) {
  std::vector<float> A = {1, 1};
  std::vector<float> W = {2, 3, 4, 5};  // N=2, K=2
  std::vector<float> C(2);
  esm::kernels::LinearRef(A.data(), W.data(), nullptr, C.data(), 1, 2, 2);
  EXPECT_NEAR(C[0], 5.0f, kFp32Tol);
  EXPECT_NEAR(C[1], 9.0f, kFp32Tol);
}

TEST(LayerNormRef, ZeroMean_UnitVar_AfterNorm) {
  std::vector<float> x = {1, 2, 3, 4};
  std::vector<float> gamma = {1, 1, 1, 1};
  std::vector<float> beta = {0, 0, 0, 0};
  std::vector<float> out(4);
  esm::kernels::LayerNormRef(x.data(), gamma.data(), beta.data(), 1e-5f,
                             out.data(), 1, 4);
  // mean = 2.5, var = (2.25+0.25+0.25+2.25)/4 = 1.25
  // normed[i] = (x[i] - 2.5) / sqrt(1.25 + 1e-5)
  float inv_std = 1.0f / std::sqrt(1.25f + 1e-5f);
  std::vector<float> expected = {-1.5f * inv_std, -0.5f * inv_std,
                                  0.5f * inv_std, 1.5f * inv_std};
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(out[i], expected[i], 1e-5f);
}

TEST(LayerNormRef, GammaBetaApplied) {
  std::vector<float> x = {1, 2, 3, 4};
  std::vector<float> gamma = {2, 2, 2, 2};
  std::vector<float> beta = {1, 1, 1, 1};
  std::vector<float> out(4);
  esm::kernels::LayerNormRef(x.data(), gamma.data(), beta.data(), 1e-5f,
                             out.data(), 1, 4);
  float inv_std = 1.0f / std::sqrt(1.25f + 1e-5f);
  std::vector<float> base = {-1.5f * inv_std, -0.5f * inv_std,
                              0.5f * inv_std, 1.5f * inv_std};
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(out[i], 2.0f * base[i] + 1.0f, 1e-5f);
}

TEST(GeluRef, ZeroAtZero) {
  float x[] = {0.0f};
  float y[1];
  esm::kernels::GeluRef(x, y, 1);
  EXPECT_NEAR(y[0], 0.0f, 1e-7f);
}

TEST(GeluRef, MonotonicAndKnownValues) {
  // gelu(1) = 1 * 0.5 * (1 + erf(1/sqrt(2))) ≈ 0.8413447
  // gelu(-1) ≈ -0.1586553
  float x[] = {1.0f, -1.0f, 2.0f};
  float y[3];
  esm::kernels::GeluRef(x, y, 3);
  EXPECT_NEAR(y[0], 0.8413447f, 1e-5f);
  EXPECT_NEAR(y[1], -0.1586553f, 1e-5f);
  EXPECT_NEAR(y[2], 1.9544997f, 1e-5f);
}

#if defined(__x86_64__) || defined(_M_X64)
namespace esm::kernels { void GeluAvx512(const float* x, float* out, std::size_t n); }

TEST(GeluAvx512, MatchesRefOnDenseSweep) {
  // [-5, 5] is the range GELU actually sees inside ESM-2 (post-LN
  // activations are O(1)). Dense sweep with a non-16-multiple count
  // exercises the masked tail path too.
  constexpr int kN = 8195;  // odd so the tail handler runs
  std::vector<float> x(kN), ref(kN), got(kN);
  for (int i = 0; i < kN; ++i) {
    x[i] = -5.0f + 10.0f * (static_cast<float>(i) / static_cast<float>(kN - 1));
  }
  esm::kernels::GeluRef(x.data(), ref.data(), kN);
  esm::kernels::GeluAvx512(x.data(), got.data(), kN);
  for (int i = 0; i < kN; ++i) {
    // Polynomial erf (A&S 7.1.26) is ~1.5e-7; gelu = x * 0.5 * (1+erf),
    // |x| <= 5, so absolute diff stays well under 1e-5.
    EXPECT_NEAR(got[i], ref[i], 1e-5f)
        << "i=" << i << " x=" << x[i] << " ref=" << ref[i] << " got=" << got[i];
  }
}

TEST(GeluAvx512, ZeroAndExtremes) {
  // Sign + saturation: GELU(0) == 0, GELU(very negative) ~ 0, GELU(very
  // positive) ~ x. Polynomial erf saturates to ±1 outside ~|x| > 4.
  float x[] = {0.0f, 10.0f, -10.0f, 1.0f, -1.0f};
  float ref[5], got[5];
  esm::kernels::GeluRef(x, ref, 5);
  esm::kernels::GeluAvx512(x, got, 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(got[i], ref[i], 1e-5f) << "i=" << i << " x=" << x[i];
  }
}
#endif  // x86_64

TEST(RopeBuildTables, CosSinAreDuplicatedHalfThenHalf) {
  // For head_dim=4, half=2. Position t=1.
  // inv_freq[0] = 10000^(-0/4) = 1
  // inv_freq[1] = 10000^(-2/4) = 1/100
  std::vector<float> cos(4 * 4), sin(4 * 4);  // seq_len=4, head_dim=4
  esm::kernels::RopeBuildTables(4, 4, cos.data(), sin.data());
  // At t=0: cos = [1, 1, 1, 1], sin = [0, 0, 0, 0]
  for (int j = 0; j < 4; ++j) {
    EXPECT_NEAR(cos[j], 1.0f, 1e-6f);
    EXPECT_NEAR(sin[j], 0.0f, 1e-6f);
  }
  // Half-then-half property at every t: cos[t, j] == cos[t, j+half]
  for (int t = 0; t < 4; ++t) {
    EXPECT_NEAR(cos[t * 4 + 0], cos[t * 4 + 2], 1e-6f);
    EXPECT_NEAR(cos[t * 4 + 1], cos[t * 4 + 3], 1e-6f);
    EXPECT_NEAR(sin[t * 4 + 0], sin[t * 4 + 2], 1e-6f);
    EXPECT_NEAR(sin[t * 4 + 1], sin[t * 4 + 3], 1e-6f);
  }
}

TEST(RopeApplyInplace, IdentityAtPositionZero) {
  // At t=0 all sin terms are zero and cos terms are 1, so RoPE is identity.
  const int H = 1, L = 1, dh = 4;
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> cos(L * dh), sin(L * dh);
  esm::kernels::RopeBuildTables(L, dh, cos.data(), sin.data());
  esm::kernels::RopeApplyInplaceRef(x.data(), cos.data(), sin.data(), H, L, dh);
  EXPECT_NEAR(x[0], 1.0f, 1e-6f);
  EXPECT_NEAR(x[1], 2.0f, 1e-6f);
  EXPECT_NEAR(x[2], 3.0f, 1e-6f);
  EXPECT_NEAR(x[3], 4.0f, 1e-6f);
}

TEST(RopeApplyInplace, HalfThenHalfRotation) {
  // head_dim=4, half=2. At position t=1, inv_freq=[1, 1/100].
  // angles = [1*1, 1*0.01] = [1.0, 0.01]
  // c = [cos(1), cos(0.01), cos(1), cos(0.01)]
  // s = [sin(1), sin(0.01), sin(1), sin(0.01)]
  // x = [x1, x2, x3, x4]
  // rotate_half(x) = [-x3, -x4, x1, x2]
  // y = x * c + rotate_half(x) * s
  //   y[0] = x1 * cos(1) + (-x3) * sin(1)
  //   y[1] = x2 * cos(.01) + (-x4) * sin(.01)
  //   y[2] = x3 * cos(1) + x1 * sin(1)
  //   y[3] = x4 * cos(.01) + x2 * sin(.01)
  const int H = 1, L = 2, dh = 4;
  std::vector<float> x = {0, 0, 0, 0, 1, 2, 3, 4};  // pos 0 untouched, pos 1 = [1,2,3,4]
  std::vector<float> cos(L * dh), sin(L * dh);
  esm::kernels::RopeBuildTables(L, dh, cos.data(), sin.data());
  esm::kernels::RopeApplyInplaceRef(x.data(), cos.data(), sin.data(), H, L, dh);
  float c1 = std::cos(1.0f), s1 = std::sin(1.0f);
  float c01 = std::cos(0.01f), s01 = std::sin(0.01f);
  EXPECT_NEAR(x[4], 1.0f * c1 + (-3.0f) * s1, 1e-5f);
  EXPECT_NEAR(x[5], 2.0f * c01 + (-4.0f) * s01, 1e-5f);
  EXPECT_NEAR(x[6], 3.0f * c1 + 1.0f * s1, 1e-5f);
  EXPECT_NEAR(x[7], 4.0f * c01 + 2.0f * s01, 1e-5f);
}

TEST(AttentionRef, SingleHeadSingleTokenIsIdentityOnV) {
  // L=1, H=1, dh=2. softmax(QK^T) = 1, so out = V.
  std::vector<float> Q = {0.5f, 0.5f};
  std::vector<float> K = {0.1f, 0.2f};
  std::vector<float> V = {7.0f, 8.0f};
  std::vector<float> out(2);
  esm::kernels::AttentionRef(Q.data(), K.data(), V.data(), nullptr,
                              out.data(), 1, 1, 2);
  EXPECT_NEAR(out[0], 7.0f, 1e-5f);
  EXPECT_NEAR(out[1], 8.0f, 1e-5f);
}

TEST(AttentionRef, MaskZeroesOutAttention) {
  // L=2, but mask[1]=0 means pad. Softmax should put all weight on j=0.
  std::vector<float> Q = {1, 0};               // [H=1, L=2*?, no — H=1, L=2 means 4 floats wait]
  // Q, K, V shape [H, L, dh]; H=1, L=2, dh=1
  std::vector<float> q = {1.0f, 1.0f};         // queries at L=2
  std::vector<float> k = {2.0f, 100.0f};       // huge K for j=1 — would dominate without mask
  std::vector<float> v = {5.0f, 99.0f};
  std::vector<int> mask = {1, 0};              // mask j=1
  std::vector<float> out(2);
  esm::kernels::AttentionRef(q.data(), k.data(), v.data(), mask.data(),
                              out.data(), 1, 2, 1);
  // For each query (i=0,1): only j=0 contributes, so out = v[0] = 5.
  EXPECT_NEAR(out[0], 5.0f, 1e-5f);
  EXPECT_NEAR(out[1], 5.0f, 1e-5f);
}
