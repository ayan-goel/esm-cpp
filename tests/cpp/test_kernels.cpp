#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "esm_cpp/cpu_features.h"
#include "esm_cpp/kernels.h"

namespace {
constexpr float kFp32Tol = 1e-5f;

// Build-time x86 != run-time AVX-512. CI runners are commonly Skylake-
// client / Haswell-class with only AVX2; the AVX-512 OBJECT lib still
// compiles into the binary but its instructions trap at run time. Gate
// each AVX-512 test on actual host capability.
#if defined(__x86_64__) || defined(_M_X64)
bool HostHasAvx512() {
  const esm::Isa isa = esm::HostIsa();
  return isa == esm::Isa::Avx512 || isa == esm::Isa::Avx512Vnni ||
         isa == esm::Isa::Amx;
}
#endif
}  // namespace

TEST(ResidualAddInplaceRef, BasicMath) {
  std::vector<float> y = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> x = {0.5f, -1.0f, 10.0f, 100.0f};
  esm::kernels::ResidualAddInplaceRef(y.data(), x.data(), y.size());
  std::vector<float> expected = {1.5f, 1.0f, 13.0f, 104.0f};
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(y[i], expected[i], 1e-6f);
}

#if defined(__x86_64__) || defined(_M_X64)
namespace esm::kernels {
void ResidualAddInplaceAvx512(float* y, const float* x, std::size_t n);
void ScaleInplaceAvx512(float* x, std::size_t n, float scale);
}  // namespace esm::kernels

TEST(ScaleInplaceAvx512, MatchesRefAcrossScalesAndSizes) {
  if (!HostHasAvx512()) GTEST_SKIP() << "host lacks AVX-512";
  for (float scale : {1.0f, 0.0f, -1.0f, 0.125f, 1.0f / 8.0f /*Q-scale @ dh=64*/}) {
    for (std::size_t n : {std::size_t{1}, std::size_t{15}, std::size_t{17},
                           std::size_t{32}, std::size_t{4097},
                           std::size_t{8 * 1280}}) {
      std::vector<float> x_ref(n), x_got(n);
      for (std::size_t i = 0; i < n; ++i) {
        x_ref[i] = 0.5f * std::cos(static_cast<float>(i) * 0.019f);
        x_got[i] = x_ref[i];
      }
      esm::kernels::ScaleInplaceRef(x_ref.data(), n, scale);
      esm::kernels::ScaleInplaceAvx512(x_got.data(), n, scale);
      for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_got[i], x_ref[i], 1e-6f)
            << "n=" << n << " scale=" << scale << " i=" << i;
      }
    }
  }
}

TEST(ResidualAddInplaceAvx512, MatchesRefOnDenseSweep) {
  if (!HostHasAvx512()) GTEST_SKIP() << "host lacks AVX-512";
  // Sizes cover tail cases (< 16, multiples of 16, multiples of 4096, and
  // a 650M-scale ffn-sized vector to exercise the parallel chunking).
  for (std::size_t n : {std::size_t{1}, std::size_t{15}, std::size_t{16},
                         std::size_t{17}, std::size_t{31}, std::size_t{32},
                         std::size_t{1023}, std::size_t{1024}, std::size_t{1025},
                         std::size_t{4095}, std::size_t{4096}, std::size_t{4097},
                         std::size_t{8 * 1280}, std::size_t{2048 * 5120}}) {
    std::vector<float> y_ref(n), y_got(n), x(n);
    for (std::size_t i = 0; i < n; ++i) {
      y_ref[i] = 0.3f * std::sin(static_cast<float>(i) * 0.011f);
      y_got[i] = y_ref[i];
      x[i] = 0.4f * std::cos(static_cast<float>(i) * 0.017f);
    }
    esm::kernels::ResidualAddInplaceRef(y_ref.data(), x.data(), n);
    esm::kernels::ResidualAddInplaceAvx512(y_got.data(), x.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
      // Elementwise add is bit-identical (no reordering); 1e-6 is overkill
      // but covers the masked-tail rounding edge if any.
      EXPECT_NEAR(y_got[i], y_ref[i], 1e-6f) << "n=" << n << " i=" << i;
    }
  }
}
#endif  // x86_64

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

#if defined(__x86_64__) || defined(_M_X64)
namespace esm::kernels {
void LayerNormAvx512(const float* x, const float* gamma, const float* beta,
                     float eps, float* out, int num_rows, int d);
}

TEST(LayerNormAvx512, MatchesRefAcrossEsmDims) {
  if (!HostHasAvx512()) GTEST_SKIP() << "host lacks AVX-512";
  // Sweep d across the ESM-2 dims plus tail-exercising odd sizes.
  for (int d : {17, 47, 320, 480, 640, 1280, 2560, 5120}) {
    constexpr int kRows = 4;
    std::vector<float> x(static_cast<std::size_t>(kRows) * d);
    std::vector<float> gamma(d), beta(d);
    for (int i = 0; i < d * kRows; ++i) {
      // Mix of magnitudes so mean and variance are non-trivial.
      x[static_cast<std::size_t>(i)] =
          0.5f * std::sin(static_cast<float>(i) * 0.013f) +
          0.2f * std::cos(static_cast<float>(i) * 0.007f);
    }
    for (int j = 0; j < d; ++j) {
      gamma[static_cast<std::size_t>(j)] = 1.0f + 0.1f * std::sin(static_cast<float>(j) * 0.31f);
      beta[static_cast<std::size_t>(j)] = 0.05f * std::cos(static_cast<float>(j) * 0.17f);
    }
    std::vector<float> ref(static_cast<std::size_t>(kRows) * d);
    std::vector<float> got(static_cast<std::size_t>(kRows) * d);
    esm::kernels::LayerNormRef(x.data(), gamma.data(), beta.data(), 1e-5f,
                                ref.data(), kRows, d);
    esm::kernels::LayerNormAvx512(x.data(), gamma.data(), beta.data(), 1e-5f,
                                   got.data(), kRows, d);
    for (int i = 0; i < kRows * d; ++i) {
      // FP32-reduction drift vs the Ref FP64 accumulator. Multi-accumulator
      // pattern bounds it well below 1e-5 for d <= 5120.
      EXPECT_NEAR(got[static_cast<std::size_t>(i)],
                  ref[static_cast<std::size_t>(i)], 1e-5f)
          << "d=" << d << " i=" << i;
    }
  }
}
#endif  // x86_64

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
  if (!HostHasAvx512()) GTEST_SKIP() << "host lacks AVX-512";
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
  if (!HostHasAvx512()) GTEST_SKIP() << "host lacks AVX-512";
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

#if defined(__x86_64__) || defined(_M_X64)
namespace esm::kernels {
void RopeApplyVarlenAvx512(float* x, const float* cos, const float* sin,
                           const int* cu_seqlens, int batch_size,
                           int num_heads, int head_dim);
}  // namespace esm::kernels

namespace {
struct RopeShape {
  int B;
  int H;
  int dh;
  std::vector<int> seq_lens;  // length B
};

void RunRopeShape(const RopeShape& sh) {
  int T = 0;
  for (int l : sh.seq_lens) T += l;
  int max_seq = 0;
  for (int l : sh.seq_lens) max_seq = std::max(max_seq, l);
  std::vector<int> cu(sh.B + 1, 0);
  for (int b = 0; b < sh.B; ++b) cu[b + 1] = cu[b] + sh.seq_lens[b];

  std::vector<float> cos(static_cast<std::size_t>(max_seq) * sh.dh);
  std::vector<float> sin(static_cast<std::size_t>(max_seq) * sh.dh);
  esm::kernels::RopeBuildTables(max_seq, sh.dh, cos.data(), sin.data());

  std::vector<float> x_ref(static_cast<std::size_t>(T) * sh.H * sh.dh);
  for (std::size_t i = 0; i < x_ref.size(); ++i) {
    x_ref[i] = 0.3f * std::sin(static_cast<float>(i) * 0.019f);
  }
  std::vector<float> x_got = x_ref;

  esm::kernels::RopeApplyVarlenRef(x_ref.data(), cos.data(), sin.data(),
                                    cu.data(), sh.B, sh.H, sh.dh);
  esm::kernels::RopeApplyVarlenAvx512(x_got.data(), cos.data(), sin.data(),
                                       cu.data(), sh.B, sh.H, sh.dh);
  for (std::size_t i = 0; i < x_ref.size(); ++i) {
    EXPECT_NEAR(x_got[i], x_ref[i], 1e-6f)
        << "B=" << sh.B << " H=" << sh.H << " dh=" << sh.dh
        << " i=" << i;
  }
}
}  // namespace

TEST(RopeApplyVarlenAvx512, MatchesRefAcrossEsmDims) {
  if (!HostHasAvx512()) GTEST_SKIP() << "host lacks AVX-512";
  // ESM-2 head_dim coverage: 16 (8M), 24 (35M), 32 (150M), 64 (650M).
  // Mix of B (1 / 4 / 8) and seq_lens including odd values to exercise
  // the half < 16 masked-tail path (dh=16 → half=8) and the dh-not-
  // multiple-of-16 case (dh=24 → half=12).
  RunRopeShape({1, 5,  16, {100}});                  // 8M
  RunRopeShape({8, 5,  16, {100, 100, 100, 100, 100, 100, 100, 100}}); // 8M batch
  RunRopeShape({4, 20, 24, {128, 128, 128, 128}});   // 35M
  RunRopeShape({2, 20, 32, {256, 37}});              // 150M + odd seq tail
  RunRopeShape({8, 20, 64, {256, 256, 256, 256, 256, 256, 256, 256}}); // 650M
  RunRopeShape({3, 20, 64, {1, 1, 1}});              // single-token edge
}
#endif  // x86_64

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

#if defined(__aarch64__) || defined(_M_ARM64)
namespace esm::kernels {
void LayerNormNeon(const float* x, const float* gamma, const float* beta,
                   float eps, float* out, int num_rows, int d);
void GeluNeon(const float* x, float* out, std::size_t n);
void ResidualAddInplaceNeon(float* y, const float* x, std::size_t n);
void ScaleInplaceNeon(float* x, std::size_t n, float scale);
}  // namespace esm::kernels

TEST(LayerNormNeon, MatchesRefAcrossEsmDims) {
  for (int d : {17, 47, 320, 480, 640, 1280, 2560, 5120}) {
    constexpr int kRows = 4;
    std::vector<float> x(static_cast<std::size_t>(kRows) * d);
    std::vector<float> gamma(d), beta(d);
    for (int i = 0; i < d * kRows; ++i) {
      x[static_cast<std::size_t>(i)] =
          0.5f * std::sin(static_cast<float>(i) * 0.013f) +
          0.2f * std::cos(static_cast<float>(i) * 0.007f);
    }
    for (int j = 0; j < d; ++j) {
      gamma[static_cast<std::size_t>(j)] =
          1.0f + 0.1f * std::sin(static_cast<float>(j) * 0.31f);
      beta[static_cast<std::size_t>(j)] =
          0.05f * std::cos(static_cast<float>(j) * 0.17f);
    }
    std::vector<float> ref(static_cast<std::size_t>(kRows) * d);
    std::vector<float> got(static_cast<std::size_t>(kRows) * d);
    esm::kernels::LayerNormRef(x.data(), gamma.data(), beta.data(), 1e-5f,
                               ref.data(), kRows, d);
    esm::kernels::LayerNormNeon(x.data(), gamma.data(), beta.data(), 1e-5f,
                                got.data(), kRows, d);
    for (int i = 0; i < kRows * d; ++i) {
      EXPECT_NEAR(got[static_cast<std::size_t>(i)],
                  ref[static_cast<std::size_t>(i)], 1e-5f)
          << "d=" << d << " i=" << i;
    }
  }
}

TEST(GeluNeon, MatchesRefOnDenseSweep) {
  constexpr int kN = 8195;  // odd so the scalar tail runs
  std::vector<float> x(kN), ref(kN), got(kN);
  for (int i = 0; i < kN; ++i) {
    x[i] = -5.0f + 10.0f * (static_cast<float>(i) / static_cast<float>(kN - 1));
  }
  esm::kernels::GeluRef(x.data(), ref.data(), kN);
  esm::kernels::GeluNeon(x.data(), got.data(), kN);
  for (int i = 0; i < kN; ++i) {
    EXPECT_NEAR(got[i], ref[i], 1e-5f) << "i=" << i << " x=" << x[i];
  }
}

TEST(GeluNeon, LargeMagnitudeStaysFiniteAndMatchesRef) {
  // Guards the ExpNeon input clamp on the gelu path: ErfNeon computes
  // ExpNeon(-x*x), so |x| > ~9.3 drives the argument below ExpNeon's -87 clamp.
  // Without the clamp the 2^n exponent-bit path wraps to garbage — the same
  // class of bug as the P9 attention NaN (which a timing-only bench hid). fc1
  // outputs can be large, so gelu must stay finite and match the scalar ref.
  std::vector<float> x = {-100.0f, -50.0f, -30.0f, -9.5f, -5.0f, 0.0f,
                          5.0f,    9.5f,   30.0f,  50.0f,  100.0f};
  std::vector<float> ref(x.size()), got(x.size());
  esm::kernels::GeluRef(x.data(), ref.data(), x.size());
  esm::kernels::GeluNeon(x.data(), got.data(), x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_TRUE(std::isfinite(got[i])) << "x=" << x[i] << " got=" << got[i];
    EXPECT_NEAR(got[i], ref[i], 1e-4f * (1.0f + std::fabs(ref[i]))) << "x=" << x[i];
  }
}

TEST(ResidualAddInplaceNeon, MatchesRefOnDenseSweep) {
  for (std::size_t n : {1u, 4u, 15u, 64u, 4097u}) {
    std::vector<float> y_ref(n), y_got(n), x(n);
    for (std::size_t i = 0; i < n; ++i) {
      y_ref[i] = y_got[i] = std::sin(static_cast<float>(i) * 0.1f);
      x[i] = std::cos(static_cast<float>(i) * 0.2f);
    }
    esm::kernels::ResidualAddInplaceRef(y_ref.data(), x.data(), n);
    esm::kernels::ResidualAddInplaceNeon(y_got.data(), x.data(), n);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_NEAR(y_got[i], y_ref[i], 1e-6f) << "n=" << n << " i=" << i;
  }
}

TEST(ScaleInplaceNeon, MatchesRefAcrossScalesAndSizes) {
  for (std::size_t n : {1u, 7u, 16u, 4099u}) {
    for (float scale : {0.0f, 0.88f, -2.5f}) {
      std::vector<float> x_ref(n), x_got(n);
      for (std::size_t i = 0; i < n; ++i)
        x_ref[i] = x_got[i] = std::sin(static_cast<float>(i) * 0.05f);
      esm::kernels::ScaleInplaceRef(x_ref.data(), n, scale);
      esm::kernels::ScaleInplaceNeon(x_got.data(), n, scale);
      for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR(x_got[i], x_ref[i], 1e-6f) << "n=" << n << " i=" << i;
    }
  }
}
#endif  // aarch64
