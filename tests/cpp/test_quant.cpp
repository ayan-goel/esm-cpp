// Phase 2 Slice 2 acceptance: per-channel symmetric INT8 weight packing
// (range [-127, 127], no zero-point) plus a scalar dequant-on-the-fly
// matmul that matches LinearRef on FP32-dequantized weights within the
// per-channel quantization error bound.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "esm_cpp/kernels.h"
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

TEST(QuantPack, RoundTripMaxAbsBoundedByPerChannelStep) {
  const int N = 7, K = 13;
  auto W = RandomVec(N * K, 0x9001);
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), N, K, &qt);
  ASSERT_EQ(static_cast<int>(qt.packed.size()), N * K);
  ASSERT_EQ(static_cast<int>(qt.per_channel_scales.size()), N);
  // Per-row scale must be max-abs / 127 (and strictly positive).
  for (int n = 0; n < N; ++n) {
    float row_max = 0.0f;
    for (int k = 0; k < K; ++k) row_max = std::max(row_max, std::fabs(W[n * K + k]));
    EXPECT_NEAR(qt.per_channel_scales[n], row_max / 127.0f, 1e-7f * row_max + 1e-12f);
    EXPECT_GT(qt.per_channel_scales[n], 0.0f);
  }
  // Round-trip error per element bounded by the row's step size.
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      float reconstructed = static_cast<float>(qt.packed[n * K + k]) *
                            qt.per_channel_scales[n];
      float err = std::fabs(reconstructed - W[n * K + k]);
      EXPECT_LE(err, qt.per_channel_scales[n] * 1.000001f) << "n=" << n << " k=" << k;
    }
  }
}

#if defined(__x86_64__) || defined(_M_X64)
TEST(QuantPack, BuildVnniCachePopulatesPackedAndColSum) {
  // After Quantize, packed_vnni + col_sum derived fields must be populated
  // so that LinearVnni can skip per-call PackWeight + col_sum recomputation.
  // packed_vnni layout: for each 16-N panel, for each 4-K tile, 64 bytes
  // arranged as [N0:k0,k1,k2,k3, N1:k0,..., N15:k0,k1,k2,k3]. K_pad rounds
  // K up to a multiple of 4; N-tail and K-tail are zero-padded.
  const int N = 19, K = 13;  // N tail (% 16 != 0) and K tail (% 4 != 0)
  auto W = RandomVec(N * K, 0xc101);
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), N, K, &qt);
  const int K_pad = (K + 3) & ~3;
  const int N_pad = (N + 15) & ~15;
  ASSERT_EQ(static_cast<int>(qt.packed_vnni.size()), N_pad * K_pad);
  ASSERT_EQ(static_cast<int>(qt.col_sum.size()), N);
  for (int n = 0; n < N; ++n) {
    std::int32_t expect = 0;
    for (int k = 0; k < K; ++k) {
      expect += static_cast<std::int32_t>(qt.packed[n * K + k]);
    }
    EXPECT_EQ(qt.col_sum[n], expect) << "n=" << n;
  }
  for (int nb = 0; nb < N; nb += 16) {
    const int n_block = std::min(16, N - nb);
    const std::int8_t* panel = qt.packed_vnni.data() + nb * K_pad;
    for (int kb = 0; kb < K_pad; kb += 4) {
      const std::int8_t* tile = panel + kb * 16;
      for (int nn = 0; nn < n_block; ++nn) {
        for (int kk = 0; kk < 4; ++kk) {
          const std::int8_t expect = (kb + kk < K)
              ? qt.packed[(nb + nn) * K + (kb + kk)]
              : std::int8_t{0};
          EXPECT_EQ(tile[nn * 4 + kk], expect)
              << "nb=" << nb << " kb=" << kb << " nn=" << nn << " kk=" << kk;
        }
      }
    }
  }
}

TEST(QuantPack, BuildVnniCacheIdempotent) {
  // BuildVnniCache called explicitly after Quantize must produce identical
  // results; the cache is a deterministic function of qt.packed.
  const int N = 5, K = 13;
  auto W = RandomVec(N * K, 0xc102);
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), N, K, &qt);
  auto packed_vnni_1 = qt.packed_vnni;
  auto col_sum_1 = qt.col_sum;
  esm::quant::BuildVnniCache(&qt);
  EXPECT_EQ(qt.packed_vnni, packed_vnni_1);
  EXPECT_EQ(qt.col_sum, col_sum_1);
}
#endif  // x86_64

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(QuantPack, BuildArmCachePopulatesPackedArm) {
  // After Quantize on ARM, packed_arm holds SDOT tiles: each 4-N panel has
  // K_pad/4 tiles of 16 bytes laid out [n0:k0..k3, n1:.., n2:.., n3:..].
  // N_pad rounds N up to 4, K_pad rounds K up to 4; tails are zero-padded.
  const int N = 19, K = 37;  // N tail (% 4 != 0) and K tail (% 4 != 0)
  auto W = RandomVec(N * K, 0xa401);
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), N, K, &qt);
  // Build the SDOT cache explicitly: which cache Quantize builds by default
  // depends on the host tier (NeonI8mm builds SMMLA instead). BuildArmCache
  // is pure C++ and callable on any host.
  esm::quant::BuildArmCache(&qt);
  const int K_pad = (K + 3) & ~3;
  const int N_pad = (N + 3) & ~3;
  ASSERT_EQ(static_cast<int>(qt.packed_arm.size()), N_pad * K_pad);
  EXPECT_TRUE(qt.packed_vnni.empty());  // x86 cache not built on ARM
  for (int nb = 0; nb < N_pad; nb += 4) {
    const std::int8_t* panel = qt.packed_arm.data() + nb * K_pad;
    for (int kb = 0; kb < K_pad; kb += 4) {
      const std::int8_t* tile = panel + kb * 4;
      for (int nn = 0; nn < 4; ++nn) {
        for (int kk = 0; kk < 4; ++kk) {
          const bool real = (nb + nn < N) && (kb + kk < K);
          const std::int8_t expect =
              real ? qt.packed[(nb + nn) * K + (kb + kk)] : std::int8_t{0};
          EXPECT_EQ(tile[nn * 4 + kk], expect)
              << "nb=" << nb << " kb=" << kb << " nn=" << nn << " kk=" << kk;
        }
      }
    }
  }
}

TEST(QuantPack, BuildArmCacheIdempotent) {
  const int N = 7, K = 13;
  auto W = RandomVec(N * K, 0xa402);
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), N, K, &qt);
  esm::quant::BuildArmCache(&qt);
  auto packed_arm_1 = qt.packed_arm;
  esm::quant::BuildArmCache(&qt);
  EXPECT_EQ(qt.packed_arm, packed_arm_1);
}

TEST(QuantPack, BuildI8mmCachePopulatesPacked) {
  // SMMLA tiles: each 2-col pair holds K_pad8/8 tiles of 16 bytes laid out
  // [n0:k0..k7, n1:k0..k7]. N_pad2 rounds N up to 2, K_pad8 rounds K to 8.
  const int N = 19, K = 37;  // N tail (% 2 != 0) and K tail (% 8 != 0)
  auto W = RandomVec(N * K, 0xa403);
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), N, K, &qt);
  esm::quant::BuildI8mmCache(&qt);  // pure C++, callable on any host
  const int K_pad8 = (K + 7) & ~7;
  const int N_pad2 = (N + 1) & ~1;
  ASSERT_EQ(static_cast<int>(qt.packed_arm_i8mm.size()), N_pad2 * K_pad8);
  for (int nb = 0; nb < N_pad2; nb += 2) {
    const std::int8_t* pair = qt.packed_arm_i8mm.data() + nb * K_pad8;
    for (int kb = 0; kb < K_pad8; kb += 8) {
      const std::int8_t* tile = pair + kb * 2;
      for (int nn = 0; nn < 2; ++nn) {
        for (int kk = 0; kk < 8; ++kk) {
          const bool real = (nb + nn < N) && (kb + kk < K);
          const std::int8_t expect =
              real ? qt.packed[(nb + nn) * K + (kb + kk)] : std::int8_t{0};
          EXPECT_EQ(tile[nn * 8 + kk], expect)
              << "nb=" << nb << " kb=" << kb << " nn=" << nn << " kk=" << kk;
        }
      }
    }
  }
}
#endif  // aarch64

TEST(QuantPack, AllZeroRowGetsZeroScaleAndZeroPacked) {
  const int N = 2, K = 5;
  std::vector<float> W(N * K, 0.0f);
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), N, K, &qt);
  for (int n = 0; n < N; ++n) {
    EXPECT_EQ(qt.per_channel_scales[n], 0.0f);
    for (int k = 0; k < K; ++k) EXPECT_EQ(qt.packed[n * K + k], 0);
  }
}

TEST(QuantPack, IntermediateValuesRoundCorrectly) {
  // Hand-rolled: row of values where we know exactly what rounds to what.
  // max_abs = 1.0; scale = 1/127. Then 0.5 should round to round(0.5 * 127) = 64 (banker's or
  // standard half-away-from-zero round; either way, 63 or 64). 1.0 -> 127. -1.0 -> -127.
  std::vector<float> W = {1.0f, -1.0f, 0.5f, -0.5f, 0.0f, 0.25f, -0.25f, 0.99f};
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), 1, static_cast<int>(W.size()), &qt);
  EXPECT_NEAR(qt.per_channel_scales[0], 1.0f / 127.0f, 1e-9f);
  EXPECT_EQ(qt.packed[0], 127);
  EXPECT_EQ(qt.packed[1], -127);
  EXPECT_EQ(qt.packed[4], 0);
  // Half-rounding deltas: |q - x*127| < 1 for every element.
  for (std::size_t i = 0; i < W.size(); ++i) {
    EXPECT_LE(std::fabs(static_cast<float>(qt.packed[i]) - W[i] * 127.0f), 1.0f);
  }
}

TEST(QuantPack, ClampsAt127Not128) {
  // Symmetric range avoids -128 so |min| == max. Verify clamping.
  std::vector<float> W = {-1.0f, 1.0f};  // max_abs = 1, scale = 1/127
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), 1, 2, &qt);
  EXPECT_GE(qt.packed[0], -127);
  EXPECT_LE(qt.packed[1], 127);
}

TEST(LinearInt8Ref, MatchesFp32LinearWithinPerChannelStep) {
  // For each output column n, the error vs FP32 Linear is bounded by
  // sum_k |A[m, k]| * |W[n, k] - reconstructed|. We use random [-1, 1]
  // inputs so the typical per-element error is ~ scale[n]/2 and the
  // accumulated bound is roughly scale[n] * sqrt(K) (cancellation).
  const int M = 5, N = 9, K = 17;
  auto A = RandomVec(M * K, 0xa101);
  auto W = RandomVec(N * K, 0xa102);
  auto bias = RandomVec(N, 0xa103);
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), N, K, &qt);
  std::vector<float> ref(M * N), got(M * N);
  esm::kernels::LinearRef(A.data(), W.data(), bias.data(), ref.data(), M, N, K);
  esm::kernels::LinearInt8Ref(A.data(), qt, bias.data(), got.data(), M, N, K);
  // Bound: per output element, error <= sum_k |A[m,k]| * scale[n]. For
  // |A| <= 1 and our test, that's K * scale[n] worst case.
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float bound = static_cast<float>(K) * qt.per_channel_scales[n];
      float diff = std::fabs(ref[m * N + n] - got[m * N + n]);
      EXPECT_LE(diff, bound + 1e-5f) << "m=" << m << " n=" << n
                                       << " bound=" << bound << " diff=" << diff;
    }
  }
}

TEST(LinearInt8Ref, MatchesFp32LinearOnEsmShapeSweep) {
  // Smaller subset of the GEMM shape sweep — enough to exercise tail
  // handling without taking minutes on the scalar reference.
  struct Shape { int M, N, K; };
  const Shape shapes[] = {
      {1, 1, 1}, {3, 5, 7}, {17, 33, 19}, {64, 320, 320},
  };
  for (const auto& s : shapes) {
    auto A = RandomVec(s.M * s.K, 0xb101);
    auto W = RandomVec(s.N * s.K, 0xb102);
    auto bias = RandomVec(s.N, 0xb103);
    esm::quant::QuantizedTensor qt;
    esm::quant::Quantize(W.data(), s.N, s.K, &qt);
    std::vector<float> ref(s.M * s.N), got(s.M * s.N);
    esm::kernels::LinearRef(A.data(), W.data(), bias.data(), ref.data(),
                             s.M, s.N, s.K);
    esm::kernels::LinearInt8Ref(A.data(), qt, bias.data(), got.data(),
                                 s.M, s.N, s.K);
    for (int i = 0; i < s.M * s.N; ++i) {
      // K * max_scale is a loose-but-safe upper bound.
      float max_scale = 0.0f;
      for (auto s2 : qt.per_channel_scales) max_scale = std::max(max_scale, s2);
      float bound = static_cast<float>(s.K) * max_scale;
      EXPECT_LE(std::fabs(ref[i] - got[i]), bound + 1e-3f)
          << "shape=[" << s.M << "," << s.N << "," << s.K << "] i=" << i;
    }
  }
}
