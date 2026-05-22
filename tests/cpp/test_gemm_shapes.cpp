// Shape-sweep correctness for the Linear facade across every registered ISA.
// Every (M, N, K) combination is matmul'd via Linear() under each forced ISA
// and compared against LinearRef. Tolerance is loose-enough for non-Ref FMA
// reordering but tight enough that any algorithm bug stands out.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "esm_cpp/cpu_features.h"
#include "esm_cpp/kernels.h"
#include "esm_cpp/quant.h"

namespace {

class ForceIsa {
 public:
  explicit ForceIsa(const char* value) {
    prev_ = std::getenv("ESM_FORCE_ISA");
    if (prev_) prev_copy_ = prev_;
    if (value) {
      ::setenv("ESM_FORCE_ISA", value, 1);
    } else {
      ::unsetenv("ESM_FORCE_ISA");
    }
  }
  ~ForceIsa() {
    if (prev_) {
      ::setenv("ESM_FORCE_ISA", prev_copy_.c_str(), 1);
    } else {
      ::unsetenv("ESM_FORCE_ISA");
    }
  }

 private:
  const char* prev_;
  std::string prev_copy_;
};

struct Shape {
  int M, N, K;
};

std::vector<float> RandomVec(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

void RunShape(const char* isa, const Shape& s, bool with_bias, float tol) {
  ForceIsa guard(isa);
  const std::size_t Msz = static_cast<std::size_t>(s.M);
  const std::size_t Nsz = static_cast<std::size_t>(s.N);
  const std::size_t Ksz = static_cast<std::size_t>(s.K);
  auto A = RandomVec(Msz * Ksz, 0x1234);
  auto W = RandomVec(Nsz * Ksz, 0x5678);
  auto bias = with_bias ? RandomVec(Nsz, 0x9abc) : std::vector<float>{};
  std::vector<float> ref(Msz * Nsz);
  std::vector<float> got(Msz * Nsz);
  const float* bias_ptr = with_bias ? bias.data() : nullptr;
  esm::kernels::LinearRef(A.data(), W.data(), bias_ptr, ref.data(), s.M, s.N, s.K);
  esm::kernels::Linear(A.data(), W.data(), bias_ptr, got.data(), s.M, s.N, s.K);
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float a = ref[i];
    const float b = got[i];
    const float abs_diff = std::fabs(a - b);
    const float ref_mag = std::fabs(a);
    const bool within = abs_diff <= tol * (1.0f + ref_mag);
    ASSERT_TRUE(within) << "isa=" << isa << " shape=[" << s.M << "," << s.N
                        << "," << s.K << "] bias=" << with_bias
                        << " i=" << i << " ref=" << a << " got=" << b
                        << " diff=" << abs_diff;
  }
}

const std::vector<Shape>& EsmShapes() {
  // Selection covers the shapes the forward graph actually exercises plus
  // a few small/non-power-of-two cases to catch tail-handling bugs in
  // packing kernels (Slice 3 SIMD paths inherit these tests).
  static const std::vector<Shape> shapes = {
      // tiny + odd dims
      {1, 1, 1},
      {3, 5, 7},
      {17, 33, 19},
      // ESM-2-8M shapes (d=320, dh=16, ffn=1280)
      {64, 320, 320},     // q/k/v/out_proj
      {64, 1280, 320},    // fc1
      {64, 320, 1280},    // fc2
      {64, 33, 320},      // lm_head decoder
      // 35M shapes (d=480, dh=24, ffn=1920)
      {128, 480, 480},
      {128, 1920, 480},
      {128, 480, 1920},
      // 650M-shaped (d=1280, ffn=5120) — small M to stay fast
      {16, 1280, 1280},
      {16, 5120, 1280},
      {16, 1280, 5120},
  };
  return shapes;
}

}  // namespace

TEST(GemmShapes, RefDispatchMatchesRef) {
  for (const auto& s : EsmShapes()) {
    RunShape("ref", s, /*with_bias=*/false, 0.0f);
    RunShape("ref", s, /*with_bias=*/true, 0.0f);
  }
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(GemmShapes, NeonDispatchMatchesRefWithinFmaTolerance) {
  // All three ARM tiers route FP32 Linear to the same NEON FMLA kernel.
  for (const char* isa : {"neon", "neondotprod", "neoni8mm"}) {
    for (const auto& s : EsmShapes()) {
      // FMLA reordering produces relative drift on the order of
      // eps * sqrt(K) per element. 1e-4 comfortably covers K up to ~10^4
      // with random [-1, 1] inputs and is still tighter than the Phase 0
      // hidden-state envelope of rtol=1e-3.
      RunShape(isa, s, /*with_bias=*/false, 1e-4f);
      RunShape(isa, s, /*with_bias=*/true, 1e-4f);
    }
  }
}
#endif

#if defined(__x86_64__) || defined(_M_X64)
namespace {
bool HostHasAvx512Vnni() {
  const esm::Isa isa = esm::HostIsa();
  return isa == esm::Isa::Avx512Vnni || isa == esm::Isa::Amx;
}

// Shapes that span the 6×32 multi-accumulator microkernel path:
//   - Several with M >= 12 so multiple full 6-row blocks run
//   - Several with M < 6 so the M-tail fallback runs
//   - N that is a multiple of 32 (production path) and one that isn't
//     (legacy fallback inside the Goto kernel)
//   - K-tail (K % 4 != 0)
const std::vector<Shape>& Int8Shapes() {
  static const std::vector<Shape> shapes = {
      // tiny / odd: M < 6 forces the M-tail path
      {1, 32, 32}, {3, 32, 32}, {5, 32, 64},
      // M=6 / M=12 / M=18: exactly 1 / 2 / 3 full Goto row-blocks
      {6, 64, 64}, {12, 96, 128}, {18, 128, 128},
      // N not multiple of 32 -> Goto routes whole call to legacy
      {18, 17, 64}, {64, 33, 320}, {64, 320, 17},
      // K-tail (K % 4 != 0): exercises K-tail in both kernels
      {64, 32, 13}, {64, 64, 19},
      // ESM-2-8M shapes (d=320, ffn=1280) — main 6-row blocks
      {32, 320, 320}, {32, 1280, 320}, {32, 320, 1280},
      // 650M-shaped (d=1280, ffn=5120) — small M to keep test fast
      {18, 1280, 1280}, {18, 5120, 1280}, {18, 1280, 5120},
      // AMX-amenable: M ≥ 32, N % 32 == 0, K % 64 == 0. These exercise
      // the TDPBUSD 32×32 microkernel when host_isa == amx; otherwise
      // they still pass via VNNI fallback.
      {32, 64, 64}, {32, 64, 128}, {64, 128, 64},
      {64, 64, 320}, {32, 1280, 1280},
      // AMX gate near-miss: M=33 hits AMX-main on [0, 32) then VNNI on
      // M-tail [32, 33). K=66 not multiple of 64 → falls back entirely.
      {33, 64, 64}, {32, 64, 66},
  };
  return shapes;
}

std::vector<float> RandomVecFixed(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

void RunInt8Shape(const Shape& s, bool with_bias) {
  ForceIsa guard("amx");
  const std::size_t Msz = static_cast<std::size_t>(s.M);
  const std::size_t Nsz = static_cast<std::size_t>(s.N);
  const std::size_t Ksz = static_cast<std::size_t>(s.K);
  auto A = RandomVecFixed(Msz * Ksz, 0x2468);
  auto W = RandomVecFixed(Nsz * Ksz, 0x1357);
  auto bias = with_bias ? RandomVecFixed(Nsz, 0xACE1) : std::vector<float>{};
  const float* bias_ptr = with_bias ? bias.data() : nullptr;
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), s.N, s.K, &qt);
  std::vector<float> ref(Msz * Nsz);
  std::vector<float> got(Msz * Nsz);
  esm::kernels::LinearInt8Ref(A.data(), qt, bias_ptr, ref.data(),
                                s.M, s.N, s.K);
  esm::kernels::LinearInt8(A.data(), qt, bias_ptr, got.data(),
                             s.M, s.N, s.K);
  // INT8 cross-check tolerance (per CLAUDE.md): rtol=1e-3 atol=1. Use
  // atol=1 directly since K may be small and per-element error is bounded
  // by the per-channel quantization step, not the dynamic range.
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float diff = std::fabs(ref[i] - got[i]);
    ASSERT_LE(diff, 1.0f + 1e-3f * std::fabs(ref[i]))
        << "shape=[" << s.M << "," << s.N << "," << s.K << "] bias="
        << with_bias << " i=" << i << " ref=" << ref[i] << " got=" << got[i];
  }
}

}  // namespace

TEST(GemmShapes, Avx512VnniInt8DispatchMatchesRefAcrossGotoMicrokernel) {
  if (!HostHasAvx512Vnni()) GTEST_SKIP() << "host lacks AVX-512 VNNI";
  for (const auto& s : Int8Shapes()) {
    RunInt8Shape(s, /*with_bias=*/false);
    RunInt8Shape(s, /*with_bias=*/true);
  }
}

namespace {
bool HostHasAvx512() {
  const esm::Isa isa = esm::HostIsa();
  return isa == esm::Isa::Avx512 || isa == esm::Isa::Avx512Vnni ||
         isa == esm::Isa::Amx;
}
}  // namespace

// FP32 AVX-512 Linear cross-check. Until this kernel landed, the FP32
// path delegated to LinearRef; on the gate machine all of lm_head's
// FP32 GEMMs ran scalar single-threaded, costing >50 % of the 650M
// forward. This test guards the new multi-accumulator implementation.
TEST(GemmShapes, Avx512Fp32DispatchMatchesRefAcrossEsmShapes) {
  if (!HostHasAvx512()) GTEST_SKIP() << "host lacks AVX-512";
  // Cover the shapes lm_head actually drives plus a few tail/N-tail
  // cases: N not multiple of 8 hits the per-output fallback, K not
  // multiple of 16 hits the masked tail.
  const std::vector<Shape> shapes = {
      // lm_head shapes (M=B*L, N=d, K=d) and (M=B*L, N=V=33, K=d)
      {2048, 1280, 1280},  // 650M lm_dense
      {2048, 33, 1280},    // 650M lm_decoder
      {2048, 640, 640},    // 150M lm_dense
      {2048, 33, 640},     // 150M lm_decoder
      {800, 320, 320},     // 8M lm_dense (B=8, L=100)
      {800, 33, 320},      // 8M lm_decoder
      // Tail cases: N not multiple of 8, K not multiple of 16
      {17, 11, 19},
      {32, 9, 32},
      {16, 17, 64},
  };
  for (const auto& s : shapes) {
    // FP32 FMA reordering: 1e-4 covers K up to ~10^4 against rand[-1, 1].
    // Force "avx512vnni" so kernels::Linear routes to LinearAvx512 even
    // when host_isa is amx (the Avx512 / Avx512Vnni / Amx switch arms
    // are all aliases here).
    RunShape("avx512vnni", s, /*with_bias=*/false, 1e-4f);
    RunShape("avx512vnni", s, /*with_bias=*/true, 1e-4f);
  }
}
#endif  // x86_64

#if defined(__aarch64__) || defined(_M_ARM64)
namespace {

// Shapes spanning the NEON SDOT microkernel paths:
//   - M >= 4 multiples (full 4-row blocks) and M < 4 (M-tail)
//   - N multiple of 16 (4x16 hot path), N multiple of 4 not 16 (panel mop-up),
//     and N not a multiple of 4 (partial last panel, e.g. lm_head's 33)
//   - K not a multiple of 4 (K-tail with zero-padded weights)
const std::vector<Shape>& NeonInt8Shapes() {
  static const std::vector<Shape> shapes = {
      {1, 4, 4},     {3, 8, 8},      {4, 16, 16},   {8, 32, 32},
      {5, 16, 64},   {4, 20, 128},   {7, 33, 64},   {64, 32, 13},
      {64, 17, 19},  {64, 33, 320},
      // ESM-2-8M shapes (d=320, ffn=1280)
      {32, 320, 320}, {32, 1280, 320}, {32, 320, 1280},
      // 650M-shaped (d=1280, ffn=5120) — small M to keep the test fast
      {18, 1280, 1280}, {18, 5120, 1280}, {18, 1280, 5120},
  };
  return shapes;
}

void RunNeonInt8Shape(const Shape& s, bool with_bias) {
  ForceIsa guard("neondotprod");
  const std::size_t Msz = static_cast<std::size_t>(s.M);
  const std::size_t Nsz = static_cast<std::size_t>(s.N);
  const std::size_t Ksz = static_cast<std::size_t>(s.K);
  auto A = RandomVec(Msz * Ksz, 0x2468);
  auto W = RandomVec(Nsz * Ksz, 0x1357);
  auto bias = with_bias ? RandomVec(Nsz, 0xACE1) : std::vector<float>{};
  const float* bias_ptr = with_bias ? bias.data() : nullptr;
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), s.N, s.K, &qt);
  std::vector<float> ref(Msz * Nsz);
  std::vector<float> got(Msz * Nsz);
  esm::kernels::LinearInt8Ref(A.data(), qt, bias_ptr, ref.data(), s.M, s.N, s.K);
  esm::kernels::LinearInt8(A.data(), qt, bias_ptr, got.data(), s.M, s.N, s.K);
  // INT8 cross-check tolerance (CLAUDE.md): rtol=1e-3 atol=1. Never widened.
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float diff = std::fabs(ref[i] - got[i]);
    ASSERT_LE(diff, 1.0f + 1e-3f * std::fabs(ref[i]))
        << "shape=[" << s.M << "," << s.N << "," << s.K << "] bias=" << with_bias
        << " i=" << i << " ref=" << ref[i] << " got=" << got[i];
  }
}

}  // namespace

TEST(GemmShapes, NeonDotProdInt8DispatchMatchesRef) {
  for (const auto& s : NeonInt8Shapes()) {
    RunNeonInt8Shape(s, /*with_bias=*/false);
    RunNeonInt8Shape(s, /*with_bias=*/true);
  }
}

namespace {
bool HostHasI8mm() { return esm::HostIsa() == esm::Isa::NeonI8mm; }

void RunNeonI8mmShape(const Shape& s, bool with_bias) {
  ForceIsa guard("neoni8mm");
  const std::size_t Msz = static_cast<std::size_t>(s.M);
  const std::size_t Nsz = static_cast<std::size_t>(s.N);
  const std::size_t Ksz = static_cast<std::size_t>(s.K);
  auto A = RandomVec(Msz * Ksz, 0x2468);
  auto W = RandomVec(Nsz * Ksz, 0x1357);
  auto bias = with_bias ? RandomVec(Nsz, 0xACE1) : std::vector<float>{};
  const float* bias_ptr = with_bias ? bias.data() : nullptr;
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), s.N, s.K, &qt);  // builds SMMLA cache (forced)
  std::vector<float> ref(Msz * Nsz);
  std::vector<float> got(Msz * Nsz);
  esm::kernels::LinearInt8Ref(A.data(), qt, bias_ptr, ref.data(), s.M, s.N, s.K);
  esm::kernels::LinearInt8(A.data(), qt, bias_ptr, got.data(), s.M, s.N, s.K);
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float diff = std::fabs(ref[i] - got[i]);
    ASSERT_LE(diff, 1.0f + 1e-3f * std::fabs(ref[i]))
        << "shape=[" << s.M << "," << s.N << "," << s.K << "] bias=" << with_bias
        << " i=" << i << " ref=" << ref[i] << " got=" << got[i];
  }
}
}  // namespace

TEST(GemmShapes, NeonI8mmInt8DispatchMatchesRef) {
  if (!HostHasI8mm()) GTEST_SKIP() << "host lacks FEAT_I8MM";
  for (const auto& s : NeonInt8Shapes()) {
    RunNeonI8mmShape(s, /*with_bias=*/false);
    RunNeonI8mmShape(s, /*with_bias=*/true);
  }
}
#endif  // aarch64
