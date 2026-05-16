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
  for (const auto& s : EsmShapes()) {
    // BLAS FMA reordering produces relative drift on the order of
    // eps * sqrt(K) per element. 1e-4 comfortably covers K up to ~10^4
    // with random [-1, 1] inputs and is still tighter than the Phase 0
    // hidden-state envelope of rtol=1e-3.
    RunShape("neon", s, /*with_bias=*/false, 1e-4f);
    RunShape("neon", s, /*with_bias=*/true, 1e-4f);
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
      // M=6 / M=12 / M=18: exactly 1 / 2 / 3 full row-blocks
      {6, 64, 64}, {12, 96, 128}, {18, 128, 128},
      // N not multiple of 32 -> Goto routes whole call to legacy
      {18, 17, 64}, {64, 33, 320}, {64, 320, 17},
      // K-tail (K % 4 != 0): exercises K-tail in both kernels
      {64, 32, 13}, {64, 64, 19},
      // ESM-2-8M shapes (d=320, ffn=1280) — main 6-row blocks
      {32, 320, 320}, {32, 1280, 320}, {32, 320, 1280},
      // 650M-shaped (d=1280, ffn=5120) — small M to keep test fast
      {18, 1280, 1280}, {18, 5120, 1280}, {18, 1280, 5120},
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
#endif  // x86_64
