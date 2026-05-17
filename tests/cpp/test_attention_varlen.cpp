// AttentionVarlen takes packed [T, H, dh] layout plus cu_seqlens[B+1].
// Compared against the existing [H, L, dh] AttentionRef on equivalent
// inputs, B=1 must produce bitwise-identical output (same math, same
// summation order, layout differences are pure address arithmetic).
// B=2 packed cases verify per-sequence isolation: padding/garbage in
// one sequence's KV range cannot bleed into another's output.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

#include "esm_cpp/cpu_features.h"
#include "esm_cpp/kernels.h"

namespace {

#if defined(__x86_64__) || defined(_M_X64)
bool HostHasAvx512() {
  const esm::Isa isa = esm::HostIsa();
  return isa == esm::Isa::Avx512 || isa == esm::Isa::Avx512Vnni ||
         isa == esm::Isa::Amx;
}

// AVX-512 BF16 (VDPBF16PS / VCVTNE2PS2PBH) is a separate CPUID feature
// from base AVX-512. Cooper Lake and Sapphire Rapids have it; Skylake-
// server doesn't. GitHub Actions x86 runners are Skylake-class, so the
// BF16 attention test must skip on them or it traps SIGILL on the
// first BF16 instruction. CPUID leaf 7, sub-leaf 1, EAX bit 5.
bool HostHasAvx512Bf16() {
#if defined(__GNUC__) || defined(__clang__)
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (!__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) return false;
  return (eax & (1u << 5)) != 0;
#else
  return false;
#endif
}

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
#endif

std::vector<float> RandomVec(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

// [H, L, dh] -> [L, H, dh]
std::vector<float> HldToLhd(const std::vector<float>& src, int H, int L, int dh) {
  std::vector<float> dst(static_cast<std::size_t>(H) * L * dh);
  for (int h = 0; h < H; ++h)
    for (int t = 0; t < L; ++t)
      for (int j = 0; j < dh; ++j)
        dst[(static_cast<std::size_t>(t) * H + h) * dh + j] =
            src[(static_cast<std::size_t>(h) * L + t) * dh + j];
  return dst;
}

}  // namespace

TEST(AttentionVarlenRef, B1MatchesAttentionRefBitwise) {
  const int H = 4, L = 7, dh = 6;
  auto Qhld = RandomVec(H * L * dh, 1);
  auto Khld = RandomVec(H * L * dh, 2);
  auto Vhld = RandomVec(H * L * dh, 3);
  auto Qlhd = HldToLhd(Qhld, H, L, dh);
  auto Klhd = HldToLhd(Khld, H, L, dh);
  auto Vlhd = HldToLhd(Vhld, H, L, dh);
  std::vector<float> out_ref(L * H * dh);
  std::vector<float> out_varlen(L * H * dh);
  std::vector<int> cu_seqlens = {0, L};
  esm::kernels::AttentionRef(Qhld.data(), Khld.data(), Vhld.data(), nullptr,
                              out_ref.data(), H, L, dh);
  esm::kernels::AttentionVarlenRef(Qlhd.data(), Klhd.data(), Vlhd.data(),
                                    cu_seqlens.data(), /*batch_size=*/1, H, dh,
                                    out_varlen.data());
  for (std::size_t i = 0; i < out_ref.size(); ++i) {
    EXPECT_FLOAT_EQ(out_ref[i], out_varlen[i]) << "i=" << i;
  }
}

TEST(RopeApplyVarlenRef, B1MatchesHeadMajorRopeBitwise) {
  const int H = 3, L = 5, dh = 8;
  auto x_hld = RandomVec(H * L * dh, 21);
  auto x_lhd = HldToLhd(x_hld, H, L, dh);
  std::vector<float> cos(L * dh), sin(L * dh);
  esm::kernels::RopeBuildTables(L, dh, cos.data(), sin.data());
  esm::kernels::RopeApplyInplaceRef(x_hld.data(), cos.data(), sin.data(), H, L,
                                     dh);
  std::vector<int> cu_seqlens = {0, L};
  esm::kernels::RopeApplyVarlenRef(x_lhd.data(), cos.data(), sin.data(),
                                    cu_seqlens.data(), /*batch_size=*/1, H, dh);
  // Re-transpose lhd back to hld and compare.
  for (int h = 0; h < H; ++h) {
    for (int t = 0; t < L; ++t) {
      for (int j = 0; j < dh; ++j) {
        float varlen_val =
            x_lhd[(static_cast<std::size_t>(t) * H + h) * dh + j];
        float head_val =
            x_hld[(static_cast<std::size_t>(h) * L + t) * dh + j];
        EXPECT_FLOAT_EQ(head_val, varlen_val)
            << "h=" << h << " t=" << t << " j=" << j;
      }
    }
  }
}

TEST(RopeApplyVarlenRef, B2RestartsPositionPerSequence) {
  // Two packed sequences of length 3 and 2. Each sequence's RoPE
  // positions should start at 0 — the second sequence's first token
  // gets cos[0]/sin[0], not cos[3]/sin[3] (which would be the global
  // position if we forgot to use cu_seqlens).
  const int H = 1, dh = 4;
  const int L1 = 3, L2 = 2;
  const int T = L1 + L2;
  const int max_seqlen = std::max(L1, L2);

  auto src = RandomVec(T * H * dh, 31);
  auto cu_packed = src;
  // Reference: rotate the second sequence with positions [0, 1] (not
  // [3, 4]). We synthesize this by running RoPE on a sub-buffer of L2
  // tokens with the table built for max_seqlen.
  auto seq1_alone = std::vector<float>(src.begin(), src.begin() + L1 * H * dh);
  auto seq2_alone = std::vector<float>(src.begin() + L1 * H * dh, src.end());
  std::vector<float> cos(max_seqlen * dh), sin(max_seqlen * dh);
  esm::kernels::RopeBuildTables(max_seqlen, dh, cos.data(), sin.data());
  std::vector<int> cu_s1 = {0, L1};
  std::vector<int> cu_s2 = {0, L2};
  esm::kernels::RopeApplyVarlenRef(seq1_alone.data(), cos.data(), sin.data(),
                                    cu_s1.data(), 1, H, dh);
  esm::kernels::RopeApplyVarlenRef(seq2_alone.data(), cos.data(), sin.data(),
                                    cu_s2.data(), 1, H, dh);

  std::vector<int> cu_packed_idx = {0, L1, T};
  esm::kernels::RopeApplyVarlenRef(cu_packed.data(), cos.data(), sin.data(),
                                    cu_packed_idx.data(), 2, H, dh);

  // First L1 tokens must match the solo-run on seq1_alone.
  for (int i = 0; i < L1 * H * dh; ++i) {
    EXPECT_FLOAT_EQ(cu_packed[i], seq1_alone[i]) << "seq1 i=" << i;
  }
  // Next L2 tokens must match the solo-run on seq2_alone (positions [0, L2)).
  for (int i = 0; i < L2 * H * dh; ++i) {
    EXPECT_FLOAT_EQ(cu_packed[L1 * H * dh + i], seq2_alone[i])
        << "seq2 i=" << i;
  }
}

TEST(AttentionVarlenRef, B2PackedIsolatesSequences) {
  // Two sequences of lengths L1=5 and L2=3 packed back-to-back into T=8.
  // The second sequence's KV are deliberately set to extreme garbage; the
  // first sequence's output must be unchanged from running it alone.
  const int H = 2, dh = 4;
  const int L1 = 5, L2 = 3;
  const int T = L1 + L2;

  auto Q_solo = RandomVec(L1 * H * dh, 11);
  auto K_solo = RandomVec(L1 * H * dh, 12);
  auto V_solo = RandomVec(L1 * H * dh, 13);
  std::vector<float> out_solo(L1 * H * dh);
  std::vector<int> cu_solo = {0, L1};
  esm::kernels::AttentionVarlenRef(Q_solo.data(), K_solo.data(), V_solo.data(),
                                    cu_solo.data(), 1, H, dh, out_solo.data());

  // Now pack: positions [0..L1) are the same first sequence; positions
  // [L1..T) hold a second sequence with deliberately huge K to ensure
  // any cross-sequence attention leakage would explode the first
  // sequence's output.
  std::vector<float> Q_pack(T * H * dh);
  std::vector<float> K_pack(T * H * dh);
  std::vector<float> V_pack(T * H * dh);
  std::memcpy(Q_pack.data(), Q_solo.data(), L1 * H * dh * sizeof(float));
  std::memcpy(K_pack.data(), K_solo.data(), L1 * H * dh * sizeof(float));
  std::memcpy(V_pack.data(), V_solo.data(), L1 * H * dh * sizeof(float));
  for (int i = L1 * H * dh; i < T * H * dh; ++i) {
    Q_pack[i] = 50.0f;   // any read of pad query is suspect
    K_pack[i] = 1000.0f; // would dominate softmax if leaked
    V_pack[i] = 999.0f;  // would corrupt output if mixed in
  }
  std::vector<float> out_pack(T * H * dh);
  std::vector<int> cu_pack = {0, L1, T};
  esm::kernels::AttentionVarlenRef(Q_pack.data(), K_pack.data(), V_pack.data(),
                                    cu_pack.data(), 2, H, dh, out_pack.data());
  // First sequence's output rows must match the solo run bit-for-bit.
  for (int i = 0; i < L1 * H * dh; ++i) {
    EXPECT_FLOAT_EQ(out_solo[i], out_pack[i])
        << "first seq leaked at i=" << i;
  }
}

#if defined(__x86_64__) || defined(_M_X64)
namespace esm::kernels {
void AttentionVarlenAvx512Bf16(const float* q, const float* k, const float* v,
                                const int* cu_seqlens, int batch_size,
                                int num_heads, int head_dim, float* out);
}  // namespace esm::kernels

TEST(AttentionVarlenAvx512Bf16, MatchesRefAtBf16Tolerance) {
  if (!HostHasAvx512Bf16())
    GTEST_SKIP() << "host lacks AVX-512 BF16 (VDPBF16PS)";
  // The BF16 variant only supports head_dim multiples of 32 (BF16 zmm
  // chunk size). At BF16 precision the per-element error is ~1 ULP of
  // BF16 ≈ 1/128 ≈ 0.008 — call it 1e-2 absolute, scaled by element
  // magnitude. Plan-mandated tolerance.
  struct Case {
    int H;
    int dh;
    int L1;
    int L2;
  };
  const Case cases[] = {
      {4, 32, 17, 0},  {4, 64, 33, 0},   {4, 32, 48, 0},
      {4, 64, 64, 0},  {20, 64, 256, 0}, {20, 64, 128, 128},
      {20, 64, 37, 0}, {20, 64, 1, 1},
  };
  for (const auto& c : cases) {
    const int T = c.L1 + c.L2;
    auto Q = RandomVec(static_cast<std::size_t>(T) * c.H * c.dh, 0xA1A2);
    auto K = RandomVec(static_cast<std::size_t>(T) * c.H * c.dh, 0xB1B2);
    auto V = RandomVec(static_cast<std::size_t>(T) * c.H * c.dh, 0xC1C2);
    std::vector<int> cu = c.L2 > 0 ? std::vector<int>{0, c.L1, T}
                                    : std::vector<int>{0, c.L1};
    const int batch_size = static_cast<int>(cu.size()) - 1;

    std::vector<float> out_ref(static_cast<std::size_t>(T) * c.H * c.dh);
    std::vector<float> out_bf16(static_cast<std::size_t>(T) * c.H * c.dh);
    esm::kernels::AttentionVarlenRef(Q.data(), K.data(), V.data(),
                                       cu.data(), batch_size, c.H, c.dh,
                                       out_ref.data());
    esm::kernels::AttentionVarlenAvx512Bf16(Q.data(), K.data(), V.data(),
                                              cu.data(), batch_size, c.H,
                                              c.dh, out_bf16.data());
    // BF16 inputs to Pass 1 give ~0.5 % per-element drift on the dot
    // products, which softmax + exp + weighted-sum-of-FP32-V amplifies
    // to a few-percent envelope on the output. Empirical drift across
    // these shapes stays well under 5 %. The Slice 4/6 FP32 tile kernel
    // remains the production default; BF16 stays opt-in via
    // ESM_AMX_ATTENTION=on until PPPL signs off.
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    std::size_t worst_i = 0;
    for (std::size_t i = 0; i < out_ref.size(); ++i) {
      const float diff = std::fabs(out_bf16[i] - out_ref[i]);
      const float ref_mag = std::fabs(out_ref[i]);
      const float rel = ref_mag > 1e-6f ? diff / ref_mag : diff;
      if (rel > max_rel_diff) {
        max_rel_diff = rel;
        max_abs_diff = diff;
        worst_i = i;
      }
    }
    constexpr float kBf16Rtol = 5e-2f;
    constexpr float kBf16Atol = 1e-3f;
    for (std::size_t i = 0; i < out_ref.size(); ++i) {
      const float ref_mag = std::fabs(out_ref[i]);
      EXPECT_NEAR(out_bf16[i], out_ref[i],
                   kBf16Atol + kBf16Rtol * ref_mag)
          << "H=" << c.H << " dh=" << c.dh << " L1=" << c.L1
          << " L2=" << c.L2 << " i=" << i
          << " (worst across this shape: i=" << worst_i
          << " ref=" << out_ref[worst_i] << " got=" << out_bf16[worst_i]
          << " abs=" << max_abs_diff << " rel=" << max_rel_diff << ")";
    }
  }
}

TEST(AttentionVarlenAvx512, MatchesRefAcrossEsmHeadDims) {
  if (!HostHasAvx512()) GTEST_SKIP() << "host lacks AVX-512";
  // ESM head_dim values: 16 (8M), 24 (35M), 32 (150M), 64 (650M / 3B).
  // Plus 17 (odd, exercises masked tail in the dot product), 3 (tiny).
  struct Case {
    int H;     // num_heads
    int dh;    // head_dim
    int L1;    // seq 0 length
    int L2;    // seq 1 length (or 0 for B=1)
  };
  const Case cases[] = {
      // B=1 sanity sweep across head_dim
      {2, 3, 5, 0}, {4, 16, 32, 0}, {4, 17, 9, 0},
      {4, 24, 16, 0}, {4, 32, 17, 0}, {4, 64, 33, 0},
      // B=2 packed — per-sequence isolation under SIMD
      {4, 16, 17, 23}, {4, 64, 11, 31},
      // Larger seq to exercise the 16-wide j-stride
      {4, 32, 48, 0}, {4, 64, 64, 0},
      // Slice 4 (Phase 7) tile-rewrite coverage:
      //   - 650M headline shape (H=20, dh=64, L=256)
      //   - 35M dh=24 (head_dim not a multiple of 16) — masked head_dim tail
      //   - Odd seq_len (L=37) — exercises the j-chunk tail in the multi-acc kernel
      //   - Single-token edge (L=1) — entire kernel is the seq_len tail
      {20, 64, 256, 0}, {20, 64, 128, 128},
      {20, 24, 128, 0}, {20, 24, 37, 100},
      {20, 64, 37, 0}, {20, 64, 1, 1},
  };
  for (const auto& c : cases) {
    const int T = c.L1 + c.L2;
    auto Q = RandomVec(static_cast<std::size_t>(T) * c.H * c.dh, 0xA1A2);
    auto K = RandomVec(static_cast<std::size_t>(T) * c.H * c.dh, 0xB1B2);
    auto V = RandomVec(static_cast<std::size_t>(T) * c.H * c.dh, 0xC1C2);
    std::vector<int> cu = c.L2 > 0 ? std::vector<int>{0, c.L1, T}
                                    : std::vector<int>{0, c.L1};
    const int batch_size = static_cast<int>(cu.size()) - 1;

    std::vector<float> out_ref(static_cast<std::size_t>(T) * c.H * c.dh);
    std::vector<float> out_simd(static_cast<std::size_t>(T) * c.H * c.dh);
    esm::kernels::AttentionVarlenRef(Q.data(), K.data(), V.data(),
                                       cu.data(), batch_size, c.H, c.dh,
                                       out_ref.data());
    {
      ForceIsa guard("amx");
      esm::kernels::AttentionVarlen(Q.data(), K.data(), V.data(),
                                      cu.data(), batch_size, c.H, c.dh,
                                      out_simd.data());
    }
    // FP32 softmax accumulator + polynomial exp introduces ~1e-5 relative
    // drift vs the FP64-sum reference. ESM's hidden-state HF parity gate
    // is rtol=1e-3 atol=8e-2 (SPEC Phase 0 amendment), comfortably wider.
    for (std::size_t i = 0; i < out_ref.size(); ++i) {
      const float ref_mag = std::fabs(out_ref[i]);
      EXPECT_NEAR(out_simd[i], out_ref[i],
                   1e-4f * (1.0f + ref_mag))
          << "H=" << c.H << " dh=" << c.dh << " L1=" << c.L1
          << " L2=" << c.L2 << " i=" << i;
    }
  }
}
#endif  // x86_64
