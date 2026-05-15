// AttentionVarlen takes packed [T, H, dh] layout plus cu_seqlens[B+1].
// Compared against the existing [H, L, dh] AttentionRef on equivalent
// inputs, B=1 must produce bitwise-identical output (same math, same
// summation order, layout differences are pure address arithmetic).
// B=2 packed cases verify per-sequence isolation: padding/garbage in
// one sequence's KV range cannot bleed into another's output.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "esm_cpp/kernels.h"

namespace {

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
