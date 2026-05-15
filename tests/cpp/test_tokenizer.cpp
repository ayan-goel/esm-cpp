#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "esm_cpp/tokenizer.h"

using esm::Tokenizer;

namespace {

std::vector<int32_t> Encode(const Tokenizer& t, std::string_view s,
                            bool add_special = true) {
  return t.Encode(s, add_special, /*truncate=*/true);
}

}  // namespace

TEST(Tokenizer, VocabIsUR50FrequencyOrder) {
  Tokenizer t;
  EXPECT_EQ(t.IdToToken(0), "<cls>");
  EXPECT_EQ(t.IdToToken(1), "<pad>");
  EXPECT_EQ(t.IdToToken(2), "<eos>");
  EXPECT_EQ(t.IdToToken(3), "<unk>");
  // First few canonical amino acids — order matters and is NOT alphabetical.
  EXPECT_EQ(t.IdToToken(4), "L");
  EXPECT_EQ(t.IdToToken(5), "A");
  EXPECT_EQ(t.IdToToken(6), "G");
  EXPECT_EQ(t.IdToToken(7), "V");
  // Spot-check rare aa and gap/null/mask near the end.
  EXPECT_EQ(t.IdToToken(24), "X");
  EXPECT_EQ(t.IdToToken(28), "O");
  EXPECT_EQ(t.IdToToken(29), ".");
  EXPECT_EQ(t.IdToToken(30), "-");
  EXPECT_EQ(t.IdToToken(31), "<null_1>");
  EXPECT_EQ(t.IdToToken(32), "<mask>");
}

TEST(Tokenizer, CanonicalSequenceMatchesHfReference) {
  Tokenizer t;
  // HF: tok("MKTGV", add_special_tokens=True) -> [0, 20, 15, 11, 6, 7, 2]
  EXPECT_EQ(Encode(t, "MKTGV"), (std::vector<int32_t>{0, 20, 15, 11, 6, 7, 2}));
}

TEST(Tokenizer, NoSpecialTokens) {
  Tokenizer t;
  EXPECT_EQ(t.Encode("MKTGV", /*add_special=*/false, /*truncate=*/true),
            (std::vector<int32_t>{20, 15, 11, 6, 7}));
}

TEST(Tokenizer, RareAminoAcids) {
  Tokenizer t;
  // HF: tok("XBUZ", add_special) -> [0, 24, 25, 26, 27, 2]
  EXPECT_EQ(Encode(t, "XBUZ"), (std::vector<int32_t>{0, 24, 25, 26, 27, 2}));
  // HF: tok("MKTGVBUZO.X-", add_special) ->
  //   [0, 20, 15, 11, 6, 7, 25, 26, 27, 28, 29, 24, 30, 2]
  EXPECT_EQ(Encode(t, "MKTGVBUZO.X-"),
            (std::vector<int32_t>{0, 20, 15, 11, 6, 7, 25, 26, 27, 28, 29, 24,
                                  30, 2}));
}

TEST(Tokenizer, WhitespaceAndUnknowns) {
  Tokenizer t;
  // HF: tok("M K T G V") == tok("MKTGV")
  EXPECT_EQ(Encode(t, "M K T G V"),
            (std::vector<int32_t>{0, 20, 15, 11, 6, 7, 2}));
  // HF: tok("mktgv") -> [0, 3, 2] (single <unk>, NOT five)
  EXPECT_EQ(Encode(t, "mktgv"), (std::vector<int32_t>{0, 3, 2}));
  // HF: tok("M0X0K") -> [0, 20, 3, 24, 3, 15, 2]
  EXPECT_EQ(Encode(t, "M0X0K"),
            (std::vector<int32_t>{0, 20, 3, 24, 3, 15, 2}));
  // HF: tok("M00K") -> [0, 20, 3, 15, 2] (consecutive non-vocab collapse)
  EXPECT_EQ(Encode(t, "M00K"), (std::vector<int32_t>{0, 20, 3, 15, 2}));
  // HF: tok("M  K") -> [0, 20, 15, 2] (whitespace stripped, no <unk>)
  EXPECT_EQ(Encode(t, "M  K"), (std::vector<int32_t>{0, 20, 15, 2}));
  // HF: tok("MmK") -> [0, 20, 3, 15, 2]
  EXPECT_EQ(Encode(t, "MmK"), (std::vector<int32_t>{0, 20, 3, 15, 2}));
}

TEST(Tokenizer, MultiCharSpecials) {
  Tokenizer t;
  // HF: tok("<mask>") -> [0, 32, 2]
  EXPECT_EQ(Encode(t, "<mask>"), (std::vector<int32_t>{0, 32, 2}));
  // HF: tok("M<mask>K") -> [0, 20, 32, 15, 2]
  EXPECT_EQ(Encode(t, "M<mask>K"),
            (std::vector<int32_t>{0, 20, 32, 15, 2}));
  // HF: tok("<null_1>") -> [0, 31, 2]
  EXPECT_EQ(Encode(t, "<null_1>"), (std::vector<int32_t>{0, 31, 2}));
  // HF: tok("M<>K") -> [0, 20, 3, 15, 2] (partial '<' is <unk>)
  EXPECT_EQ(Encode(t, "M<>K"), (std::vector<int32_t>{0, 20, 3, 15, 2}));
}

TEST(Tokenizer, EmptyAndAllWhitespace) {
  Tokenizer t;
  EXPECT_EQ(Encode(t, ""), (std::vector<int32_t>{0, 2}));
  EXPECT_EQ(Encode(t, "   "), (std::vector<int32_t>{0, 2}));
  EXPECT_EQ(Encode(t, "\t\n"), (std::vector<int32_t>{0, 2}));
}

TEST(Tokenizer, TruncationKeepsClsAndEos) {
  Tokenizer t;
  std::string long_seq(2000, 'M');
  auto ids = t.Encode(long_seq, /*add_special=*/true, /*truncate=*/true);
  ASSERT_EQ(ids.size(), Tokenizer::kModelMaxLength);
  EXPECT_EQ(ids.front(), Tokenizer::kClsId);
  EXPECT_EQ(ids.back(), Tokenizer::kEosId);
  // Interior tokens are all M (id 20).
  for (size_t i = 1; i + 1 < ids.size(); ++i) EXPECT_EQ(ids[i], 20);
}

TEST(Tokenizer, NoTruncationMatchesHf) {
  Tokenizer t;
  std::string long_seq(1100, 'M');
  auto ids = t.Encode(long_seq, /*add_special=*/true, /*truncate=*/false);
  EXPECT_EQ(ids.size(), 1102u);
  EXPECT_EQ(ids.front(), 0);
  EXPECT_EQ(ids.back(), 2);
}

TEST(Tokenizer, DecodeRoundTripWithoutUnks) {
  Tokenizer t;
  std::string seq = "MKTGVAQERSILDPQNFY";
  auto ids = t.Encode(seq, /*add_special=*/false, /*truncate=*/true);
  EXPECT_EQ(t.Decode(ids), seq);
}

TEST(Tokenizer, DecodeSkipSpecials) {
  Tokenizer t;
  auto ids = t.Encode("MKTGV", /*add_special=*/true, /*truncate=*/true);
  EXPECT_EQ(t.Decode(ids, /*skip_special_tokens=*/false), "<cls>MKTGV<eos>");
  EXPECT_EQ(t.Decode(ids, /*skip_special_tokens=*/true), "MKTGV");
}

TEST(Tokenizer, TokenToIdAndIdToToken) {
  Tokenizer t;
  EXPECT_EQ(t.TokenToId("M"), 20);
  EXPECT_EQ(t.TokenToId("<mask>"), 32);
  EXPECT_EQ(t.TokenToId("nonsense"), Tokenizer::kUnkId);
  EXPECT_EQ(t.IdToToken(20), "M");
  EXPECT_EQ(t.IdToToken(999), "");
}
