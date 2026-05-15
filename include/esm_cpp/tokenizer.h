#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace esm {

// ESM-2 tokenizer matching `facebook/esm2_t6_8M_UR50D` byte-exactly.
//
// The 33-token vocabulary is hardcoded (UR50 frequency order; see SPEC.md §2
// and research-report.md §1). The encoder mimics HuggingFace EsmTokenizer:
//   - ASCII whitespace is stripped.
//   - Each vocab character ("L A G V S E R T I D P K Q N F Y M H W C X B U
//     Z O . -") becomes its own token.
//   - The six multi-char specials ("<cls>", "<pad>", "<eos>", "<unk>",
//     "<null_1>", "<mask>") are matched greedily when present in the text.
//   - Any run of non-vocab non-whitespace characters between matches
//     collapses to a single <unk>.
class Tokenizer {
 public:
  static constexpr int32_t kClsId = 0;
  static constexpr int32_t kPadId = 1;
  static constexpr int32_t kEosId = 2;
  static constexpr int32_t kUnkId = 3;
  static constexpr int32_t kMaskId = 32;
  static constexpr int32_t kVocabSize = 33;
  // 1022 residues + <cls> + <eos>. Enforced when truncate=true.
  static constexpr size_t kModelMaxLength = 1024;

  Tokenizer();

  // Encode a protein sequence to token IDs.
  //   add_special: prepend <cls> and append <eos>.
  //   truncate:    if true, clamp output to kModelMaxLength tokens,
  //                preserving <cls> at the front and <eos> at the back
  //                when add_special is true. Set false for byte-exact
  //                parity with HF EsmTokenizer's default (no truncation).
  std::vector<int32_t> Encode(std::string_view text,
                              bool add_special = true,
                              bool truncate = true) const;

  // Decode token IDs back to a string. Specials render as their tag form
  // ("<cls>", "<mask>", etc.) unless skip_special_tokens is true.
  std::string Decode(std::span<const int32_t> ids,
                     bool skip_special_tokens = false) const;

  // Look up a token string by id. Returns "" for out-of-range.
  std::string_view IdToToken(int32_t id) const;

  // Look up an id by exact token string. Returns kUnkId for unknown.
  int32_t TokenToId(std::string_view token) const;

  static constexpr int VocabSize() { return kVocabSize; }

 private:
  // ASCII char -> token id; -1 if not a single-char vocab token.
  std::array<int32_t, 128> char_to_id_{};
  std::array<std::string, kVocabSize> id_to_token_{};
  // Multi-char specials, sorted longest-first for greedy matching.
  std::vector<std::pair<std::string, int32_t>> multi_char_tokens_;
};

}  // namespace esm
