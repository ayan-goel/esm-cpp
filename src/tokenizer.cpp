#include "esm_cpp/tokenizer.h"

#include <algorithm>
#include <cassert>
#include <cctype>

namespace esm {

namespace {

// UR50 frequency order. Index in this list is the token id.
constexpr const char* kVocab[] = {
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
    "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-",
    "<null_1>", "<mask>",
};
static_assert(sizeof(kVocab) / sizeof(kVocab[0]) == Tokenizer::kVocabSize,
              "vocab size mismatch");

bool IsAsciiSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

bool StartsWith(std::string_view text, size_t pos, std::string_view needle) {
  if (pos + needle.size() > text.size()) return false;
  return text.compare(pos, needle.size(), needle) == 0;
}

}  // namespace

Tokenizer::Tokenizer() {
  char_to_id_.fill(-1);
  for (int32_t id = 0; id < kVocabSize; ++id) {
    id_to_token_[static_cast<size_t>(id)] = kVocab[id];
    const std::string& tok = id_to_token_[static_cast<size_t>(id)];
    if (tok.size() == 1) {
      char_to_id_[static_cast<unsigned char>(tok[0])] = id;
    } else {
      multi_char_tokens_.emplace_back(tok, id);
    }
  }
  std::sort(multi_char_tokens_.begin(), multi_char_tokens_.end(),
            [](const auto& a, const auto& b) {
              return a.first.size() > b.first.size();
            });
}

std::vector<int32_t> Tokenizer::Encode(std::string_view text, bool add_special,
                                       bool truncate) const {
  std::vector<int32_t> out;
  out.reserve(text.size() + 2);
  if (add_special) out.push_back(kClsId);

  bool unk_pending = false;
  auto flush_unk = [&]() {
    if (unk_pending) {
      out.push_back(kUnkId);
      unk_pending = false;
    }
  };

  size_t i = 0;
  while (i < text.size()) {
    char c = text[i];

    if (IsAsciiSpace(c)) {
      flush_unk();
      ++i;
      continue;
    }

    int32_t matched_id = -1;
    size_t matched_len = 0;

    if (c == '<') {
      for (const auto& [needle, id] : multi_char_tokens_) {
        if (StartsWith(text, i, needle)) {
          matched_id = id;
          matched_len = needle.size();
          break;
        }
      }
    }
    if (matched_len == 0) {
      auto uc = static_cast<unsigned char>(c);
      if (uc < 128) {
        int32_t id = char_to_id_[uc];
        if (id >= 0) {
          matched_id = id;
          matched_len = 1;
        }
      }
    }

    if (matched_len > 0) {
      flush_unk();
      out.push_back(matched_id);
      i += matched_len;
    } else {
      unk_pending = true;
      ++i;
    }
  }
  flush_unk();

  if (add_special) out.push_back(kEosId);

  if (truncate && out.size() > kModelMaxLength) {
    bool put_eos_back = add_special && (out.back() == kEosId);
    out.resize(kModelMaxLength);
    if (put_eos_back) out.back() = kEosId;
  }

  return out;
}

std::string Tokenizer::Decode(std::span<const int32_t> ids,
                              bool skip_special_tokens) const {
  std::string out;
  out.reserve(ids.size());
  for (int32_t id : ids) {
    if (id < 0 || id >= kVocabSize) continue;
    const std::string& tok = id_to_token_[static_cast<size_t>(id)];
    if (skip_special_tokens && tok.size() > 1 && tok.front() == '<') continue;
    out.append(tok);
  }
  return out;
}

std::string_view Tokenizer::IdToToken(int32_t id) const {
  if (id < 0 || id >= kVocabSize) return {};
  return id_to_token_[static_cast<size_t>(id)];
}

int32_t Tokenizer::TokenToId(std::string_view token) const {
  for (int32_t i = 0; i < kVocabSize; ++i) {
    if (id_to_token_[static_cast<size_t>(i)] == token) return i;
  }
  return kUnkId;
}

}  // namespace esm
