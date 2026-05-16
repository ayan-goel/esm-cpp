#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace esm {

// View over a packed batch: B sequences concatenated along the token axis.
// `cu_seqlens` has size B+1; the b-th sequence occupies token indices
// [cu_seqlens[b], cu_seqlens[b+1]) of `packed_ids`.
//
// Layout matches the cu_seqlens convention consumed by AttentionVarlen
// and RopeApplyVarlenRef. BatchView owns no storage; callers must keep
// the underlying buffers alive for the view's lifetime.
struct BatchView {
  std::span<const std::int32_t> packed_ids;
  std::span<const std::int32_t> packed_masks;  // empty if all tokens are real
  std::span<const std::int32_t> cu_seqlens;
  int batch_size;

  // Validates: batch_size >= 1, cu_seqlens.size() == batch_size + 1,
  // cu_seqlens[0] == 0, monotonic non-decreasing, cu_seqlens.back() ==
  // packed_ids.size(), masks empty or same size as packed_ids. Throws
  // std::invalid_argument on any failure.
  BatchView(std::span<const std::int32_t> ids,
            std::span<const std::int32_t> masks,
            std::span<const std::int32_t> seqlens, int batch);

  std::size_t total_tokens() const { return packed_ids.size(); }

  int sequence_length(int b) const {
    return cu_seqlens[static_cast<std::size_t>(b + 1)] -
           cu_seqlens[static_cast<std::size_t>(b)];
  }
};

}  // namespace esm
