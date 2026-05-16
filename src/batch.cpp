#include "esm_cpp/batch.h"

#include <stdexcept>
#include <string>

namespace esm {

BatchView::BatchView(std::span<const std::int32_t> ids,
                     std::span<const std::int32_t> masks,
                     std::span<const std::int32_t> seqlens, int batch)
    : packed_ids(ids), packed_masks(masks), cu_seqlens(seqlens),
      batch_size(batch) {
  if (batch_size < 1) {
    throw std::invalid_argument("BatchView: batch_size must be >= 1");
  }
  if (cu_seqlens.size() !=
      static_cast<std::size_t>(batch_size) + 1) {
    throw std::invalid_argument(
        "BatchView: cu_seqlens.size() (" +
        std::to_string(cu_seqlens.size()) +
        ") must equal batch_size + 1 (" +
        std::to_string(batch_size + 1) + ")");
  }
  if (cu_seqlens[0] != 0) {
    throw std::invalid_argument("BatchView: cu_seqlens[0] must be 0");
  }
  for (std::size_t i = 1; i < cu_seqlens.size(); ++i) {
    if (cu_seqlens[i] < cu_seqlens[i - 1]) {
      throw std::invalid_argument(
          "BatchView: cu_seqlens must be non-decreasing");
    }
  }
  if (static_cast<std::size_t>(cu_seqlens.back()) != packed_ids.size()) {
    throw std::invalid_argument(
        "BatchView: cu_seqlens.back() (" +
        std::to_string(cu_seqlens.back()) +
        ") must equal packed_ids.size() (" +
        std::to_string(packed_ids.size()) + ")");
  }
  if (!packed_masks.empty() &&
      packed_masks.size() != packed_ids.size()) {
    throw std::invalid_argument(
        "BatchView: packed_masks must be empty or match packed_ids size");
  }
}

}  // namespace esm
