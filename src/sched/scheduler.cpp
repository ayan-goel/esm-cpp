#include "esm_cpp/scheduler.h"

#include <algorithm>
#include <numeric>

namespace esm {

namespace {

// Recursively plan a sub-range of indices (already chunked under
// max_batch_size). If the bucket is imbalanced, sort by length and
// recurse on the two halves. Otherwise emit as a single batch.
void PlanChunk(std::vector<int>& indices, const std::vector<int>& lengths,
               const SchedulerConfig& cfg,
               std::vector<std::vector<int>>* out) {
  if (indices.empty()) return;
  if (indices.size() == 1) {
    out->push_back(std::move(indices));
    return;
  }
  int max_len = 0;
  long total = 0;
  for (int i : indices) {
    const int L = lengths[static_cast<std::size_t>(i)];
    if (L > max_len) max_len = L;
    total += L;
  }
  const float mean_len = static_cast<float>(total) /
                         static_cast<float>(indices.size());
  if (mean_len > 0.0f &&
      static_cast<float>(max_len) / mean_len > cfg.imbalance_threshold) {
    // Sort by length and split in half. The longer half packs against
    // similarly-long siblings (still some imbalance but tighter); the
    // shorter half is densely packed.
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) {
                return lengths[static_cast<std::size_t>(a)] <
                       lengths[static_cast<std::size_t>(b)];
              });
    const std::size_t mid = indices.size() / 2;
    std::vector<int> lo(indices.begin(),
                         indices.begin() + static_cast<long>(mid));
    std::vector<int> hi(indices.begin() + static_cast<long>(mid),
                         indices.end());
    PlanChunk(lo, lengths, cfg, out);
    PlanChunk(hi, lengths, cfg, out);
    return;
  }
  out->push_back(std::move(indices));
}

}  // namespace

std::vector<std::vector<int>> PlanBatches(const std::vector<int>& lengths,
                                          const SchedulerConfig& cfg) {
  std::vector<std::vector<int>> out;
  if (lengths.empty()) return out;
  const int max_batch =
      cfg.max_batch_size > 0 ? cfg.max_batch_size : static_cast<int>(lengths.size());
  // First, chunk by max_batch_size in caller order (so the dispatch
  // doesn't grow a single packed batch beyond memory budgets). Each
  // chunk then gets the imbalance check inside PlanChunk.
  std::vector<int> all_indices(lengths.size());
  std::iota(all_indices.begin(), all_indices.end(), 0);
  for (std::size_t start = 0; start < all_indices.size();
       start += static_cast<std::size_t>(max_batch)) {
    const std::size_t end =
        std::min(start + static_cast<std::size_t>(max_batch),
                 all_indices.size());
    std::vector<int> chunk(all_indices.begin() + static_cast<long>(start),
                            all_indices.begin() + static_cast<long>(end));
    PlanChunk(chunk, lengths, cfg, &out);
  }
  return out;
}

}  // namespace esm
