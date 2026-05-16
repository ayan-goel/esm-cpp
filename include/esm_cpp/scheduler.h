#pragma once

#include <cstddef>
#include <vector>

namespace esm {

struct SchedulerConfig {
  // Bucket-split threshold. If max(L) / mean(L) exceeds this ratio, the
  // batch is sorted by length and split into two halves. The plan SPEC
  // calls for "length-bucket only when packed-work imbalance > 20%";
  // we use the ratio max/mean which is roughly equivalent and simpler
  // to reason about than (L_max^2 * B) / sum(L^2).
  float imbalance_threshold = 1.2f;
  // Hard cap on sequences per packed forward. Beyond this, the scheduler
  // splits into multiple back-to-back packed dispatches. Default sized
  // generously so PPPL's L-masked-variants workload (B = L = hundreds)
  // packs as a single forward in typical configurations.
  int max_batch_size = 256;
};

// Given a list of per-sequence lengths (in caller order), return a list
// of batches; each batch is the list of caller-indices that should run
// through a single packed forward. The returned plan covers every input
// index exactly once. Order within batches is by ascending length when
// a bucket-split fires, otherwise input order is preserved.
//
// Free function so the bucketing logic is unit-testable without a Model.
std::vector<std::vector<int>> PlanBatches(const std::vector<int>& lengths,
                                          const SchedulerConfig& cfg);

}  // namespace esm
