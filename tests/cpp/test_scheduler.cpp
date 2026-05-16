// Phase 3 Slice 2: iteration-level scheduler. Tests cover the bucketing
// decision (PlanBatches) directly so we don't have to load a Model just
// to exercise the planner's branch logic.

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "esm_cpp/scheduler.h"

namespace {

std::vector<int> Lengths(const std::vector<std::vector<int>>& seqs) {
  std::vector<int> out;
  out.reserve(seqs.size());
  for (const auto& s : seqs) out.push_back(static_cast<int>(s.size()));
  return out;
}

}  // namespace

TEST(SchedulerPlan, SingleBatchWhenLengthsAreUniform) {
  esm::SchedulerConfig cfg;
  cfg.max_batch_size = 64;
  cfg.imbalance_threshold = 1.2f;
  // All length 100 — imbalance metric = max/mean = 1.0 < 1.2.
  std::vector<int> lengths(16, 100);
  auto plan = esm::PlanBatches(lengths, cfg);
  ASSERT_EQ(plan.size(), 1u);
  EXPECT_EQ(plan[0].size(), 16u);
}

TEST(SchedulerPlan, SplitsWhenLengthsAreImbalanced) {
  esm::SchedulerConfig cfg;
  cfg.max_batch_size = 64;
  cfg.imbalance_threshold = 1.2f;
  // 8 short (50) + 8 long (500). max/mean = 500/275 = 1.82 > 1.2 → split.
  std::vector<int> lengths;
  for (int i = 0; i < 8; ++i) lengths.push_back(50);
  for (int i = 0; i < 8; ++i) lengths.push_back(500);
  auto plan = esm::PlanBatches(lengths, cfg);
  EXPECT_GE(plan.size(), 2u);
  // Total coverage = all 16.
  std::size_t total = 0;
  for (const auto& b : plan) total += b.size();
  EXPECT_EQ(total, 16u);
}

TEST(SchedulerPlan, ChunksOnMaxBatchSize) {
  esm::SchedulerConfig cfg;
  cfg.max_batch_size = 4;
  cfg.imbalance_threshold = 1.2f;
  std::vector<int> lengths(10, 100);  // uniform; only chunking applies
  auto plan = esm::PlanBatches(lengths, cfg);
  // 10 sequences in chunks of 4 -> at least 3 batches (4+4+2).
  EXPECT_GE(plan.size(), 3u);
  std::size_t total = 0;
  for (const auto& b : plan) {
    EXPECT_LE(b.size(), 4u);
    total += b.size();
  }
  EXPECT_EQ(total, 10u);
}

TEST(SchedulerPlan, PreservesIndicesCoveringEveryInput) {
  esm::SchedulerConfig cfg;
  cfg.max_batch_size = 8;
  cfg.imbalance_threshold = 1.2f;
  std::vector<int> lengths;
  for (int i = 0; i < 17; ++i) lengths.push_back(50 + i * 30);
  auto plan = esm::PlanBatches(lengths, cfg);
  std::vector<int> seen;
  for (const auto& b : plan) {
    for (int idx : b) seen.push_back(idx);
  }
  std::sort(seen.begin(), seen.end());
  for (int i = 0; i < 17; ++i) EXPECT_EQ(seen[static_cast<std::size_t>(i)], i);
}

TEST(SchedulerPlan, EmptyInputProducesEmptyPlan) {
  auto plan = esm::PlanBatches({}, esm::SchedulerConfig{});
  EXPECT_TRUE(plan.empty());
}

TEST(SchedulerPlan, SingleSequenceIsOneBatch) {
  std::vector<int> lengths = {100};
  auto plan = esm::PlanBatches(lengths, esm::SchedulerConfig{});
  ASSERT_EQ(plan.size(), 1u);
  ASSERT_EQ(plan[0].size(), 1u);
  EXPECT_EQ(plan[0][0], 0);
}

TEST(SchedulerPlan, ThresholdAtBoundaryDoesNotSplit) {
  esm::SchedulerConfig cfg;
  cfg.imbalance_threshold = 2.0f;
  cfg.max_batch_size = 64;
  // max=500, mean=275 -> 1.82 < 2.0 -> no split.
  std::vector<int> lengths;
  for (int i = 0; i < 8; ++i) lengths.push_back(50);
  for (int i = 0; i < 8; ++i) lengths.push_back(500);
  auto plan = esm::PlanBatches(lengths, cfg);
  EXPECT_EQ(plan.size(), 1u);
}
