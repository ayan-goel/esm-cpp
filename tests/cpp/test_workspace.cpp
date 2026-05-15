#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "esm_cpp/workspace.h"

TEST(Workspace, DefaultConstructHasZeroCapacityNoBufferYet) {
  esm::Workspace ws;
  EXPECT_EQ(ws.bytes_used(), 0u);
  EXPECT_EQ(ws.bytes_capacity(), 0u);
}

TEST(Workspace, ReserveAllocatesBacking) {
  esm::Workspace ws;
  ws.reserve(1024);
  EXPECT_EQ(ws.bytes_used(), 0u);
  EXPECT_GE(ws.bytes_capacity(), 1024u);
}

TEST(Workspace, AllocateFloatsAdvancesCursor) {
  esm::Workspace ws;
  ws.reserve(1024);
  float* a = ws.allocate<float>(16);
  ASSERT_NE(a, nullptr);
  EXPECT_GE(ws.bytes_used(), 16u * sizeof(float));
  for (int i = 0; i < 16; ++i) a[i] = static_cast<float>(i);
  for (int i = 0; i < 16; ++i) EXPECT_EQ(a[i], static_cast<float>(i));
}

TEST(Workspace, AlignmentRespectedAcrossAllocations) {
  esm::Workspace ws;
  ws.reserve(4096);
  std::int32_t* a = ws.allocate<std::int32_t>(3);  // 12 bytes, leaves cursor misaligned for double
  double* b = ws.allocate<double>(4);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b) % alignof(double), 0u);
  std::int8_t* c = ws.allocate<std::int8_t>(5);
  // No alignment requirement above 1 for int8_t.
  EXPECT_NE(c, nullptr);
  // Different allocations get disjoint ranges.
  EXPECT_NE(static_cast<void*>(a), static_cast<void*>(b));
}

TEST(Workspace, ResetRewindsCursorButKeepsCapacity) {
  esm::Workspace ws;
  ws.reserve(4096);
  ws.allocate<float>(64);
  std::size_t cap_before = ws.bytes_capacity();
  ASSERT_GT(ws.bytes_used(), 0u);
  ws.reset();
  EXPECT_EQ(ws.bytes_used(), 0u);
  EXPECT_EQ(ws.bytes_capacity(), cap_before);
  // Second allocation reuses the same backing memory.
  float* p2 = ws.allocate<float>(64);
  EXPECT_NE(p2, nullptr);
}

TEST(Workspace, GrowsOnOverflow) {
  esm::Workspace ws;
  ws.reserve(64);
  // Allocation larger than initial capacity must succeed by growing.
  float* p = ws.allocate<float>(1024);
  ASSERT_NE(p, nullptr);
  for (int i = 0; i < 1024; ++i) p[i] = static_cast<float>(i);
  EXPECT_GE(ws.bytes_capacity(), 1024u * sizeof(float));
}

TEST(Workspace, NoGrowthOnRepeatedSameSizedRuns) {
  esm::Workspace ws;
  ws.reserve(4096);
  for (int i = 0; i < 4; ++i) {
    ws.allocate<float>(128);
    ws.allocate<int>(256);
  }
  std::size_t cap_after_first = ws.bytes_capacity();
  ws.reset();
  for (int i = 0; i < 4; ++i) {
    ws.allocate<float>(128);
    ws.allocate<int>(256);
  }
  EXPECT_EQ(ws.bytes_capacity(), cap_after_first);
}

TEST(Workspace, ScopedActivationFlagsInUse) {
  esm::Workspace ws;
  EXPECT_FALSE(ws.in_use());
  {
    auto guard = ws.activate();
    EXPECT_TRUE(ws.in_use());
  }
  EXPECT_FALSE(ws.in_use());
}
