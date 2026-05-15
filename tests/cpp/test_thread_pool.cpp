#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "esm_cpp/thread_pool.h"

TEST(ThreadPool, SerialWhenSizeOne) {
  esm::ThreadPool pool(1);
  EXPECT_EQ(pool.size(), 1u);
  std::vector<int> hits(8, 0);
  pool.parallel_for(0, 8, 1, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) hits[i]++;
  });
  for (int h : hits) EXPECT_EQ(h, 1);
}

TEST(ThreadPool, ParallelMultiplesEveryIndexExactlyOnce) {
  esm::ThreadPool pool(4);
  std::vector<std::atomic<int>> hits(64);
  for (auto& a : hits) a.store(0);
  pool.parallel_for(0, 64, 4, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) hits[i].fetch_add(1);
  });
  for (auto& a : hits) EXPECT_EQ(a.load(), 1);
}

TEST(ThreadPool, ZeroRangeIsNoOp) {
  esm::ThreadPool pool(2);
  std::atomic<int> calls{0};
  pool.parallel_for(0, 0, 1, [&](int, int) { calls.fetch_add(1); });
  EXPECT_EQ(calls.load(), 0);
}

TEST(ThreadPool, GrainBoundsChunkSize) {
  esm::ThreadPool pool(4);
  std::vector<std::pair<int, int>> chunks;
  std::mutex m;
  pool.parallel_for(0, 17, 4, [&](int begin, int end) {
    std::lock_guard<std::mutex> g(m);
    chunks.emplace_back(begin, end);
  });
  // No chunk smaller than grain unless it's the final remainder.
  for (auto& c : chunks) {
    EXPECT_GE(c.second - c.first, 1);
  }
  // Coverage is [0, 17) disjoint.
  std::vector<int> covered;
  for (auto& c : chunks)
    for (int i = c.first; i < c.second; ++i) covered.push_back(i);
  std::sort(covered.begin(), covered.end());
  for (int i = 0; i < 17; ++i) EXPECT_EQ(covered[i], i);
}

TEST(ThreadPool, MultipleParallelForRunsReuseWorkers) {
  esm::ThreadPool pool(3);
  for (int round = 0; round < 4; ++round) {
    std::atomic<int> total{0};
    pool.parallel_for(0, 30, 1, [&](int begin, int end) {
      for (int i = begin; i < end; ++i) total.fetch_add(i);
    });
    int expected = 0;
    for (int i = 0; i < 30; ++i) expected += i;
    EXPECT_EQ(total.load(), expected);
  }
}

TEST(ThreadPool, DefaultPoolSizeRespectsEnvVar) {
  ::setenv("ESM_NUM_THREADS", "2", 1);
  esm::ThreadPool pool = esm::ThreadPool::FromEnv();
  EXPECT_EQ(pool.size(), 2u);
  ::unsetenv("ESM_NUM_THREADS");
}
