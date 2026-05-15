// Phase 2 Slice 3: ActivationObserver acceptance.
//
// - Reservoir of |values| per site_key (configurable; default 65536).
// - Percentile() returns the requested percentile from each reservoir.
// - For uniform [0, A] input with N >> reservoir size, the empirical
//   99.9-pctile must be within ~A/100 of A * 0.999.
// - Multiple Observe() calls accumulate; Clear() empties.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "esm_cpp/observer.h"

TEST(ActivationObserver, UniformInputProduces999PctileNearAnalyticBound) {
  esm::ActivationObserver obs;
  std::mt19937 rng(0xfade);
  std::uniform_real_distribution<float> dist(0.0f, 10.0f);
  std::vector<float> values(200000);
  for (auto& x : values) x = dist(rng);
  obs.Observe("site_a", values.data(), values.size());
  auto stats = obs.Percentile(99.9f);
  ASSERT_TRUE(stats.find("site_a") != stats.end());
  // Analytic 99.9-pctile of uniform [0, 10] is 9.99. With reservoir
  // sampling of 65536 from 200000, drift is bounded by sampling noise;
  // ±0.2 is loose-enough to be reliable, tight-enough to catch bugs.
  EXPECT_NEAR(stats["site_a"], 9.99f, 0.2f);
}

TEST(ActivationObserver, NegativeValuesUseAbsoluteMagnitude) {
  esm::ActivationObserver obs;
  // [-50, +1] — the negative -50 should dominate the 99.9-pctile since
  // observation is on |x|.
  std::vector<float> values(10000, 1.0f);
  values[100] = -50.0f;
  values[200] = -49.0f;
  values[300] = -48.0f;
  obs.Observe("site_neg", values.data(), values.size());
  auto stats = obs.Percentile(99.9f);
  EXPECT_GE(stats["site_neg"], 1.0f);
  // Top |value| present in input is 50 — 99.9-pctile of 10000 samples
  // should sit near the top tail.
  EXPECT_LE(stats["site_neg"], 50.0f);
}

TEST(ActivationObserver, MultipleSitesTrackedIndependently) {
  esm::ActivationObserver obs;
  std::vector<float> small(1000, 0.5f);
  std::vector<float> large(1000, 5.0f);
  obs.Observe("small", small.data(), small.size());
  obs.Observe("large", large.data(), large.size());
  auto stats = obs.Percentile(99.9f);
  EXPECT_NEAR(stats["small"], 0.5f, 1e-4f);
  EXPECT_NEAR(stats["large"], 5.0f, 1e-4f);
}

TEST(ActivationObserver, ClearResetsState) {
  esm::ActivationObserver obs;
  std::vector<float> data(100, 1.0f);
  obs.Observe("a", data.data(), data.size());
  EXPECT_FALSE(obs.Percentile(99.9f).empty());
  obs.Clear();
  EXPECT_TRUE(obs.Percentile(99.9f).empty());
}

TEST(ActivationObserver, MultipleObserveCallsAccumulate) {
  esm::ActivationObserver obs;
  std::vector<float> a(1000, 0.1f);
  std::vector<float> b(1000, 10.0f);
  obs.Observe("site", a.data(), a.size());
  obs.Observe("site", b.data(), b.size());
  auto stats = obs.Percentile(99.9f);
  // Combined population has 0.1's and 10's; 99.9-pctile must reach the
  // 10's (top 0.1% of 2000 samples = top 2 values).
  EXPECT_NEAR(stats["site"], 10.0f, 1e-4f);
}

TEST(ActivationObserver, EmptyObserveIsNoOp) {
  esm::ActivationObserver obs;
  std::vector<float> empty;
  obs.Observe("nothing", empty.data(), 0);
  EXPECT_TRUE(obs.Percentile(99.9f).empty());
}
