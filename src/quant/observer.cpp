#include "esm_cpp/observer.h"

#include <algorithm>
#include <cmath>

namespace esm {

void ActivationObserver::Observe(std::string_view site_key, const float* values,
                                  std::size_t n) {
  if (n == 0) return;
  auto it = reservoirs_.find(std::string(site_key));
  if (it == reservoirs_.end()) {
    Reservoir empty;
    empty.samples.reserve(kReservoirSize);
    it = reservoirs_.emplace(std::string(site_key), std::move(empty)).first;
  }
  Reservoir& r = it->second;
  for (std::size_t i = 0; i < n; ++i) {
    const float a = std::fabs(values[i]);
    if (r.samples.size() < kReservoirSize) {
      r.samples.push_back(a);
    } else {
      // Standard reservoir sampling (Algorithm R): each new element
      // replaces a random slot with probability kReservoirSize / total_seen.
      std::uniform_int_distribution<std::size_t> dist(0, r.total_seen);
      const std::size_t j = dist(r.rng);
      if (j < kReservoirSize) r.samples[j] = a;
    }
    ++r.total_seen;
  }
}

std::unordered_map<std::string, float> ActivationObserver::Percentile(
    float pctile) const {
  std::unordered_map<std::string, float> out;
  out.reserve(reservoirs_.size());
  for (const auto& [key, r] : reservoirs_) {
    if (r.samples.empty()) {
      out[key] = 0.0f;
      continue;
    }
    std::vector<float> sorted = r.samples;
    std::sort(sorted.begin(), sorted.end());
    const float frac = std::clamp(pctile / 100.0f, 0.0f, 1.0f);
    std::size_t idx = static_cast<std::size_t>(
        std::floor(frac * static_cast<float>(sorted.size() - 1)));
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    out[key] = sorted[idx];
  }
  return out;
}

void ActivationObserver::Clear() { reservoirs_.clear(); }

}  // namespace esm
