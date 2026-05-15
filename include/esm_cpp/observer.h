#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace esm {

// Per-site activation observer for static-quant calibration. Each call
// to Observe(site_key, values, n) feeds |values| into a reservoir sampled
// down to kReservoirSize (default 65536). After calibration, Percentile()
// returns the requested percentile per site — typically 99.9 for
// SmoothQuant's activation outlier estimate.
//
// Not thread-safe. The calibration driver runs one forward at a time
// to keep observations sequenced; the observer's reservoir RNG is
// deterministic so the same calibration data produces the same stats.
class ActivationObserver {
 public:
  static constexpr std::size_t kReservoirSize = 65536;

  ActivationObserver() = default;

  // Feed |values[0..n)| into the reservoir for site_key. Absolute values
  // are recorded so the percentile reflects activation magnitude.
  void Observe(std::string_view site_key, const float* values, std::size_t n);

  // Returns the per-site `pctile`-th percentile (0..100). Empty when no
  // observations have been made.
  std::unordered_map<std::string, float> Percentile(float pctile) const;

  // Reset all reservoirs back to empty.
  void Clear();

 private:
  struct Reservoir {
    std::vector<float> samples;  // |values|, reservoir-sampled
    std::size_t total_seen = 0;
    std::mt19937 rng{0xc1b1ce};
  };
  std::unordered_map<std::string, Reservoir> reservoirs_;
};

}  // namespace esm
