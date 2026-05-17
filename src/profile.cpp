#include "esm_cpp/profile.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace esm::profile {

namespace {

// Read ESM_PROFILE once on first call. Any non-empty / non-"0" value
// enables the profiler.
bool ReadEnvOnce() {
  const char* v = std::getenv("ESM_PROFILE");
  if (!v || !*v) return false;
  if (std::strcmp(v, "0") == 0) return false;
  return true;
}

bool& EnabledFlag() {
  static bool flag = ReadEnvOnce();
  return flag;
}

// Per-thread accumulators. Pointer indirection lets the global dumper
// walk all live thread accumulators without forcing every thread to
// register up-front.
struct ThreadAcc {
  std::unordered_map<std::string, std::int64_t> ns;
  std::unordered_map<std::string, std::int64_t> counts;
};

std::mutex& Registry() {
  static std::mutex m;
  return m;
}

std::vector<ThreadAcc*>& AllAccs() {
  static std::vector<ThreadAcc*> v;
  return v;
}

ThreadAcc& TLS() {
  thread_local ThreadAcc acc;
  thread_local bool registered = false;
  if (!registered) {
    std::lock_guard<std::mutex> g(Registry());
    AllAccs().push_back(&acc);
    registered = true;
  }
  return acc;
}

}  // namespace

bool Enabled() { return EnabledFlag(); }

void AddNs(const char* section, std::int64_t ns) {
  if (!Enabled()) return;
  TLS().ns[section] += ns;
}

void BumpCounter(const char* name, std::int64_t by) {
  if (!Enabled()) return;
  TLS().counts[name] += by;
}

void DumpAndReset() {
  if (!Enabled()) return;
  std::map<std::string, std::int64_t> agg_ns;
  std::map<std::string, std::int64_t> agg_counts;
  {
    std::lock_guard<std::mutex> g(Registry());
    for (auto* a : AllAccs()) {
      for (auto& [k, v] : a->ns) agg_ns[k] += v;
      for (auto& [k, v] : a->counts) agg_counts[k] += v;
      a->ns.clear();
      a->counts.clear();
    }
  }
  std::int64_t total_ns = 0;
  for (auto& [k, v] : agg_ns) total_ns += v;
  std::vector<std::pair<std::string, std::int64_t>> rows(agg_ns.begin(),
                                                          agg_ns.end());
  std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::fprintf(stderr, "[esm-profile] forward breakdown (total wall = %.3f ms)\n",
                total_ns / 1.0e6);
  for (const auto& [k, v] : rows) {
    const double ms = v / 1.0e6;
    const double pct = total_ns > 0
                            ? 100.0 * static_cast<double>(v) /
                                  static_cast<double>(total_ns)
                            : 0.0;
    std::fprintf(stderr, "  %-32s %8.3f ms  %5.1f%%\n", k.c_str(), ms, pct);
  }
  if (!agg_counts.empty()) {
    std::fprintf(stderr, "[esm-profile] counters\n");
    for (const auto& [k, v] : agg_counts) {
      std::fprintf(stderr, "  %-32s %12lld\n", k.c_str(),
                    static_cast<long long>(v));
    }
  }
}

}  // namespace esm::profile
