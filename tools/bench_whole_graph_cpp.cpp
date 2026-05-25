// Phase 13 T2/T3: standalone C++ bench for the AppleWholeGraphContext bridge.
//
// Loads a whole-graph .mlmodelc built by tools/build_whole_graph_artifacts.py,
// runs the bridge predict in a hot loop, and reports p10/p50/p90 latency for
// each compute_units configuration. The GO/NO-GO gate (T2) compares the C++
// bridge p50 at compute_units CPU_AND_NE (and ALL) against the Phase-11 AMX
// e2e baseline at the same (B, L).
//
// Build (via the regular CMake build — see end of file for the line in
// CMakeLists.txt). Then run:
//   ./build/tools/bench_whole_graph_cpp \
//       --artifact weights/esm2_8m.whole-graph/B-8_L-256/whole_graph.mlmodelc \
//       --batch 8 --seq-len 256 --vocab 33 --compute-units cpu_and_ne

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "esm_cpp/apple_whole_graph.h"

using clk = std::chrono::steady_clock;

static double MsSince(clk::time_point t) {
  return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

int main(int argc, char** argv) {
  std::string artifact;
  int B = 8, L = 256, V = 33;
  int warmup = 3, iters = 11;
  std::string cu_str = "cpu_and_ne";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--artifact") artifact = next();
    else if (a == "--batch") B = std::stoi(next());
    else if (a == "--seq-len") L = std::stoi(next());
    else if (a == "--vocab") V = std::stoi(next());
    else if (a == "--warmup") warmup = std::stoi(next());
    else if (a == "--iters") iters = std::stoi(next());
    else if (a == "--compute-units") cu_str = next();
    else {
      std::fprintf(stderr, "unknown arg %s\n", a.c_str());
      return 2;
    }
  }
  if (artifact.empty()) {
    std::fprintf(stderr, "usage: --artifact <path/to/whole_graph.mlmodelc> "
                          "--batch N --seq-len N --vocab N "
                          "--compute-units {cpu_only,cpu_and_ne,all}\n");
    return 2;
  }
  esm::WholeGraphComputeUnits cu = esm::WholeGraphComputeUnits::kCpuAndNeuralEngine;
  if (cu_str == "cpu_only") cu = esm::WholeGraphComputeUnits::kCpuOnly;
  else if (cu_str == "all") cu = esm::WholeGraphComputeUnits::kAll;
  else if (cu_str == "cpu_and_ne") cu = esm::WholeGraphComputeUnits::kCpuAndNeuralEngine;
  else { std::fprintf(stderr, "bad --compute-units: %s\n", cu_str.c_str()); return 2; }

  std::fprintf(stderr, "[bench] artifact=%s\n", artifact.c_str());
  std::fprintf(stderr, "        B=%d L=%d V=%d compute_units=%s\n", B, L, V, cu_str.c_str());

  auto t_load = clk::now();
  auto ctx = esm::AppleWholeGraphContext::LoadFromDir(artifact, B, L, V, cu);
  if (!ctx) {
    std::fprintf(stderr, "FAILED to load context (non-Apple build, or "
                          "missing/invalid mlmodelc)\n");
    return 1;
  }
  std::fprintf(stderr, "[bench] loaded in %.0f ms\n", MsSince(t_load));

  std::vector<std::int32_t> ids(static_cast<std::size_t>(B) * L);
  std::vector<std::int32_t> mask(static_cast<std::size_t>(B) * L, 1);
  // Deterministic amino-acid-ish IDs (4..23). Values don't matter for perf.
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = 4 + static_cast<std::int32_t>(i % 20);
  }
  std::vector<float> out(static_cast<std::size_t>(B) * L * V);

  // Warmup
  for (int w = 0; w < warmup; ++w) {
    auto s = ctx->Execute(ids.data(), mask.data(), out.data());
    if (!s.ok()) {
      std::fprintf(stderr, "warmup Execute failed: %.*s\n",
                   static_cast<int>(s.message().size()), s.message().data());
      return 1;
    }
  }

  std::vector<double> ts;
  ts.reserve(static_cast<std::size_t>(iters));
  for (int it = 0; it < iters; ++it) {
    auto t = clk::now();
    auto s = ctx->Execute(ids.data(), mask.data(), out.data());
    if (!s.ok()) {
      std::fprintf(stderr, "Execute failed: %.*s\n",
                   static_cast<int>(s.message().size()), s.message().data());
      return 1;
    }
    ts.push_back(MsSince(t));
  }
  std::sort(ts.begin(), ts.end());
  double p10 = ts[ts.size() / 10];
  double p50 = ts[ts.size() / 2];
  double p90 = ts[(9 * ts.size()) / 10];
  std::fprintf(stderr, "[bench] p10=%.2f  p50=%.2f  p90=%.2f  ms  (over %zu iters)\n",
               p10, p50, p90, ts.size());
  std::fprintf(stdout, "BENCH: cpp_bridge_%s p50_ms=%.2f\n", cu_str.c_str(), p50);

  // Output sanity: at least one finite value
  bool any_finite = false;
  for (float v : out) {
    if (std::isfinite(v)) { any_finite = true; break; }
  }
  std::fprintf(stderr, "[bench] output finite-any=%d  out[0]=%.4f\n",
               static_cast<int>(any_finite), static_cast<double>(out[0]));
  return any_finite ? 0 : 1;
}
