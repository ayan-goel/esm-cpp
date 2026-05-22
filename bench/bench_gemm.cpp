// SGEMM microbenchmarks on the four critical shapes from SPEC §2:
//   QKV   [B·L, d, 3d]
//   out   [B·L, d, d]
//   fc1   [B·L, d, 4d]
//   fc2   [B·L, 4d, d]
// across the ESM-2 model dims d in {320, 640, 1280, 2560} and two B·L
// regimes (single-seq L=300, batch-16 L=300 packed). The Phase 1 gate
// is "≥80% of MKL" on these shapes; that measurement happens on an x86
// AVX-512+VNNI instance.
//
// Build:
//   cmake -B build -DCMAKE_BUILD_TYPE=Release -DESM_BUILD_BENCH=ON
//   cmake --build build -j --target bench_gemm
//   ./build/bench/bench_gemm --benchmark_out=results.json
//
// Forces ISA via ESM_FORCE_ISA=<ref|neon|avx512|avx512vnni> at run time
// so the same binary can sweep every registered path.

#include <benchmark/benchmark.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "esm_cpp/cpu_features.h"
#include "esm_cpp/kernels.h"
#include "esm_cpp/quant.h"

namespace {

std::vector<float> RandomVec(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

struct Shape {
  int M, N, K;
};

void BM_Linear(benchmark::State& state, Shape s) {
  const auto Msz = static_cast<std::size_t>(s.M);
  const auto Nsz = static_cast<std::size_t>(s.N);
  const auto Ksz = static_cast<std::size_t>(s.K);
  auto A = RandomVec(Msz * Ksz, 0x1234);
  auto W = RandomVec(Nsz * Ksz, 0x5678);
  auto bias = RandomVec(Nsz, 0x9abc);
  std::vector<float> C(Msz * Nsz);
  for (auto _ : state) {
    esm::kernels::Linear(A.data(), W.data(), bias.data(), C.data(), s.M, s.N,
                         s.K);
    benchmark::DoNotOptimize(C);
  }
  const double flops = 2.0 * static_cast<double>(Msz) *
                       static_cast<double>(Nsz) * static_cast<double>(Ksz);
  state.counters["GFLOPs"] = benchmark::Counter(
      flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::OneK::kIs1000);
  state.SetLabel(std::string("isa=") +
                 std::string(esm::IsaToString(esm::CurrentIsa())));
}

void BM_LinearInt8(benchmark::State& state, Shape s) {
  const auto Msz = static_cast<std::size_t>(s.M);
  const auto Nsz = static_cast<std::size_t>(s.N);
  const auto Ksz = static_cast<std::size_t>(s.K);
  auto A = RandomVec(Msz * Ksz, 0x1234);
  auto W = RandomVec(Nsz * Ksz, 0x5678);
  auto bias = RandomVec(Nsz, 0x9abc);
  esm::quant::QuantizedTensor qt;
  esm::quant::Quantize(W.data(), s.N, s.K, &qt);
  std::vector<float> C(Msz * Nsz);
  for (auto _ : state) {
    esm::kernels::LinearInt8(A.data(), qt, bias.data(), C.data(), s.M, s.N,
                             s.K);
    benchmark::DoNotOptimize(C);
  }
  const double flops = 2.0 * static_cast<double>(Msz) *
                       static_cast<double>(Nsz) * static_cast<double>(Ksz);
  state.counters["GFLOPs"] = benchmark::Counter(
      flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::OneK::kIs1000);
  state.SetLabel(std::string("isa=") +
                 std::string(esm::IsaToString(esm::CurrentIsa())));
}

// Static-init registration so benchmark_main can pick up the suite.
struct RegisterShapes {
  RegisterShapes() {
    struct Entry { const char* label; Shape s; };
    const Entry entries[] = {
        // 8M (d=320, ffn=1280)
        {"qkv_d320_L300",   {300,  960,  320}},
        {"out_d320_L300",   {300,  320,  320}},
        {"fc1_d320_L300",   {300, 1280,  320}},
        {"fc2_d320_L300",   {300,  320, 1280}},
        // 150M (d=640, ffn=2560)
        {"qkv_d640_L300",   {300, 1920,  640}},
        {"out_d640_L300",   {300,  640,  640}},
        {"fc1_d640_L300",   {300, 2560,  640}},
        {"fc2_d640_L300",   {300,  640, 2560}},
        // 650M (d=1280, ffn=5120) — the Phase 1 gate model
        {"qkv_d1280_L300",  {300, 3840, 1280}},
        {"out_d1280_L300",  {300, 1280, 1280}},
        {"fc1_d1280_L300",  {300, 5120, 1280}},
        {"fc2_d1280_L300",  {300, 1280, 5120}},
        {"qkv_d1280_BL4800",{4800, 3840, 1280}},
        {"out_d1280_BL4800",{4800, 1280, 1280}},
        {"fc1_d1280_BL4800",{4800, 5120, 1280}},
        {"fc2_d1280_BL4800",{4800, 1280, 5120}},
        // 3B (d=2560, ffn=10240)
        {"qkv_d2560_L300",  {300, 7680, 2560}},
        {"out_d2560_L300",  {300, 2560, 2560}},
        {"fc1_d2560_L300",  {300,10240, 2560}},
        {"fc2_d2560_L300",  {300, 2560,10240}},
    };
    for (const auto& e : entries) {
      benchmark::RegisterBenchmark(e.label, BM_Linear, e.s)
          ->Unit(benchmark::kMillisecond);
      benchmark::RegisterBenchmark(std::string("int8_") + e.label,
                                   BM_LinearInt8, e.s)
          ->Unit(benchmark::kMillisecond);
    }
  }
};

const RegisterShapes kRegister;

}  // namespace

BENCHMARK_MAIN();
