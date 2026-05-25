#pragma once

// Apple-AMX fp16 GEMM backend: per-Linear compiled BNNSGraph contexts loaded
// from on-disk .mlmodelc artifacts at Model::load and executed in the forward
// path under ESM_APPLE_AMX=on. The whole surface is gated behind
// ESM_APPLE_AMX_AVAILABLE — compiled out on non-Apple builds, so the rest of
// the engine builds cleanly on Linux ARM / Graviton with no Accelerate
// dependency. The runtime needs zero coremltools; the artifacts are produced
// at convert time by tools/build_amx_artifacts.py (Python 3.12 + coremltools).

#include <cstddef>
#include <memory>
#include <string>

#include "esm_cpp/status.h"

#ifdef ESM_APPLE_AMX_AVAILABLE

namespace esm {

// Opaque RAII wrapper around one Linear's compiled BNNSGraph context:
//   bnns_graph_t graph  (mmap'd from the .mlmodelc),
//   bnns_graph_context_t ctx,
//   workspace buffer,
//   resolved argument positions for "x" (input) and "out" (output).
// One-time setup at Model::load; the forward calls Execute() which only does
// BNNSGraphContextSetDynamicShapes(M) + BNNSGraphContextExecute. No allocation
// per forward.
class AppleAmxContext {
 public:
  // Compile + create the context from a .mlmodelc directory. `K` is the
  // Linear's input dim and `N` the output dim — both must match the artifact's
  // static dims (BNNSGraph won't tell us, but the caller knows from the
  // weight shape). Returns nullptr on failure; the caller falls back per-Linear.
  static std::unique_ptr<AppleAmxContext> LoadFromDir(const std::string& dir,
                                                     int K, int N);

  ~AppleAmxContext();
  AppleAmxContext(const AppleAmxContext&) = delete;
  AppleAmxContext& operator=(const AppleAmxContext&) = delete;

  // Run one forward: C[M, N] = X[M, K] · W_baked^T + bias_baked, with W/bias
  // baked into the graph as fp16 constants. Pre-conditions: in points to M*K
  // floats; out points to M*N floats. Returns OK on success.
  Status Execute(const float* in, float* out, int M);

 private:
  AppleAmxContext() = default;

  // Opaque pointers stored as void* to keep BNNS headers out of this header
  // (and so the public API doesn't require linking Accelerate from non-AMX
  // TUs). The .cpp casts back to the BNNS types.
  //
  // NOTE: workspace is NOT owned per-context. The BNNS workspace requirement
  // grows with M (e.g. ~3.8 KB / M for a [320,320] Linear), so per-context
  // ownership at the model-max M would balloon (200 contexts × tens of MB).
  // Instead Execute() uses a single thread_local buffer that grows lazily —
  // same allocate-once pattern as the SDOT activation staging.
  void* graph_data_ = nullptr;
  std::size_t graph_size_ = 0;
  void* ctx_data_ = nullptr;
  std::size_t ctx_size_ = 0;
  std::size_t x_pos_ = 0;
  std::size_t out_pos_ = 0;
  std::size_t arg_count_ = 0;
  int k_ = 0;
  int n_ = 0;
};

}  // namespace esm

#endif  // ESM_APPLE_AMX_AVAILABLE
