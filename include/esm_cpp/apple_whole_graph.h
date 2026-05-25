#pragma once

// Apple whole-graph CoreML bridge: one compiled MLModel for the entire ESM-2
// forward (33 encoder layers + LM head), reached via CoreML's MLModel API
// with compute_units settable at load time. The artifact is built at convert
// time by tools/build_whole_graph_artifacts.py and is keyed by (B, L) — a
// shape pair that uniquely identifies one mlmodelc on disk.
//
// Phase 12 lesson: per-Linear ANE (197 tiny MLModels) thrashes the ANE
// compiled-state cache; the integrated forward is 50-100x slower than the
// per-op spike measured. The fix is the structure of *this* class: one
// MLModel, kept hot across calls, with op-fusion opportunities ANE actually
// uses. See notes/phase12.md.
//
// The actual MLModel surface lives in src/apple_whole_graph.mm
// (Objective-C++) and is gated by ESM_APPLE_ANE_AVAILABLE — compiled out on
// non-Apple builds where LoadFromDir always returns nullptr.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "esm_cpp/status.h"

namespace esm {

// Compute-unit selection for the underlying MLModel. CPU_AND_NE is the
// Phase-13 default; ALL adds GPU as a dispatch target (CoreML picks per-op).
enum class WholeGraphComputeUnits : int {
  kCpuOnly = 0,
  kCpuAndNeuralEngine = 1,
  kAll = 2,
};

class AppleWholeGraphContext {
 public:
  // Compile + open the MLModel at `dir` (a `.mlmodelc` bundle from
  // `tools/build_whole_graph_artifacts.py`) at fixed (B, L) and the requested
  // compute_units. Returns nullptr on failure or on non-Apple builds.
  static std::unique_ptr<AppleWholeGraphContext> LoadFromDir(
      const std::string& dir, int B, int L, int vocab_size,
      WholeGraphComputeUnits compute_units);

  ~AppleWholeGraphContext();
  AppleWholeGraphContext(const AppleWholeGraphContext&) = delete;
  AppleWholeGraphContext& operator=(const AppleWholeGraphContext&) = delete;

  // Run one whole-model forward.
  //   input_ids: int32 [B*L] (caller flattens row-major)
  //   attention_mask: int32 [B*L]
  //   logits_out: fp32 [B*L*V] (V == vocab_size passed to LoadFromDir)
  // Pre-allocated MLMultiArrays in this context are reused across calls; the
  // hot-path cost per Execute is two int32 memcpys in, one fp16->fp32
  // vectorized copy out, and one predictionFromFeatures: call.
  Status Execute(const std::int32_t* input_ids,
                 const std::int32_t* attention_mask,
                 float* logits_out);

  int batch() const { return b_; }
  int seq_len() const { return l_; }
  int vocab_size() const { return v_; }

 private:
  AppleWholeGraphContext() = default;

#ifdef ESM_APPLE_ANE_AVAILABLE
  // Opaque pointers to ARC-retained CoreML objects (cast in the .mm file).
  // We keep the model_, pre-allocated input MLMultiArrays (int32), the
  // pre-allocated output MLMultiArray (fp16 — ANE's native type, vectorized
  // to fp32 at Execute), a single MLDictionaryFeatureProvider wrapping both
  // input arrays, and MLPredictionOptions with `outputBackings` set so the
  // runtime never re-allocates the output between predictions.
  void* model_ = nullptr;
  void* ids_arr_ = nullptr;
  void* mask_arr_ = nullptr;
  void* out_arr_ = nullptr;
  void* provider_ = nullptr;
  void* options_ = nullptr;
#endif
  int b_ = 0;
  int l_ = 0;
  int v_ = 0;
};

}  // namespace esm
