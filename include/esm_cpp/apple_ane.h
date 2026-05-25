#pragma once

// Apple Neural Engine (ANE) fp16 GEMM backend: per-Linear-per-bucket compiled
// MLModel artifacts loaded from on-disk .mlmodelc files at Model::load and
// executed via CoreML's MLModel runtime under ESM_APPLE_ANE=on. ANE is
// reachable ONLY through CoreML — BNNSGraph (the Phase-11 path) is CPU/AMX
// only. ANE requires static shapes, so each context represents one fixed-M
// Linear; the dispatcher in LinearProj pads M to the nearest bucket.
//
// The actual MLModel surface lives in src/apple_ane.mm (Objective-C++) and is
// gated by ESM_APPLE_ANE_AVAILABLE — compiled out on non-Apple builds, where
// this class is a stub (LoadFromDir always returns nullptr).

#include <cstddef>
#include <memory>
#include <string>

#include "esm_cpp/status.h"

namespace esm {

class AppleAneContext {
 public:
  // Compile + open the MLModel for a Linear at the fixed shape (M, K) -> (M, N)
  // with compute_units=CPU_AND_NE. Returns nullptr on failure or on non-Apple
  // builds. The caller (Model loader) knows M/K/N from the bucket + weight.
  static std::unique_ptr<AppleAneContext> LoadFromDir(const std::string& dir,
                                                     int M, int K, int N);

  ~AppleAneContext();
  AppleAneContext(const AppleAneContext&) = delete;
  AppleAneContext& operator=(const AppleAneContext&) = delete;

  // Run one forward at the context's fixed M: C[M, N] = X[M, K] · W_baked^T +
  // bias_baked. Pre-conditions: in points to M*K floats; out points to M*N
  // floats. The caller is responsible for pad-to-bucket (this method does NOT
  // re-shape).
  Status Execute(const float* in, float* out);

  int bucket_m() const { return m_; }
  int k() const { return k_; }
  int n() const { return n_; }

 private:
  AppleAneContext() = default;

#ifdef ESM_APPLE_ANE_AVAILABLE
  // Opaque pointers to ARC-retained CoreML objects (cast in the .mm file).
  // model_ + reusable input/output MLMultiArrays + MLPredictionOptions
  // (outputBackings = pre-allocated output buffer) + MLDictionaryFeatureProvider
  // pre-built around the input array. Pre-allocating these once per context
  // avoids per-call MLMultiArray allocation of large buffers (40MB at M=8192,
  // fc1) which was the main source of integrated-forward slowdown.
  void* model_ = nullptr;
  void* in_arr_ = nullptr;
  void* out_arr_ = nullptr;
  void* provider_ = nullptr;
  void* options_ = nullptr;
#endif
  int m_ = 0;
  int k_ = 0;
  int n_ = 0;
};

}  // namespace esm
