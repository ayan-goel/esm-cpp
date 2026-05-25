#include "esm_cpp/apple_ane.h"

#ifdef ESM_APPLE_ANE_AVAILABLE

#import <Accelerate/Accelerate.h>
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Per-Linear, per-bucket Neural Engine runtime via CoreML's MLModel API.
// Phase 12 lesson (post-T7 perf debug): a naive bridge that creates fresh
// MLMultiArrays per call is ~20x slower than coremltools.predict on the same
// artifact, because (a) MLModel allocates a new output MLMultiArray of the
// declared output type/shape on every prediction (40 MB for fc1 at M=8192),
// (b) `initWithDataPointer:deallocator:` doesn't actually give a zero-copy
// path on ANE — ANE has to bounce the buffer through its own allocator
// regardless, and the dataPointer:deallocator: route trips an internal slow
// path on large buffers. The fix is the same one CoreML's own clients use:
// pre-allocate the input and output MLMultiArrays at LoadFromDir time (in
// CoreML-managed memory, so ANE can map them efficiently) and pass the output
// via MLPredictionOptions.outputBackings so the runtime never re-allocates.
// Per-Execute is then just: memcpy fp32 input -> in_arr_, predict, vectorized
// fp16 -> fp32 copy from out_arr_ -> caller's `out`.

namespace esm {

namespace {

// Vectorized fp16 -> fp32 conversion via Accelerate's vImage. Beats the scalar
// `for (i) out[i]=(float)p[i]` loop ~30x for 10 MB outputs.
void Fp16ToFp32(const __fp16* src, float* dst, std::size_t n) {
  vImage_Buffer in_buf  = {(void*)src, 1, n, n * sizeof(__fp16)};
  vImage_Buffer out_buf = {dst,       1, n, n * sizeof(float)};
  vImageConvert_Planar16FtoPlanarF(&in_buf, &out_buf, 0);
}

}  // namespace

std::unique_ptr<AppleAneContext> AppleAneContext::LoadFromDir(
    const std::string& dir, int M, int K, int N) {
  if (M <= 0 || K <= 0 || N <= 0) return nullptr;

  @autoreleasepool {
    NSString* path = [NSString stringWithUTF8String:dir.c_str()];
    NSURL* url = [NSURL fileURLWithPath:path];
    MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    NSError* err = nil;
    MLModel* model =
        [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
    if (model == nil || err != nil) return nullptr;

    // Pre-allocate input MLMultiArray fp32 [M, K]. MLMultiArray allocates an
    // ANE-mappable buffer when we don't pass a dataPointer.
    NSArray* in_shape  = @[ @(M), @(K) ];
    NSArray* out_shape = @[ @(M), @(N) ];
    MLMultiArray* in_arr =
        [[MLMultiArray alloc] initWithShape:in_shape
                                   dataType:MLMultiArrayDataTypeFloat32
                                      error:&err];
    if (in_arr == nil || err != nil) return nullptr;
    // Output is fp16 (ANE's native type); we'll vectorize the fp16->fp32 copy
    // in Execute. Forcing the output to fp32 here would round-trip through an
    // MLModel-allocated fp32 buffer on every call (the slow path we're fixing).
    MLMultiArray* out_arr =
        [[MLMultiArray alloc] initWithShape:out_shape
                                   dataType:MLMultiArrayDataTypeFloat16
                                      error:&err];
    if (out_arr == nil || err != nil) return nullptr;

    MLDictionaryFeatureProvider* provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ @"x": in_arr }
                                                          error:&err];
    if (provider == nil || err != nil) return nullptr;

    MLPredictionOptions* opts = [[MLPredictionOptions alloc] init];
    opts.outputBackings = @{ @"out": out_arr };

    auto out = std::unique_ptr<AppleAneContext>(new AppleAneContext());
    out->model_    = (__bridge_retained void*)model;
    out->in_arr_   = (__bridge_retained void*)in_arr;
    out->out_arr_  = (__bridge_retained void*)out_arr;
    out->provider_ = (__bridge_retained void*)provider;
    out->options_  = (__bridge_retained void*)opts;
    out->m_ = M;
    out->k_ = K;
    out->n_ = N;
    return out;
  }
}

AppleAneContext::~AppleAneContext() {
  if (options_  != nullptr) { (void)(__bridge_transfer MLPredictionOptions*)options_;  options_  = nullptr; }
  if (provider_ != nullptr) { (void)(__bridge_transfer MLDictionaryFeatureProvider*)provider_; provider_ = nullptr; }
  if (out_arr_  != nullptr) { (void)(__bridge_transfer MLMultiArray*)out_arr_;  out_arr_  = nullptr; }
  if (in_arr_   != nullptr) { (void)(__bridge_transfer MLMultiArray*)in_arr_;   in_arr_   = nullptr; }
  if (model_    != nullptr) { (void)(__bridge_transfer MLModel*)model_;         model_    = nullptr; }
}

Status AppleAneContext::Execute(const float* in, float* out) {
  if (in == nullptr || out == nullptr) {
    return {StatusCode::kInvalidArgument, "apple_ane: null in/out"};
  }
  if (model_ == nullptr || in_arr_ == nullptr || out_arr_ == nullptr ||
      provider_ == nullptr || options_ == nullptr) {
    return {StatusCode::kInternal, "apple_ane: context not initialized"};
  }

  static const bool debug = std::getenv("ESM_APPLE_ANE_DEBUG") != nullptr;
  auto t0 = std::chrono::steady_clock::now();
  auto stamp = [&t0, this](const char* label) {
    auto t = std::chrono::steady_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t - t0)
                    .count();
    std::fprintf(stderr, "[ane M=%d K=%d N=%d] %-10s %.2f ms\n", m_, k_, n_,
                 label, us / 1e3);
    t0 = t;
  };

  @autoreleasepool {
    MLModel* model      = (__bridge MLModel*)model_;
    MLMultiArray* xArr  = (__bridge MLMultiArray*)in_arr_;
    MLMultiArray* yArr  = (__bridge MLMultiArray*)out_arr_;
    id<MLFeatureProvider> provider = (__bridge id<MLFeatureProvider>)provider_;
    MLPredictionOptions* opts      = (__bridge MLPredictionOptions*)options_;

    // Copy caller's fp32 input into the pre-allocated MLMultiArray (which is
    // in CoreML/ANE-friendly memory). For M=8192 K=5120 this is ~160 MB and
    // takes a few ms, but it's the entry the runtime can map zero-copy to ANE.
    std::memcpy(xArr.dataPointer, in,
                static_cast<std::size_t>(m_) * k_ * sizeof(float));
    if (debug) stamp("copy_in");

    NSError* err = nil;
    id<MLFeatureProvider> result =
        [model predictionFromFeatures:provider options:opts error:&err];
    if (debug) stamp("predict");
    if (result == nil || err != nil) {
      return {StatusCode::kInternal, "apple_ane: predictionFromFeatures failed"};
    }
    // With outputBackings set, the result's "out" feature aliases yArr.
    // Vectorized fp16 -> fp32 into the caller's `out` (Accelerate ~30x faster
    // than a scalar __fp16-cast loop for tens of MB).
    Fp16ToFp32((const __fp16*)yArr.dataPointer, out,
               static_cast<std::size_t>(m_) * n_);
    if (debug) stamp("copy_out");
    return Status::Ok();
  }
}

}  // namespace esm

#else  // !ESM_APPLE_ANE_AVAILABLE

namespace esm {
std::unique_ptr<AppleAneContext> AppleAneContext::LoadFromDir(
    const std::string& /*dir*/, int /*M*/, int /*K*/, int /*N*/) {
  return nullptr;
}
AppleAneContext::~AppleAneContext() = default;
Status AppleAneContext::Execute(const float* /*in*/, float* /*out*/) {
  return {StatusCode::kInternal, "apple_ane: unavailable on this build"};
}
}  // namespace esm

#endif  // ESM_APPLE_ANE_AVAILABLE
