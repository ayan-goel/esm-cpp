#include "esm_cpp/apple_whole_graph.h"

#ifdef ESM_APPLE_ANE_AVAILABLE

#import <Accelerate/Accelerate.h>
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Phase 13: whole-model CoreML runtime. ONE MLModel for the entire ESM-2
// forward (encoder + LM head), built at convert time from a clean traced
// PyTorch wrapper (tools/esm_traceable.py). Mirrors the AppleAneContext
// (Phase 12) bridge pattern: pre-allocated MLMultiArrays + outputBackings +
// vectorized fp16->fp32 conversion. The architectural fix vs Phase 12 is
// that there's ONE model kept hot, not 198 tiny ones that thrash the ANE
// compiled-state cache (see notes/phase12.md).

namespace esm {

namespace {

// Vectorized fp16 -> fp32 conversion via Accelerate's vImage.
void Fp16ToFp32(const __fp16* src, float* dst, std::size_t n) {
  vImage_Buffer in_buf  = {(void*)src, 1, n, n * sizeof(__fp16)};
  vImage_Buffer out_buf = {dst,        1, n, n * sizeof(float)};
  vImageConvert_Planar16FtoPlanarF(&in_buf, &out_buf, 0);
}

MLComputeUnits ToMLComputeUnits(WholeGraphComputeUnits u) {
  switch (u) {
    case WholeGraphComputeUnits::kCpuOnly:
      return MLComputeUnitsCPUOnly;
    case WholeGraphComputeUnits::kCpuAndNeuralEngine:
      return MLComputeUnitsCPUAndNeuralEngine;
    case WholeGraphComputeUnits::kAll:
      return MLComputeUnitsAll;
  }
  return MLComputeUnitsCPUAndNeuralEngine;
}

}  // namespace

std::unique_ptr<AppleWholeGraphContext> AppleWholeGraphContext::LoadFromDir(
    const std::string& dir, int B, int L, int V,
    WholeGraphComputeUnits compute_units) {
  if (B <= 0 || L <= 0 || V <= 0) return nullptr;

  @autoreleasepool {
    NSString* path = [NSString stringWithUTF8String:dir.c_str()];
    NSURL* url = [NSURL fileURLWithPath:path];
    MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = ToMLComputeUnits(compute_units);
    NSError* err = nil;
    MLModel* model =
        [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
    if (model == nil || err != nil) return nullptr;

    NSArray* in_shape = @[ @(B), @(L) ];
    NSArray* out_shape = @[ @(B), @(L), @(V) ];

    MLMultiArray* ids_arr =
        [[MLMultiArray alloc] initWithShape:in_shape
                                   dataType:MLMultiArrayDataTypeInt32
                                      error:&err];
    if (ids_arr == nil || err != nil) return nullptr;
    MLMultiArray* mask_arr =
        [[MLMultiArray alloc] initWithShape:in_shape
                                   dataType:MLMultiArrayDataTypeInt32
                                      error:&err];
    if (mask_arr == nil || err != nil) return nullptr;
    // Output is fp16 (ANE/GPU native); vectorize fp16->fp32 in Execute.
    MLMultiArray* out_arr =
        [[MLMultiArray alloc] initWithShape:out_shape
                                   dataType:MLMultiArrayDataTypeFloat16
                                      error:&err];
    if (out_arr == nil || err != nil) return nullptr;

    MLDictionaryFeatureProvider* provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
          @"input_ids": ids_arr,
          @"attention_mask": mask_arr,
        } error:&err];
    if (provider == nil || err != nil) return nullptr;

    MLPredictionOptions* opts = [[MLPredictionOptions alloc] init];
    opts.outputBackings = @{ @"logits": out_arr };

    auto out = std::unique_ptr<AppleWholeGraphContext>(new AppleWholeGraphContext());
    out->model_    = (__bridge_retained void*)model;
    out->ids_arr_  = (__bridge_retained void*)ids_arr;
    out->mask_arr_ = (__bridge_retained void*)mask_arr;
    out->out_arr_  = (__bridge_retained void*)out_arr;
    out->provider_ = (__bridge_retained void*)provider;
    out->options_  = (__bridge_retained void*)opts;
    out->b_ = B;
    out->l_ = L;
    out->v_ = V;
    return out;
  }
}

AppleWholeGraphContext::~AppleWholeGraphContext() {
  if (options_  != nullptr) { (void)(__bridge_transfer MLPredictionOptions*)options_;  options_  = nullptr; }
  if (provider_ != nullptr) { (void)(__bridge_transfer MLDictionaryFeatureProvider*)provider_; provider_ = nullptr; }
  if (out_arr_  != nullptr) { (void)(__bridge_transfer MLMultiArray*)out_arr_;  out_arr_  = nullptr; }
  if (mask_arr_ != nullptr) { (void)(__bridge_transfer MLMultiArray*)mask_arr_; mask_arr_ = nullptr; }
  if (ids_arr_  != nullptr) { (void)(__bridge_transfer MLMultiArray*)ids_arr_;  ids_arr_  = nullptr; }
  if (model_    != nullptr) { (void)(__bridge_transfer MLModel*)model_;         model_    = nullptr; }
}

Status AppleWholeGraphContext::Execute(const std::int32_t* input_ids,
                                       const std::int32_t* attention_mask,
                                       float* logits_out) {
  if (input_ids == nullptr || attention_mask == nullptr || logits_out == nullptr) {
    return {StatusCode::kInvalidArgument, "apple_whole_graph: null input/output"};
  }
  if (model_ == nullptr || ids_arr_ == nullptr || mask_arr_ == nullptr ||
      out_arr_ == nullptr || provider_ == nullptr || options_ == nullptr) {
    return {StatusCode::kInternal, "apple_whole_graph: context not initialized"};
  }

  static const bool debug = std::getenv("ESM_APPLE_ANE_GRAPH_DEBUG") != nullptr;
  auto t0 = std::chrono::steady_clock::now();
  auto stamp = [&t0, this](const char* label) {
    auto t = std::chrono::steady_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t - t0)
                    .count();
    std::fprintf(stderr, "[wg B=%d L=%d V=%d] %-10s %.2f ms\n", b_, l_, v_,
                 label, us / 1e3);
    t0 = t;
  };

  @autoreleasepool {
    MLModel* model      = (__bridge MLModel*)model_;
    MLMultiArray* ids   = (__bridge MLMultiArray*)ids_arr_;
    MLMultiArray* mask  = (__bridge MLMultiArray*)mask_arr_;
    MLMultiArray* yArr  = (__bridge MLMultiArray*)out_arr_;
    id<MLFeatureProvider> provider = (__bridge id<MLFeatureProvider>)provider_;
    MLPredictionOptions* opts      = (__bridge MLPredictionOptions*)options_;

    const std::size_t n_tokens = static_cast<std::size_t>(b_) * l_;
    std::memcpy(ids.dataPointer,  input_ids,      n_tokens * sizeof(std::int32_t));
    std::memcpy(mask.dataPointer, attention_mask, n_tokens * sizeof(std::int32_t));
    if (debug) stamp("copy_in");

    NSError* err = nil;
    id<MLFeatureProvider> result =
        [model predictionFromFeatures:provider options:opts error:&err];
    if (debug) stamp("predict");
    if (result == nil || err != nil) {
      return {StatusCode::kInternal, "apple_whole_graph: predictionFromFeatures failed"};
    }
    // outputBackings means result["logits"] aliases yArr.
    Fp16ToFp32((const __fp16*)yArr.dataPointer, logits_out,
               n_tokens * static_cast<std::size_t>(v_));
    if (debug) stamp("copy_out");
    return Status::Ok();
  }
}

}  // namespace esm

#else  // !ESM_APPLE_ANE_AVAILABLE

namespace esm {
std::unique_ptr<AppleWholeGraphContext> AppleWholeGraphContext::LoadFromDir(
    const std::string& /*dir*/, int /*B*/, int /*L*/, int /*V*/,
    WholeGraphComputeUnits /*u*/) {
  return nullptr;
}
AppleWholeGraphContext::~AppleWholeGraphContext() = default;
Status AppleWholeGraphContext::Execute(const std::int32_t* /*input_ids*/,
                                       const std::int32_t* /*attention_mask*/,
                                       float* /*logits_out*/) {
  return {StatusCode::kInternal, "apple_whole_graph: unavailable on this build"};
}
}  // namespace esm

#endif  // ESM_APPLE_ANE_AVAILABLE
