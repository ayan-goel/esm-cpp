#include "esm_cpp/apple_ane.h"

#ifdef ESM_APPLE_ANE_AVAILABLE

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include <cstring>

// Per-Linear, per-bucket Neural Engine runtime via CoreML's MLModel API.
// BNNSGraph (the Phase-11 AMX path) is CPU/AMX-only — there is no compute-unit
// selector in its API — so ANE has to go through MLModel. The graph itself
// is the same single-op `linear` MLModel that build_amx_artifacts.py emits with
// --compute-units CPU_AND_NE and a fixed M. Per-call overhead is dominated by
// the input/output MLMultiArray construction; we zero-copy the input via
// `initWithDataPointer:` and only copy the output back into the caller's
// buffer.

namespace esm {

namespace {

// Wrap the caller's contiguous fp32 buffer in an MLMultiArray view (no copy).
// `dataPtr` lifetime must outlive the MLMultiArray's use inside predict.
MLMultiArray* WrapFp32Input(const float* dataPtr, int M, int K) {
  NSArray<NSNumber*>* shape =
      @[ @(M), @(K) ];
  // Row-major: stride[0] = K, stride[1] = 1.
  NSArray<NSNumber*>* strides = @[ @(K), @(1) ];
  NSError* err = nil;
  // initWithDataPointer doesn't take ownership; pass a no-op deallocator.
  MLMultiArray* a =
      [[MLMultiArray alloc] initWithDataPointer:(void*)dataPtr
                                          shape:shape
                                       dataType:MLMultiArrayDataTypeFloat32
                                        strides:strides
                                    deallocator:^(void* /*ptr*/) {}
                                          error:&err];
  if (err) return nil;
  return a;
}

// Copy MLMultiArray (any supported numeric type) into a contiguous fp32 buffer
// of size M*N. ANE may return fp16; CoreML up-casts at access time but we go
// through the explicit getBytesWithHandler: path for safety.
bool CopyOutFp32(MLMultiArray* a, float* out, int M, int N) {
  if (a == nil) return false;
  if (a.count != (NSInteger)((long)M * N)) return false;
  // Walk the most common dtypes; fall back through KVC float access if needed.
  const NSInteger total = a.count;
  switch (a.dataType) {
    case MLMultiArrayDataTypeFloat32: {
      const float* p = (const float*)a.dataPointer;
      std::memcpy(out, p, total * sizeof(float));
      return true;
    }
    case MLMultiArrayDataTypeFloat16: {
      // ARM ABI defines __fp16 as IEEE half; safe cast.
      const __fp16* p = (const __fp16*)a.dataPointer;
      for (NSInteger i = 0; i < total; ++i) out[i] = (float)p[i];
      return true;
    }
    case MLMultiArrayDataTypeDouble: {
      const double* p = (const double*)a.dataPointer;
      for (NSInteger i = 0; i < total; ++i) out[i] = (float)p[i];
      return true;
    }
    default:
      return false;
  }
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

    auto out = std::unique_ptr<AppleAneContext>(new AppleAneContext());
    // Bridge-retain across the ARC/non-ARC boundary; ~AppleAneContext releases.
    out->model_ = (__bridge_retained void*)model;
    out->m_ = M;
    out->k_ = K;
    out->n_ = N;
    return out;
  }
}

AppleAneContext::~AppleAneContext() {
  if (model_ != nullptr) {
    // Transfer ownership back to ARC so the MLModel deallocs.
    MLModel* m = (__bridge_transfer MLModel*)model_;
    (void)m;
    model_ = nullptr;
  }
}

Status AppleAneContext::Execute(const float* in, float* out) {
  if (in == nullptr || out == nullptr) {
    return {StatusCode::kInvalidArgument, "apple_ane: null in/out"};
  }
  if (model_ == nullptr) {
    return {StatusCode::kInternal, "apple_ane: no model"};
  }

  @autoreleasepool {
    MLModel* model = (__bridge MLModel*)model_;
    MLMultiArray* xArr = WrapFp32Input(in, m_, k_);
    if (xArr == nil) {
      return {StatusCode::kInternal, "apple_ane: WrapFp32Input failed"};
    }
    NSError* err = nil;
    NSDictionary<NSString*, MLMultiArray*>* inputs = @{ @"x": xArr };
    MLDictionaryFeatureProvider* provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs
                                                          error:&err];
    if (provider == nil || err != nil) {
      return {StatusCode::kInternal,
              "apple_ane: MLDictionaryFeatureProvider init failed"};
    }
    id<MLFeatureProvider> result =
        [model predictionFromFeatures:provider error:&err];
    if (result == nil || err != nil) {
      return {StatusCode::kInternal, "apple_ane: predictionFromFeatures failed"};
    }
    MLFeatureValue* outVal = [result featureValueForName:@"out"];
    MLMultiArray* outArr = outVal != nil ? outVal.multiArrayValue : nil;
    if (!CopyOutFp32(outArr, out, m_, n_)) {
      return {StatusCode::kInternal, "apple_ane: output extract failed"};
    }
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
