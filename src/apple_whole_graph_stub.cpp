// Non-Apple stubs for AppleWholeGraphContext. Sibling of apple_ane_stub.cpp —
// real impl lives in src/apple_whole_graph.mm and is only added to the
// source list on Apple builds. This file provides the link surface on
// every other host so callers of AppleWholeGraphContext::LoadFromDir
// (Model::LoadWholeGraphArtifact, ForwardWholeGraph) link cleanly.

#include "esm_cpp/apple_whole_graph.h"

#ifndef ESM_APPLE_ANE_AVAILABLE

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
  return {StatusCode::kInternal,
          "apple_whole_graph: unavailable on this build"};
}

}  // namespace esm

#endif  // !ESM_APPLE_ANE_AVAILABLE
