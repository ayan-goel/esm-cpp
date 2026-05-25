// Non-Apple stubs for AppleAneContext. The real Obj-C++ implementation lives
// in src/apple_ane.mm and is only compiled when CMake sees an Apple host
// (it links against CoreML.framework). This .cpp ships in the source list
// unconditionally so Model::LoadAneArtifacts and friends link cleanly on
// Linux ARM / Graviton / Windows.
//
// `#ifdef ESM_APPLE_ANE_AVAILABLE` keeps the body empty on Apple — the .mm
// file owns those symbols there. The .cpp is harmless dead code on Apple
// and provides the actual link surface elsewhere.

#include "esm_cpp/apple_ane.h"

#ifndef ESM_APPLE_ANE_AVAILABLE

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

#endif  // !ESM_APPLE_ANE_AVAILABLE
