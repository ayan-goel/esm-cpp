#pragma once

// Phase 14: artifact discovery for the auto-engage flow.
//
// Apple-side speedups (AMX-fp16 per-Linear artifacts, whole-graph CoreML
// `.mlmodelc`) need on-disk artifacts to be reachable from Model::Load*
// without the user calling LoadAmxArtifacts(...) / LoadWholeGraphArtifact(...)
// by hand. This helper centralizes WHERE those artifacts live and HOW the
// cache key is derived from a weights path.
//
// Two well-known locations, checked in order at Model::Load*:
//   1. Sibling: `<weights_dir>/<weights_basename>.apple/{amx-fp16,whole-graph}/`
//      — user-managed; convenient for one-off builds.
//   2. Cache:   `<cache_root>/<cache_key>/{amx-fp16,whole-graph}/`
//      — CLI-managed by `esm-cpp-fetch-artifacts`.
//
// `<cache_root>` is `$ESM_CPP_CACHE_DIR` if set, else platform default:
//   - macOS/Linux: `$HOME/.cache/esm_cpp/`
//   - Windows:     `%LOCALAPPDATA%\esm_cpp\`
//
// `<cache_key>` derivation:
//   - If the weights path matches the HF cache pattern
//     `models--<id>--<name>/snapshots/<hash>/model.safetensors`, the key is
//     `<id>--<name>` (e.g., `facebook--esm2_t33_650M_UR50D`).
//   - Otherwise it's the weights file's stem (e.g.,
//     `/weights/esm2_650m.gguf` -> `esm2_650m`).
//
// Pure path math + filesystem existence — no I/O of artifacts themselves.
// Compiles + runs identically on Linux, no Apple deps. The downstream
// auto-engage code (T2/T3 in Phase 14) ifdef-gates the actual artifact load
// on Apple as before; CandidateRoots just returns paths.

#include <filesystem>
#include <string>
#include <vector>

namespace esm {

class ArtifactCache {
 public:
  // Resolve a stable cache key from a weights path. Never throws — falls
  // back to `path.stem()` if no HF cache pattern is recognized. Result is
  // safe to use as a directory name on all supported platforms.
  static std::string CacheKeyFor(const std::filesystem::path& weights_path);

  // Returns the candidate artifact root directories, in priority order
  // (sibling first, cache second), filtered to those that actually exist.
  // The caller is expected to look for `<root>/amx-fp16/`,
  // `<root>/whole-graph/B-X_L-Y/...`, etc. inside each entry.
  static std::vector<std::filesystem::path> CandidateRoots(
      const std::filesystem::path& weights_path);

  // The platform-default cache root (no env override applied). Useful for
  // tests + diagnostics. Trailing component is always `esm_cpp` so the
  // caller can identify a misconfiguration.
  static std::filesystem::path DefaultCacheDir();

  // The cache root actually in use (env override > default). Test helper.
  static std::filesystem::path EffectiveCacheDir();
};

}  // namespace esm
