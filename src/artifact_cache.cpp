#include "esm_cpp/artifact_cache.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace esm {

namespace fs = std::filesystem;

namespace {

// True if `s` looks like the HF cache directory naming convention
// `models--<id>--<name>` (org--repo).
bool IsHfModelsDir(const std::string& s) {
  return s.rfind("models--", 0) == 0;
}

// Walk up `weights_path`'s parents looking for an `models--<id>--<name>`
// directory under a `snapshots/<hash>/` directory. Returns the
// `<id>--<name>` portion, or empty string if no match.
std::string TryHfCacheKey(const fs::path& weights_path) {
  // The pattern: .../models--<id>--<name>/snapshots/<hash>/<file>
  // The file's immediate parent is <hash>, its grandparent is `snapshots`,
  // and its great-grandparent is the `models--<id>--<name>` dir.
  fs::path p = weights_path.parent_path();         // <hash>
  if (p.empty()) return "";
  fs::path snaps = p.parent_path();                // snapshots
  if (snaps.empty() || snaps.filename() != "snapshots") return "";
  fs::path models = snaps.parent_path();           // models--<id>--<name>
  if (models.empty()) return "";
  const std::string name = models.filename().string();
  if (!IsHfModelsDir(name)) return "";
  return name.substr(std::string("models--").size());
}

}  // namespace

std::string ArtifactCache::CacheKeyFor(const fs::path& weights_path) {
  if (auto hf = TryHfCacheKey(weights_path); !hf.empty()) return hf;
  return weights_path.stem().string();
}

fs::path ArtifactCache::DefaultCacheDir() {
#if defined(_WIN32)
  if (const char* lap = std::getenv("LOCALAPPDATA"); lap && *lap) {
    return fs::path(lap) / "esm_cpp";
  }
  if (const char* up = std::getenv("USERPROFILE"); up && *up) {
    return fs::path(up) / ".cache" / "esm_cpp";
  }
#else
  if (const char* home = std::getenv("HOME"); home && *home) {
    return fs::path(home) / ".cache" / "esm_cpp";
  }
#endif
  // Last-resort fallback so the function always returns something absolute.
  return fs::temp_directory_path() / "esm_cpp";
}

fs::path ArtifactCache::EffectiveCacheDir() {
  if (const char* env = std::getenv("ESM_CPP_CACHE_DIR"); env && *env) {
    return fs::path(env);
  }
  return DefaultCacheDir();
}

std::vector<fs::path> ArtifactCache::CandidateRoots(const fs::path& weights_path) {
  std::vector<fs::path> out;
  std::error_code ec;

  // 1. Sibling: <weights_dir>/<weights_stem>.apple/
  const fs::path sibling =
      weights_path.parent_path() / (weights_path.stem().string() + ".apple");
  if (fs::is_directory(sibling, ec)) out.push_back(sibling);

  // 2. Cache: <cache_root>/<cache_key>/
  const fs::path cache_root = EffectiveCacheDir();
  const fs::path cache_entry = cache_root / CacheKeyFor(weights_path);
  if (fs::is_directory(cache_entry, ec)) out.push_back(cache_entry);

  return out;
}

}  // namespace esm
