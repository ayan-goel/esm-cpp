// Phase 14 T1: tests for esm::ArtifactCache. Pure path math + filesystem
// existence — no Apple deps, no Model, runs identically on Linux + macOS.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "esm_cpp/artifact_cache.h"

namespace fs = std::filesystem;

namespace {

// RAII wrapper to scrub + restore an env var across a test.
class EnvScope {
 public:
  EnvScope(const std::string& key, const std::string& value)
      : key_(key) {
    if (const char* prev = std::getenv(key.c_str())) {
      had_prev_ = true;
      prev_ = prev;
    }
    ::setenv(key.c_str(), value.c_str(), 1);
  }
  ~EnvScope() {
    if (had_prev_) ::setenv(key_.c_str(), prev_.c_str(), 1);
    else            ::unsetenv(key_.c_str());
  }
 private:
  std::string key_;
  std::string prev_;
  bool had_prev_ = false;
};

class TempDir {
 public:
  TempDir() {
    path_ = fs::temp_directory_path() /
        ("esm_cpp_artifact_cache_test_" + std::to_string(::getpid()) + "_" +
         std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    fs::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  const fs::path& path() const { return path_; }
 private:
  fs::path path_;
};

}  // namespace

TEST(ArtifactCache, CacheKeyFromHfCachePath) {
  // HF cache pattern: .../models--<id>--<name>/snapshots/<hash>/model.safetensors
  const fs::path p = "/home/u/.cache/huggingface/hub/"
                     "models--facebook--esm2_t33_650M_UR50D/snapshots/"
                     "08e4846e537177426273712802403f7ba8261b6c/model.safetensors";
  EXPECT_EQ(esm::ArtifactCache::CacheKeyFor(p), "facebook--esm2_t33_650M_UR50D");
}

TEST(ArtifactCache, CacheKeyFallsBackToStemForArbitraryPath) {
  const fs::path p = "/some/user/dir/esm2_650m.safetensors";
  EXPECT_EQ(esm::ArtifactCache::CacheKeyFor(p), "esm2_650m");
}

TEST(ArtifactCache, CacheKeyForGgufPath) {
  // GGUF files use the same fallback.
  const fs::path p = "/weights/esm2_8m.gguf";
  EXPECT_EQ(esm::ArtifactCache::CacheKeyFor(p), "esm2_8m");
}

TEST(ArtifactCache, CandidateRootsFiltersNonExistent) {
  TempDir td;
  const fs::path weights = td.path() / "esm2_8m.safetensors";
  { std::ofstream(weights) << "dummy"; }

  // Override the cache root to a non-existent path so cache lookup is empty.
  EnvScope override_cache("ESM_CPP_CACHE_DIR", (td.path() / "nope").string());

  // No sibling .apple/ either; expect empty.
  const auto roots = esm::ArtifactCache::CandidateRoots(weights);
  EXPECT_TRUE(roots.empty());
}

TEST(ArtifactCache, CandidateRootsFindsSibling) {
  TempDir td;
  const fs::path weights = td.path() / "esm2_8m.safetensors";
  { std::ofstream(weights) << "dummy"; }
  const fs::path sibling = td.path() / "esm2_8m.apple";
  fs::create_directories(sibling);

  EnvScope override_cache("ESM_CPP_CACHE_DIR", (td.path() / "nope").string());

  const auto roots = esm::ArtifactCache::CandidateRoots(weights);
  ASSERT_EQ(roots.size(), 1u);
  EXPECT_EQ(roots[0], sibling);
}

TEST(ArtifactCache, CandidateRootsFindsCache) {
  TempDir td;
  const fs::path weights = td.path() / "esm2_8m.safetensors";
  { std::ofstream(weights) << "dummy"; }
  // Cache layout: $ESM_CPP_CACHE_DIR/<key>/  -- exists, so it should match.
  const fs::path cache_root = td.path() / "cache";
  const fs::path cache_entry = cache_root / "esm2_8m";
  fs::create_directories(cache_entry);

  EnvScope override_cache("ESM_CPP_CACHE_DIR", cache_root.string());

  const auto roots = esm::ArtifactCache::CandidateRoots(weights);
  ASSERT_EQ(roots.size(), 1u);
  EXPECT_EQ(roots[0], cache_entry);
}

TEST(ArtifactCache, CandidateRootsOrderingSiblingBeforeCache) {
  TempDir td;
  const fs::path weights = td.path() / "esm2_8m.safetensors";
  { std::ofstream(weights) << "dummy"; }
  // Both sibling and cache exist; sibling wins (first).
  const fs::path sibling = td.path() / "esm2_8m.apple";
  fs::create_directories(sibling);
  const fs::path cache_root = td.path() / "cache";
  const fs::path cache_entry = cache_root / "esm2_8m";
  fs::create_directories(cache_entry);

  EnvScope override_cache("ESM_CPP_CACHE_DIR", cache_root.string());

  const auto roots = esm::ArtifactCache::CandidateRoots(weights);
  ASSERT_EQ(roots.size(), 2u);
  EXPECT_EQ(roots[0], sibling);
  EXPECT_EQ(roots[1], cache_entry);
}

TEST(ArtifactCache, DefaultCacheDirUsesHomeWhenEnvUnset) {
  ::unsetenv("ESM_CPP_CACHE_DIR");
  const fs::path defaulted = esm::ArtifactCache::DefaultCacheDir();
  // On Apple/Linux this is $HOME/.cache/esm_cpp; on Windows $USERPROFILE/.cache/esm_cpp.
  // Either way, must be non-empty and absolute.
  EXPECT_FALSE(defaulted.empty());
  EXPECT_TRUE(defaulted.is_absolute()) << "got: " << defaulted;
  EXPECT_NE(defaulted.string().find("esm_cpp"), std::string::npos)
      << "got: " << defaulted;
}
