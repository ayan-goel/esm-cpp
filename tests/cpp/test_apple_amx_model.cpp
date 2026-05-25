// Phase 11 T6: model-level parity test for the Apple-AMX fp16 BNNSGraph path.
//
// The AMX backend is a black box — there is no scalar reference for BNNS — so
// the validation has to live at the model level: load a real ESM-2 checkpoint,
// run the FP32 forward as the reference, then load the per-Linear fp16
// mlmodelc artifacts, flip ESM_APPLE_AMX=on, and confirm the logits stay
// finite, correlate ≥ 0.999 with the FP32 reference, and agree on the argmax
// at ≥ 99 %. Mirrors the GemmShapes.AppleAmxFp32MatchesRef pattern, but
// end-to-end (Phase 9 promoted this style of validation; this test extends
// it to the new fp16 path).
//
// Fixture: 8M is the smallest real ESM-2; the artifacts at /tmp/amx_8m are
// produced by tools/build_amx_artifacts.py (run once locally). For CI / cold
// checkouts, both paths are env-var-driven and the test SKIPs cleanly if
// either fixture is absent.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <span>
#include <string>
#include <unistd.h>
#include <vector>

#include "esm_cpp/model.h"
#include "esm_cpp/tokenizer.h"

namespace {

// RAII env-var override (same pattern as test_artifact_cache.cpp).
class EnvScope {
 public:
  EnvScope(const std::string& key, const std::string& value) : key_(key) {
    if (const char* prev = std::getenv(key.c_str())) {
      had_prev_ = true;
      prev_ = prev;
    }
    ::setenv(key.c_str(), value.c_str(), 1);
  }
  EnvScope(const std::string& key, std::nullptr_t) : key_(key) {
    if (const char* prev = std::getenv(key.c_str())) {
      had_prev_ = true;
      prev_ = prev;
    }
    ::unsetenv(key.c_str());
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
    path_ = std::filesystem::temp_directory_path() /
        ("esm_cpp_auto_load_test_" + std::to_string(::getpid()) + "_" +
         std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  const std::filesystem::path& path() const { return path_; }
 private:
  std::filesystem::path path_;
};

// Pearson correlation, computed in double to avoid catastrophic cancellation
// on logit-magnitude vectors (~1e2 dynamic range).
double Correlate(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = a.size();
  if (n == 0 || a.size() != b.size()) return 0.0;
  double sa = 0, sb = 0;
  for (std::size_t i = 0; i < n; ++i) {
    sa += a[i];
    sb += b[i];
  }
  const double ma = sa / static_cast<double>(n);
  const double mb = sb / static_cast<double>(n);
  double num = 0, da = 0, db = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double xa = a[i] - ma;
    const double xb = b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  if (da == 0.0 || db == 0.0) return 0.0;
  return num / std::sqrt(da * db);
}

// Per-row argmax agreement (logits[L, V] -> L predictions; reduce to L).
double ArgmaxAgreement(const std::vector<float>& a, const std::vector<float>& b,
                       int L, int V) {
  if (L <= 0 || V <= 0 ||
      a.size() != static_cast<std::size_t>(L) * V ||
      b.size() != static_cast<std::size_t>(L) * V) {
    return 0.0;
  }
  int agree = 0;
  for (int l = 0; l < L; ++l) {
    int ai = 0, bi = 0;
    float av = a[static_cast<std::size_t>(l) * V];
    float bv = b[static_cast<std::size_t>(l) * V];
    for (int v = 1; v < V; ++v) {
      const float aa = a[static_cast<std::size_t>(l) * V + v];
      const float bb = b[static_cast<std::size_t>(l) * V + v];
      if (aa > av) { av = aa; ai = v; }
      if (bb > bv) { bv = bb; bi = v; }
    }
    if (ai == bi) ++agree;
  }
  return static_cast<double>(agree) / L;
}

bool AllFinite(const std::vector<float>& v) {
  for (float x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

}  // namespace

// FP32 forward vs FP32+AMX-fp16 forward: model-level parity guard. The AMX
// artifacts have bias baked into the compiled graph, so a bug in either the
// integration (missing/extra bias, transposed weight) shows up as a
// correlation collapse.
TEST(AppleAmx, Fp16ModelParityVsFp32Reference) {
  const char* sft = std::getenv("ESM_AMX_TEST_SAFETENSORS");
  const char* art = std::getenv("ESM_AMX_TEST_ARTIFACT_DIR");
  if (!sft || !art) {
    GTEST_SKIP()
        << "Set ESM_AMX_TEST_SAFETENSORS (path to a small ESM-2 model.safetensors) "
        << "and ESM_AMX_TEST_ARTIFACT_DIR (path to the matching "
        << ".amx-fp16/ directory built by tools/build_amx_artifacts.py).";
  }
  if (!std::filesystem::exists(sft) || !std::filesystem::is_directory(art)) {
    GTEST_SKIP() << "fixture paths do not resolve on disk: " << sft << " / " << art;
  }

  auto model = esm::Model::LoadFromSafetensors(sft);
  ASSERT_NE(model, nullptr);
  esm::Tokenizer tok;

  // A deterministic 64aa probe sequence (taken from a real ESM-2 sanity input).
  const std::string seq =
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG";
  const auto ids = tok.Encode(seq);
  std::vector<std::int32_t> mask(ids.size(), 1);

  // FP32 reference (AMX flag off — make sure even an inherited env doesn't
  // taint the reference run).
  const char* prev_flag = std::getenv("ESM_APPLE_AMX");
  std::string saved_flag = prev_flag ? prev_flag : "";
  ::unsetenv("ESM_APPLE_AMX");
  const auto ref = model->Forward(std::span<const std::int32_t>(ids),
                                   std::span<const std::int32_t>(mask));
  ASSERT_TRUE(AllFinite(ref)) << "FP32 reference has NaN/Inf";

  // Load artifacts + flip the flag.
  const std::size_t loaded = model->LoadAmxArtifacts(art);
#ifdef ESM_APPLE_AMX_AVAILABLE
  ASSERT_GT(loaded, 0u) << "no AMX artifacts loaded — fixture mismatched?";
#else
  if (loaded == 0) GTEST_SKIP() << "Apple-AMX not available on this build";
#endif
  ::setenv("ESM_APPLE_AMX", "on", 1);
  const auto amx = model->Forward(std::span<const std::int32_t>(ids),
                                   std::span<const std::int32_t>(mask));

  // Restore the original env state for the next test.
  if (prev_flag) {
    ::setenv("ESM_APPLE_AMX", saved_flag.c_str(), 1);
  } else {
    ::unsetenv("ESM_APPLE_AMX");
  }

  ASSERT_EQ(amx.size(), ref.size());
  EXPECT_TRUE(AllFinite(amx)) << "AMX logits contain NaN/Inf";

  const double corr = Correlate(std::span<const float>(ref),
                                 std::span<const float>(amx));
  EXPECT_GE(corr, 0.999) << "logit corr below 0.999 — AMX path drifted";

  const int V = model->config().vocab_size;
  const int L = static_cast<int>(amx.size() / static_cast<std::size_t>(V));
  const double agree = ArgmaxAgreement(ref, amx, L, V);
  EXPECT_GE(agree, 0.99) << "argmax agreement below 0.99 — AMX path drifted";
}

// Phase 14 T2: auto-load AMX from the sibling artifact dir at Model::Load*
// without an explicit LoadAmxArtifacts(...) call. Symlinks the existing
// AMX fixture into a temp dir as `<weights>.apple/amx-fp16/` so the
// auto-discovery in TryAutoLoadAppleArtifacts picks it up.
TEST(AppleAmx, AutoLoadFromSiblingDir) {
  const char* sft = std::getenv("ESM_AMX_TEST_SAFETENSORS");
  const char* art = std::getenv("ESM_AMX_TEST_ARTIFACT_DIR");
  if (!sft || !art) {
    GTEST_SKIP() << "fixture envs not set (ESM_AMX_TEST_SAFETENSORS, "
                 << "ESM_AMX_TEST_ARTIFACT_DIR)";
  }
  if (!std::filesystem::exists(sft) || !std::filesystem::is_directory(art)) {
    GTEST_SKIP() << "fixture paths do not resolve: " << sft << " / " << art;
  }

  namespace fs = std::filesystem;
  TempDir td;
  // Symlink targets must be absolute — a symlink to a relative path is
  // resolved relative to the symlink's location, not $PWD.
  const fs::path sft_abs = fs::absolute(sft);
  const fs::path art_abs = fs::absolute(art);
  const fs::path weights = td.path() / "model.safetensors";
  std::error_code ec;
  fs::create_symlink(sft_abs, weights, ec);
  ASSERT_FALSE(ec) << "symlink weights failed: " << ec.message();

  const fs::path apple_dir = td.path() / "model.apple";
  fs::create_directories(apple_dir);
  const fs::path amx_link = apple_dir / "amx-fp16";
  fs::create_directory_symlink(art_abs, amx_link, ec);
  ASSERT_FALSE(ec) << "symlink amx fixture failed: " << ec.message();

  // Override the cache dir to nowhere so only the sibling path is found.
  EnvScope cache_off("ESM_CPP_CACHE_DIR", (td.path() / "nope").string());
  // Leave ESM_APPLE_AMX unset so the new default-on semantics apply.
  EnvScope amx_default("ESM_APPLE_AMX", nullptr);

  auto model = esm::Model::LoadFromSafetensors(weights.string());
  ASSERT_NE(model, nullptr);
#ifdef ESM_APPLE_AMX_AVAILABLE
  EXPECT_FALSE(model->amx_artifacts_path().empty())
      << "auto-discovery did not engage on sibling dir";
  EXPECT_EQ(model->amx_artifacts_path(), amx_link.string());
#else
  EXPECT_TRUE(model->amx_artifacts_path().empty())
      << "non-Apple build should not auto-load";
#endif
}

// Phase 14 T2: ESM_APPLE_AMX=off disables auto-engage even when artifacts
// were explicitly loaded. We still LOAD them (the loader doesn't check
// env); the runtime gate is the one that gets disabled.
TEST(AppleAmx, EnvOffDisablesAutoEngage) {
  const char* sft = std::getenv("ESM_AMX_TEST_SAFETENSORS");
  const char* art = std::getenv("ESM_AMX_TEST_ARTIFACT_DIR");
  if (!sft || !art) GTEST_SKIP() << "fixture envs not set";
  if (!std::filesystem::exists(sft) || !std::filesystem::is_directory(art)) {
    GTEST_SKIP() << "fixture paths do not resolve";
  }

  EnvScope amx_off("ESM_APPLE_AMX", "off");
  auto model = esm::Model::LoadFromSafetensors(sft);
  ASSERT_NE(model, nullptr);
  model->LoadAmxArtifacts(art);  // explicit load still works

  esm::Tokenizer tok;
  const std::string seq =
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG";
  const auto ids = tok.Encode(seq);
  std::vector<std::int32_t> mask(ids.size(), 1);
  const auto logits = model->Forward(std::span<const std::int32_t>(ids),
                                      std::span<const std::int32_t>(mask));
  // Forward must still produce finite logits via the FP32 fallback path
  // (AMX gated off by env, even though artifacts are loaded).
  EXPECT_TRUE(AllFinite(logits));
}
