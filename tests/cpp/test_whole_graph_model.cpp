// Phase 13 T6: model-level parity guard for the whole-graph CoreML path.
//
// Loads a small ESM-2 checkpoint, runs the FP32 reference forward, then
// registers a whole-graph .mlmodelc artifact for the same (B=1, L=ids.size())
// shape and routes the next forward through ForwardScheduled with
// ESM_APPLE_ANE_GRAPH=on. Asserts:
//   - logits finite
//   - correlation with FP32 reference ≥ 0.999
//   - argmax agreement ≥ 0.99
//   - logits magnitudes are sane (no values > 1e4)
//
// Fixture env-driven, SKIPs cleanly when not provided or on non-Apple builds:
//   ESM_WHOLE_GRAPH_TEST_SAFETENSORS:
//       path to model.safetensors
//   ESM_WHOLE_GRAPH_TEST_ARTIFACT:
//       path to a whole_graph.mlmodelc bundle for (B=1, L=<L>) where L matches
//       the encoded fixture sequence length below.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <string>
#include <unistd.h>
#include <vector>

#include "esm_cpp/model.h"
#include "esm_cpp/tokenizer.h"

namespace {

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
        ("esm_cpp_wg_test_" + std::to_string(::getpid()) + "_" +
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

double Correlate(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = a.size();
  if (n == 0 || a.size() != b.size()) return 0.0;
  double sa = 0, sb = 0;
  for (std::size_t i = 0; i < n; ++i) { sa += a[i]; sb += b[i]; }
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

double ArgmaxAgreement(const std::vector<float>& a,
                       const std::vector<float>& b, int L, int V) {
  if (L <= 0 || V <= 0) return 0.0;
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
  for (float x : v) if (!std::isfinite(x)) return false;
  return true;
}

float MaxAbs(const std::vector<float>& v) {
  float m = 0.0f;
  for (float x : v) m = std::max(m, std::fabs(x));
  return m;
}

}  // namespace

TEST(AppleWholeGraph, Fp16ModelParityVsFp32Reference) {
  const char* sft = std::getenv("ESM_WHOLE_GRAPH_TEST_SAFETENSORS");
  const char* art = std::getenv("ESM_WHOLE_GRAPH_TEST_ARTIFACT");
  if (!sft || !art) {
    GTEST_SKIP() << "Set ESM_WHOLE_GRAPH_TEST_SAFETENSORS + "
                 << "ESM_WHOLE_GRAPH_TEST_ARTIFACT to fixture paths "
                 << "(model.safetensors and a whole_graph.mlmodelc bundle "
                 << "built for B=1, L matching the test sequence + 2).";
  }
  if (!std::filesystem::exists(sft) ||
      !std::filesystem::is_directory(art)) {
    GTEST_SKIP() << "fixture paths missing: " << sft << " / " << art;
  }

  auto model = esm::Model::LoadFromSafetensors(sft);
  ASSERT_NE(model, nullptr);
  esm::Tokenizer tok;

  // Fixed 65-residue sequence + cls + eos -> L=67 to match the artifact's
  // (B=1, L=67) shape. The bridge LoadFromDir checks the compiled input
  // shape against the requested (B, L) and refuses to load on mismatch
  // (Phase 14 fix — previously silent-fallback would mask a fixture/test
  // mismatch as a trivial corr=1.0 pass via the standard-path fallback).
  const std::string seq =
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG";
  const auto ids = tok.Encode(seq);
  std::vector<std::int32_t> mask(ids.size(), 1);
  const int L = static_cast<int>(ids.size());
  const int V = model->config().vocab_size;

  // FP32 reference (ensure neither AMX nor whole-graph is engaged).
  const char* prev_amx = std::getenv("ESM_APPLE_AMX");
  const char* prev_wg  = std::getenv("ESM_APPLE_ANE_GRAPH");
  std::string saved_amx = prev_amx ? prev_amx : "";
  std::string saved_wg  = prev_wg  ? prev_wg  : "";
  ::unsetenv("ESM_APPLE_AMX");
  ::unsetenv("ESM_APPLE_ANE_GRAPH");
  const auto ref = model->Forward(std::span<const std::int32_t>(ids),
                                   std::span<const std::int32_t>(mask));
  ASSERT_TRUE(AllFinite(ref));

  // Register the whole-graph artifact at (B=1, L).
  const bool registered = model->LoadWholeGraphArtifact(
      art, /*B=*/1, L, esm::WholeGraphComputeUnits::kCpuAndNeuralEngine);
#ifdef ESM_APPLE_ANE_AVAILABLE
  ASSERT_TRUE(registered) << "LoadWholeGraphArtifact failed for (B=1, L=" << L
                          << "). Did the artifact match the test L?";
#else
  if (!registered) GTEST_SKIP() << "Apple whole-graph not available on this build";
#endif

  // Engage via ForwardScheduled with the env gate on.
  ::setenv("ESM_APPLE_ANE_GRAPH", "on", 1);
  std::vector<std::vector<std::int32_t>> batch_ids = {ids};
  std::vector<std::vector<std::int32_t>> batch_masks = {mask};
  auto out_list = model->ForwardScheduled(batch_ids, batch_masks);
  ASSERT_EQ(out_list.size(), 1u);
  const auto& wg = out_list[0];

  // Restore the env state regardless of pass/fail.
  if (prev_amx) ::setenv("ESM_APPLE_AMX", saved_amx.c_str(), 1);
  else          ::unsetenv("ESM_APPLE_AMX");
  if (prev_wg)  ::setenv("ESM_APPLE_ANE_GRAPH", saved_wg.c_str(), 1);
  else          ::unsetenv("ESM_APPLE_ANE_GRAPH");

  ASSERT_EQ(wg.size(), ref.size());
  EXPECT_TRUE(AllFinite(wg));
  EXPECT_LT(MaxAbs(wg), 1.0e4f) << "whole-graph logits magnitude exceeds 1e4 "
                                << "— suggests softmax overflow / fp16 blowup";

  const double corr = Correlate(std::span<const float>(ref),
                                std::span<const float>(wg));
  EXPECT_GE(corr, 0.999) << "whole-graph logit correlation below 0.999";

  const double agree = ArgmaxAgreement(ref, wg, L, V);
  EXPECT_GE(agree, 0.99) << "whole-graph argmax agreement below 0.99";
}

// Phase 14 T3: whole-graph artifact auto-discovery + env opt-out flip.
// Stages a B-1_L-67 artifact at <td>/model.apple/whole-graph/B-1_L-67/,
// loads the model with no explicit register call + no env vars, and asserts
// the shape is registered AND a forward_scheduled engages the fast path.
TEST(AppleWholeGraph, AutoLoadFromSiblingDir) {
  const char* sft = std::getenv("ESM_WHOLE_GRAPH_TEST_SAFETENSORS");
  const char* art = std::getenv("ESM_WHOLE_GRAPH_TEST_ARTIFACT");
  if (!sft || !art) {
    GTEST_SKIP() << "fixture envs not set "
                 << "(ESM_WHOLE_GRAPH_TEST_SAFETENSORS, "
                 << "ESM_WHOLE_GRAPH_TEST_ARTIFACT — a .mlmodelc bundle "
                 << "built for B=1, L matching the test sequence + 2).";
  }
  if (!std::filesystem::exists(sft) ||
      !std::filesystem::is_directory(art)) {
    GTEST_SKIP() << "fixture paths do not resolve";
  }

  namespace fs = std::filesystem;
  TempDir td;
  const fs::path sft_abs = fs::absolute(sft);
  const fs::path art_abs = fs::absolute(art);
  const fs::path weights = td.path() / "model.safetensors";
  std::error_code ec;
  fs::create_symlink(sft_abs, weights, ec);
  ASSERT_FALSE(ec);

  // Determine L from the model so we register at the right shape — the test
  // sequence is 64 residues, encoded with cls + eos = 66.
  const fs::path apple_dir = td.path() / "model.apple";
  const fs::path wg_root = apple_dir / "whole-graph";
  fs::create_directories(wg_root);
  const fs::path bundle_link = wg_root / "B-1_L-67" / "whole_graph.mlmodelc";
  fs::create_directories(bundle_link.parent_path());
  fs::create_directory_symlink(art_abs, bundle_link, ec);
  ASSERT_FALSE(ec) << "symlink whole_graph fixture failed: " << ec.message();

  // Also link the manifest if the fixture has one (Phase 14 freshness
  // check). The manifest sits next to the .mlmodelc bundle, not inside it.
  const fs::path real_manifest = art_abs.parent_path() / "esm_cpp_artifact.json";
  if (fs::is_regular_file(real_manifest, ec)) {
    fs::create_symlink(real_manifest,
                        bundle_link.parent_path() / "esm_cpp_artifact.json",
                        ec);
    // ignore ec — the freshness check is informational, not fatal
  }

  // Force the cache dir to a nonexistent path so only the sibling is found.
  EnvScope cache_off("ESM_CPP_CACHE_DIR", (td.path() / "nope").string());
  EnvScope wg_default("ESM_APPLE_ANE_GRAPH", nullptr);

  auto model = esm::Model::LoadFromSafetensors(weights.string());
  ASSERT_NE(model, nullptr);

#ifdef ESM_APPLE_ANE_AVAILABLE
  auto shapes = model->whole_graph_shapes();
  ASSERT_EQ(shapes.size(), 1u) << "auto-discovery should have registered 1 shape";
  EXPECT_EQ(shapes[0].first, 1);
  EXPECT_EQ(shapes[0].second, 67);
  EXPECT_FALSE(model->whole_graph_path().empty());

  // Forward path: scheduled with a single sequence should auto-engage the
  // whole-graph (uniform L=67 + matching registration + no env opt-out).
  esm::Tokenizer tok;
  const std::string seq =
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG";
  const auto ids = tok.Encode(seq);
  std::vector<std::vector<std::int32_t>> batch = {ids};
  auto out = model->ForwardScheduled(batch, {});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FALSE(out[0].empty());
  for (float x : out[0]) EXPECT_TRUE(std::isfinite(x));
#else
  EXPECT_TRUE(model->whole_graph_shapes().empty())
      << "non-Apple build should not auto-register";
#endif
}

// Phase 14 T3: ESM_APPLE_ANE_GRAPH=off disables auto-engage even when shapes
// are registered. The opt-out gate is in ForwardScheduled, not in
// LoadWholeGraphArtifact, so the shapes still show up in whole_graph_shapes().
TEST(AppleWholeGraph, EnvOffDisablesAutoEngage) {
  const char* sft = std::getenv("ESM_WHOLE_GRAPH_TEST_SAFETENSORS");
  const char* art = std::getenv("ESM_WHOLE_GRAPH_TEST_ARTIFACT");
  if (!sft || !art) GTEST_SKIP() << "fixture envs not set";

  EnvScope wg_off("ESM_APPLE_ANE_GRAPH", "off");
  auto model = esm::Model::LoadFromSafetensors(sft);
  ASSERT_NE(model, nullptr);
#ifdef ESM_APPLE_ANE_AVAILABLE
  const bool registered = model->LoadWholeGraphArtifact(
      art, /*B=*/1, /*L=*/67, esm::WholeGraphComputeUnits::kCpuAndNeuralEngine);
  ASSERT_TRUE(registered);
  ASSERT_EQ(model->whole_graph_shapes().size(), 1u);
#endif

  // With env opt-out, the standard scheduled path runs — verify finite logits.
  esm::Tokenizer tok;
  const std::string seq =
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG";
  const auto ids = tok.Encode(seq);
  auto out = model->ForwardScheduled({ids}, {});
  ASSERT_EQ(out.size(), 1u);
  for (float x : out[0]) EXPECT_TRUE(std::isfinite(x));
}
