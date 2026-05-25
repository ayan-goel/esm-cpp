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
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "esm_cpp/model.h"
#include "esm_cpp/tokenizer.h"

namespace {

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
