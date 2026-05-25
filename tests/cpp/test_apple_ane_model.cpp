// Phase 12 T6: model-level parity guard for the ANE fp16 / CoreML path.
//
// Like the Phase-11 AppleAmx test, ANE is a black box: validation lives at the
// model level. Load a small ESM-2 checkpoint, run the FP32 forward as the
// reference, then load the per-Linear-per-bucket ANE artifacts (built by
// `tools/build_amx_artifacts.py --compute-units CPU_AND_NE --buckets …`), flip
// ESM_APPLE_ANE=on, and confirm the logits stay finite, correlate ≥ 0.999 with
// the FP32 reference, and agree on the argmax at ≥ 99 %. Fixture is
// env-var-driven (ESM_ANE_TEST_SAFETENSORS + ESM_ANE_TEST_ARTIFACT_DIR) so the
// test SKIPs cleanly without fixtures or on non-Apple CI.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "esm_cpp/model.h"
#include "esm_cpp/tokenizer.h"

namespace {

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

}  // namespace

TEST(AppleAne, Fp16ModelParityVsFp32Reference) {
  const char* sft = std::getenv("ESM_ANE_TEST_SAFETENSORS");
  const char* art = std::getenv("ESM_ANE_TEST_ARTIFACT_DIR");
  if (!sft || !art) {
    GTEST_SKIP() << "Set ESM_ANE_TEST_SAFETENSORS + ESM_ANE_TEST_ARTIFACT_DIR "
                 << "to the model.safetensors and the *.ane-fp16/ dir built by "
                 << "tools/build_amx_artifacts.py --compute-units CPU_AND_NE.";
  }
  if (!std::filesystem::exists(sft) || !std::filesystem::is_directory(art)) {
    GTEST_SKIP() << "fixture paths missing: " << sft << " / " << art;
  }

  auto model = esm::Model::LoadFromSafetensors(sft);
  ASSERT_NE(model, nullptr);
  esm::Tokenizer tok;

  const std::string seq =
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG";
  const auto ids = tok.Encode(seq);
  std::vector<std::int32_t> mask(ids.size(), 1);

  const char* prev = std::getenv("ESM_APPLE_ANE");
  std::string saved = prev ? prev : "";
  ::unsetenv("ESM_APPLE_ANE");
  const auto ref = model->Forward(std::span<const std::int32_t>(ids),
                                   std::span<const std::int32_t>(mask));
  ASSERT_TRUE(AllFinite(ref));

  const std::size_t loaded = model->LoadAneArtifacts(art);
#ifdef ESM_APPLE_ANE_AVAILABLE
  ASSERT_GT(loaded, 0u) << "no ANE artifacts loaded — fixture mismatched?";
#else
  if (loaded == 0) GTEST_SKIP() << "Apple-ANE not available on this build";
#endif
  ::setenv("ESM_APPLE_ANE", "on", 1);
  const auto ane = model->Forward(std::span<const std::int32_t>(ids),
                                   std::span<const std::int32_t>(mask));

  if (prev) ::setenv("ESM_APPLE_ANE", saved.c_str(), 1);
  else      ::unsetenv("ESM_APPLE_ANE");

  ASSERT_EQ(ane.size(), ref.size());
  EXPECT_TRUE(AllFinite(ane));
  const double corr = Correlate(std::span<const float>(ref),
                                 std::span<const float>(ane));
  EXPECT_GE(corr, 0.999) << "ANE logit correlation below 0.999";

  const int V = model->config().vocab_size;
  const int L = static_cast<int>(ane.size() / static_cast<std::size_t>(V));
  const double agree = ArgmaxAgreement(ref, ane, L, V);
  EXPECT_GE(agree, 0.99) << "ANE argmax agreement below 0.99";
}
