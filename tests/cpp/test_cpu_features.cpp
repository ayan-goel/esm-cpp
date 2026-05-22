#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "esm_cpp/cpu_features.h"

namespace {

class IsaEnvGuard {
 public:
  explicit IsaEnvGuard(const char* var) : var_(var) {
    const char* prev = std::getenv(var_);
    had_prev_ = prev != nullptr;
    if (had_prev_) prev_value_ = prev;
  }
  ~IsaEnvGuard() {
    if (had_prev_) {
      ::setenv(var_, prev_value_.c_str(), 1);
    } else {
      ::unsetenv(var_);
    }
  }
  void Set(const char* value) const {
    if (value) {
      ::setenv(var_, value, 1);
    } else {
      ::unsetenv(var_);
    }
  }

 private:
  const char* var_;
  bool had_prev_;
  std::string prev_value_;
};

}  // namespace

TEST(CpuFeatures, IsaToStringRoundTrip) {
  for (auto isa : {esm::Isa::Ref, esm::Isa::Neon, esm::Isa::NeonDotProd,
                   esm::Isa::NeonI8mm, esm::Isa::Avx2, esm::Isa::Avx512,
                   esm::Isa::Avx512Vnni, esm::Isa::Amx}) {
    auto s = esm::IsaToString(isa);
    auto parsed = esm::StringToIsa(s);
    ASSERT_TRUE(parsed.has_value()) << "round trip failed for " << s;
    EXPECT_EQ(*parsed, isa);
  }
}

TEST(CpuFeatures, IsaToStringStableNames) {
  EXPECT_EQ(esm::IsaToString(esm::Isa::Ref), "ref");
  EXPECT_EQ(esm::IsaToString(esm::Isa::Neon), "neon");
  EXPECT_EQ(esm::IsaToString(esm::Isa::NeonDotProd), "neondotprod");
  EXPECT_EQ(esm::IsaToString(esm::Isa::NeonI8mm), "neoni8mm");
  EXPECT_EQ(esm::IsaToString(esm::Isa::Avx2), "avx2");
  EXPECT_EQ(esm::IsaToString(esm::Isa::Avx512), "avx512");
  EXPECT_EQ(esm::IsaToString(esm::Isa::Avx512Vnni), "avx512vnni");
  EXPECT_EQ(esm::IsaToString(esm::Isa::Amx), "amx");
}

TEST(CpuFeatures, ForceIsaNeonTiersAcceptedRegardlessOfHost) {
  IsaEnvGuard guard("ESM_FORCE_ISA");
  guard.Set("neondotprod");
  EXPECT_EQ(esm::CurrentIsa(), esm::Isa::NeonDotProd);
  guard.Set("neoni8mm");
  EXPECT_EQ(esm::CurrentIsa(), esm::Isa::NeonI8mm);
}

TEST(CpuFeatures, StringToIsaRejectsUnknown) {
  EXPECT_FALSE(esm::StringToIsa("").has_value());
  EXPECT_FALSE(esm::StringToIsa("sse").has_value());
  EXPECT_FALSE(esm::StringToIsa("REF").has_value());
  EXPECT_FALSE(esm::StringToIsa("avx 512").has_value());
  EXPECT_FALSE(esm::StringToIsa("avx512f").has_value());
}

TEST(CpuFeatures, HostIsaIsConsistentForBuildTarget) {
  auto host = esm::HostIsa();
#if defined(__aarch64__) || defined(_M_ARM64)
  // Detection picks the best ARM tier the host supports: Neon (FMLA only),
  // NeonDotProd (FEAT_DotProd), or NeonI8mm (FEAT_I8MM). All are valid.
  EXPECT_TRUE(host == esm::Isa::Neon || host == esm::Isa::NeonDotProd ||
              host == esm::Isa::NeonI8mm)
      << "unexpected ARM host ISA: " << esm::IsaToString(host);
#elif defined(__x86_64__) || defined(_M_X64)
  EXPECT_NE(host, esm::Isa::Neon);
  EXPECT_GE(static_cast<int>(host), static_cast<int>(esm::Isa::Ref));
#else
  EXPECT_EQ(host, esm::Isa::Ref);
#endif
}

TEST(CpuFeatures, HostIsaIsIdempotent) {
  EXPECT_EQ(esm::HostIsa(), esm::HostIsa());
}

TEST(CpuFeatures, ForceIsaRefOverridesHost) {
  IsaEnvGuard guard("ESM_FORCE_ISA");
  guard.Set("ref");
  EXPECT_EQ(esm::CurrentIsa(), esm::Isa::Ref);
}

TEST(CpuFeatures, ForceIsaNeonAcceptedRegardlessOfHost) {
  IsaEnvGuard guard("ESM_FORCE_ISA");
  guard.Set("neon");
  EXPECT_EQ(esm::CurrentIsa(), esm::Isa::Neon);
}

TEST(CpuFeatures, ForceIsaUnknownFallsBackToHost) {
  IsaEnvGuard guard("ESM_FORCE_ISA");
  guard.Set("nonsense");
  EXPECT_EQ(esm::CurrentIsa(), esm::HostIsa());
}

TEST(CpuFeatures, ForceIsaEmptyFallsBackToHost) {
  IsaEnvGuard guard("ESM_FORCE_ISA");
  guard.Set("");
  EXPECT_EQ(esm::CurrentIsa(), esm::HostIsa());
}

TEST(CpuFeatures, ForceIsaUnsetReturnsHost) {
  IsaEnvGuard guard("ESM_FORCE_ISA");
  guard.Set(nullptr);
  EXPECT_EQ(esm::CurrentIsa(), esm::HostIsa());
}

TEST(CpuFeatures, MaybeLogIsaOnceIsSafeToCall) {
  IsaEnvGuard log_guard("ESM_LOG_ISA");
  log_guard.Set("0");
  esm::MaybeLogIsaOnce();
  esm::MaybeLogIsaOnce();
  SUCCEED();
}
