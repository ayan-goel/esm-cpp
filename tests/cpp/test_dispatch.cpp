#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include "esm_cpp/cpu_features.h"
#include "esm_cpp/kernels.h"

namespace {

class ForceIsa {
 public:
  explicit ForceIsa(const char* value) {
    prev_ = std::getenv("ESM_FORCE_ISA");
    if (prev_) prev_copy_ = prev_;
    if (value) {
      ::setenv("ESM_FORCE_ISA", value, 1);
    } else {
      ::unsetenv("ESM_FORCE_ISA");
    }
  }
  ~ForceIsa() {
    if (prev_) {
      ::setenv("ESM_FORCE_ISA", prev_copy_.c_str(), 1);
    } else {
      ::unsetenv("ESM_FORCE_ISA");
    }
  }

 private:
  const char* prev_;
  std::string prev_copy_;
};

std::vector<float> RandomVec(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

void ExpectAllClose(const std::vector<float>& a, const std::vector<float>& b,
                    float tol) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    ASSERT_NEAR(a[i], b[i], tol) << "at index " << i;
  }
}

}  // namespace

TEST(Dispatch, LinearFacadeMatchesRef) {
  ForceIsa guard("ref");
  const int M = 3, N = 5, K = 4;
  auto A = RandomVec(M * K, 1);
  auto W = RandomVec(N * K, 2);
  auto bias = RandomVec(N, 3);
  std::vector<float> c_ref(M * N), c_facade(M * N);
  esm::kernels::LinearRef(A.data(), W.data(), bias.data(), c_ref.data(), M, N, K);
  esm::kernels::Linear(A.data(), W.data(), bias.data(), c_facade.data(), M, N, K);
  ExpectAllClose(c_ref, c_facade, 0.0f);
}

TEST(Dispatch, LayerNormFacadeMatchesRef) {
  ForceIsa guard("ref");
  const int rows = 4, d = 8;
  auto x = RandomVec(rows * d, 11);
  auto gamma = RandomVec(d, 12);
  auto beta = RandomVec(d, 13);
  std::vector<float> y_ref(rows * d), y_facade(rows * d);
  esm::kernels::LayerNormRef(x.data(), gamma.data(), beta.data(), 1e-5f,
                              y_ref.data(), rows, d);
  esm::kernels::LayerNorm(x.data(), gamma.data(), beta.data(), 1e-5f,
                          y_facade.data(), rows, d);
  ExpectAllClose(y_ref, y_facade, 0.0f);
}

TEST(Dispatch, GeluFacadeMatchesRef) {
  ForceIsa guard("ref");
  const std::size_t n = 33;
  auto x = RandomVec(n, 21);
  std::vector<float> y_ref(n), y_facade(n);
  esm::kernels::GeluRef(x.data(), y_ref.data(), n);
  esm::kernels::Gelu(x.data(), y_facade.data(), n);
  ExpectAllClose(y_ref, y_facade, 0.0f);
}

TEST(Dispatch, RopeApplyFacadeMatchesRef) {
  ForceIsa guard("ref");
  const int H = 2, L = 5, dh = 8;
  auto x_ref = RandomVec(H * L * dh, 31);
  auto x_facade = x_ref;
  std::vector<float> cos(L * dh), sin(L * dh);
  esm::kernels::RopeBuildTables(L, dh, cos.data(), sin.data());
  esm::kernels::RopeApplyInplaceRef(x_ref.data(), cos.data(), sin.data(), H, L,
                                     dh);
  esm::kernels::RopeApplyInplace(x_facade.data(), cos.data(), sin.data(), H, L,
                                  dh);
  ExpectAllClose(x_ref, x_facade, 0.0f);
}

TEST(Dispatch, AttentionFacadeMatchesRef) {
  ForceIsa guard("ref");
  const int H = 2, L = 4, dh = 3;
  auto Q = RandomVec(H * L * dh, 41);
  auto K = RandomVec(H * L * dh, 42);
  auto V = RandomVec(H * L * dh, 43);
  std::vector<float> out_ref(L * H * dh), out_facade(L * H * dh);
  esm::kernels::AttentionRef(Q.data(), K.data(), V.data(), nullptr,
                              out_ref.data(), H, L, dh);
  esm::kernels::Attention(Q.data(), K.data(), V.data(), nullptr,
                          out_facade.data(), H, L, dh);
  ExpectAllClose(out_ref, out_facade, 0.0f);
}

TEST(Dispatch, FacadeRespectsForceIsaRef) {
  ForceIsa guard("ref");
  EXPECT_EQ(esm::CurrentIsa(), esm::Isa::Ref);
  const int M = 2, N = 2, K = 2;
  std::vector<float> A = {1, 2, 3, 4};
  std::vector<float> W = {1, 0, 0, 1};
  std::vector<float> C(4);
  esm::kernels::Linear(A.data(), W.data(), nullptr, C.data(), M, N, K);
  EXPECT_FLOAT_EQ(C[0], 1.0f);
  EXPECT_FLOAT_EQ(C[1], 2.0f);
  EXPECT_FLOAT_EQ(C[2], 3.0f);
  EXPECT_FLOAT_EQ(C[3], 4.0f);
}
