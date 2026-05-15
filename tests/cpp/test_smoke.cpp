#include <gtest/gtest.h>

#include "esm_cpp/status.h"
#include "esm_cpp/version.h"

TEST(Smoke, VersionStringMatchesComponents) {
  EXPECT_EQ(esm::kVersionMajor, 0);
  EXPECT_EQ(esm::kVersionMinor, 1);
  EXPECT_EQ(esm::kVersionPatch, 0);
  EXPECT_STREQ(esm::kVersionString, "0.1.0");
}

TEST(Smoke, DefaultStatusIsOk) {
  esm::Status s;
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(s.code(), esm::StatusCode::kOk);
  EXPECT_EQ(s.ToString(), "Ok");
}

TEST(Smoke, ErrorStatusCarriesMessage) {
  esm::Status s(esm::StatusCode::kShapeMismatch, "expected [3,4]");
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), esm::StatusCode::kShapeMismatch);
  EXPECT_EQ(s.ToString(), "ShapeMismatch: expected [3,4]");
}
