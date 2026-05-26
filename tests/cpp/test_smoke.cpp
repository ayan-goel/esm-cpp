#include <gtest/gtest.h>

#include <cstdio>

#include "esm_cpp/status.h"
#include "esm_cpp/version.h"

TEST(Smoke, VersionStringMatchesComponents) {
  // The literal version values live in include/esm_cpp/version.h. This test
  // guards against drift between the int triple and the formatted string —
  // bumping one without the other is the common slip-up.
  char expected[32];
  std::snprintf(expected, sizeof(expected), "%d.%d.%d",
                esm::kVersionMajor, esm::kVersionMinor, esm::kVersionPatch);
  EXPECT_STREQ(esm::kVersionString, expected);
  // Sanity: not all zero (catches the case where someone accidentally
  // zeroed out the constants).
  EXPECT_GE(esm::kVersionMajor + esm::kVersionMinor + esm::kVersionPatch, 1);
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
