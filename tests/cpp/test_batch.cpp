#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "esm_cpp/batch.h"

namespace {

TEST(BatchView, ConstructsFromValidPackedInputs) {
  // Two sequences of lengths {3, 5} packed back-to-back.
  std::vector<std::int32_t> ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::int32_t> cu_seqlens = {0, 3, 8};
  esm::BatchView view(ids, {}, cu_seqlens, /*batch_size=*/2);
  EXPECT_EQ(view.batch_size, 2);
  EXPECT_EQ(view.total_tokens(), 8);
  EXPECT_EQ(view.sequence_length(0), 3);
  EXPECT_EQ(view.sequence_length(1), 5);
}

TEST(BatchView, ConstructsWithMatchingMasks) {
  std::vector<std::int32_t> ids = {0, 1, 2, 3, 4};
  std::vector<std::int32_t> masks = {1, 1, 1, 1, 0};
  std::vector<std::int32_t> cu_seqlens = {0, 5};
  esm::BatchView view(ids, masks, cu_seqlens, /*batch_size=*/1);
  EXPECT_EQ(view.total_tokens(), 5);
  EXPECT_EQ(view.batch_size, 1);
  EXPECT_FALSE(view.packed_masks.empty());
}

TEST(BatchView, RejectsCuSeqlensSizeMismatch) {
  std::vector<std::int32_t> ids = {0, 1, 2};
  std::vector<std::int32_t> cu_seqlens = {0, 3};  // size 2; batch_size=2 needs 3
  EXPECT_THROW(esm::BatchView(ids, {}, cu_seqlens, /*batch_size=*/2),
               std::invalid_argument);
}

TEST(BatchView, RejectsNonMonotonicCuSeqlens) {
  std::vector<std::int32_t> ids = {0, 1, 2, 3, 4};
  std::vector<std::int32_t> cu_seqlens = {0, 3, 2};  // decreasing
  EXPECT_THROW(esm::BatchView(ids, {}, cu_seqlens, /*batch_size=*/2),
               std::invalid_argument);
}

TEST(BatchView, RejectsCuSeqlensTotalMismatch) {
  std::vector<std::int32_t> ids = {0, 1, 2, 3, 4};
  std::vector<std::int32_t> cu_seqlens = {0, 3, 6};  // back != ids.size()
  EXPECT_THROW(esm::BatchView(ids, {}, cu_seqlens, /*batch_size=*/2),
               std::invalid_argument);
}

TEST(BatchView, RejectsMaskSizeMismatch) {
  std::vector<std::int32_t> ids = {0, 1, 2, 3, 4};
  std::vector<std::int32_t> masks = {1, 1, 1};  // wrong size
  std::vector<std::int32_t> cu_seqlens = {0, 5};
  EXPECT_THROW(esm::BatchView(ids, masks, cu_seqlens, /*batch_size=*/1),
               std::invalid_argument);
}

TEST(BatchView, RejectsNegativeCuSeqlens) {
  std::vector<std::int32_t> ids = {0, 1, 2};
  std::vector<std::int32_t> cu_seqlens = {-1, 3};
  EXPECT_THROW(esm::BatchView(ids, {}, cu_seqlens, /*batch_size=*/1),
               std::invalid_argument);
}

TEST(BatchView, RejectsZeroBatchSize) {
  std::vector<std::int32_t> ids = {};
  std::vector<std::int32_t> cu_seqlens = {0};
  EXPECT_THROW(esm::BatchView(ids, {}, cu_seqlens, /*batch_size=*/0),
               std::invalid_argument);
}

TEST(BatchView, EmptyMasksMeansAllReal) {
  std::vector<std::int32_t> ids = {0, 1, 2};
  std::vector<std::int32_t> cu_seqlens = {0, 3};
  esm::BatchView view(ids, {}, cu_seqlens, /*batch_size=*/1);
  EXPECT_TRUE(view.packed_masks.empty());
}

}  // namespace
