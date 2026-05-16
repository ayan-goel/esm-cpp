// Phase 3 Slice 4: GGUF reader/writer round-trip.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "esm_cpp/io.h"

namespace {

std::string TempPath(const std::string& tag) {
  auto dir = std::filesystem::temp_directory_path() / "esm_cpp_gguf_tests";
  std::filesystem::create_directories(dir);
  return (dir / (tag + ".gguf")).string();
}

}  // namespace

TEST(GgufFile, FP32SingleTensorRoundTrip) {
  std::vector<float> tensor = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::unordered_map<std::string, esm::io::GgufFile::MetadataValue> meta = {
      {"general.architecture", std::string("test")},
      {"general.alignment", static_cast<std::int64_t>(32)},
  };
  esm::io::GgufFile::WriteTensor wt;
  wt.dtype = esm::io::GgufFile::GgmlType::F32;
  wt.shape = {3, 2};  // [K=3, N=2] = 6 floats
  wt.data = tensor.data();
  wt.size_bytes = tensor.size() * sizeof(float);

  const std::string path = TempPath("fp32_single");
  esm::io::GgufFile::Write(path, meta, {{"my_tensor", wt}});

  ASSERT_TRUE(esm::io::GgufFile::LooksLikeGguf(path));
  auto gf = esm::io::GgufFile::Open(path);
  auto info = gf->Get("my_tensor");
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->dtype, esm::io::GgufFile::GgmlType::F32);
  EXPECT_EQ(info->shape, (std::vector<std::uint64_t>{3, 2}));
  EXPECT_EQ(info->size_bytes, tensor.size() * sizeof(float));
  std::vector<float> roundtrip(tensor.size());
  std::memcpy(roundtrip.data(), info->data, info->size_bytes);
  EXPECT_EQ(roundtrip, tensor);
}

TEST(GgufFile, MetadataRoundTripsAcrossTypes) {
  std::unordered_map<std::string, esm::io::GgufFile::MetadataValue> meta = {
      {"general.architecture", std::string("esm")},
      {"general.alignment", static_cast<std::int64_t>(32)},
      {"esm.block_count", static_cast<std::int64_t>(6)},
      {"esm.attention.layer_norm_epsilon", 1.0e-5},
      {"esm.token_dropout", true},
      {"esm.dims",
       std::vector<std::int64_t>{320, 6, 20, 1280}},
  };
  std::vector<float> dummy = {0.0f};
  esm::io::GgufFile::WriteTensor wt;
  wt.dtype = esm::io::GgufFile::GgmlType::F32;
  wt.shape = {1};
  wt.data = dummy.data();
  wt.size_bytes = sizeof(float);

  const std::string path = TempPath("metadata");
  esm::io::GgufFile::Write(path, meta, {{"dummy", wt}});
  auto gf = esm::io::GgufFile::Open(path);

  ASSERT_TRUE(gf->Metadata("general.architecture").has_value());
  EXPECT_EQ(std::get<std::string>(*gf->Metadata("general.architecture")),
            "esm");
  EXPECT_EQ(std::get<std::int64_t>(*gf->Metadata("esm.block_count")), 6);
  EXPECT_DOUBLE_EQ(
      std::get<double>(*gf->Metadata("esm.attention.layer_norm_epsilon")),
      1.0e-5);
  EXPECT_EQ(std::get<bool>(*gf->Metadata("esm.token_dropout")), true);
  // gf->Metadata returns a value-typed optional; binding a reference
  // through std::get on the temporary leaves a dangling reference once
  // the full-expression ends. Copy into a local instead.
  auto dims_opt = gf->Metadata("esm.dims");
  ASSERT_TRUE(dims_opt.has_value());
  const auto dims = std::get<std::vector<std::int64_t>>(*dims_opt);
  EXPECT_EQ(dims, (std::vector<std::int64_t>{320, 6, 20, 1280}));
}

TEST(GgufFile, MultipleTensorsRoundTrip) {
  std::vector<float> a(64);
  std::vector<float> b(128);
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i);
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = -static_cast<float>(i);

  esm::io::GgufFile::WriteTensor ta, tb;
  ta.dtype = esm::io::GgufFile::GgmlType::F32;
  ta.shape = {8, 8};
  ta.data = a.data();
  ta.size_bytes = a.size() * sizeof(float);
  tb.dtype = esm::io::GgufFile::GgmlType::F32;
  tb.shape = {16, 8};
  tb.data = b.data();
  tb.size_bytes = b.size() * sizeof(float);

  const std::string path = TempPath("multi");
  esm::io::GgufFile::Write(
      path, {{"general.architecture", std::string("test")}},
      {{"alpha", ta}, {"beta", tb}});
  auto gf = esm::io::GgufFile::Open(path);
  ASSERT_TRUE(gf->Has("alpha"));
  ASSERT_TRUE(gf->Has("beta"));
  auto ia = gf->Get("alpha");
  auto ib = gf->Get("beta");
  std::vector<float> ra(a.size()), rb(b.size());
  std::memcpy(ra.data(), ia->data, ia->size_bytes);
  std::memcpy(rb.data(), ib->data, ib->size_bytes);
  EXPECT_EQ(ra, a);
  EXPECT_EQ(rb, b);
}

TEST(GgufFile, RejectsBadMagic) {
  const std::string path = TempPath("bad_magic");
  {
    std::ofstream f(path, std::ios::binary);
    const char garbage[] = "NOTGGUFCONTENTSHERE";
    f.write(garbage, sizeof(garbage));
  }
  EXPECT_FALSE(esm::io::GgufFile::LooksLikeGguf(path));
  EXPECT_THROW(esm::io::GgufFile::Open(path), std::runtime_error);
}

TEST(GgufFile, RejectsWrongVersion) {
  const std::string path = TempPath("bad_version");
  {
    std::ofstream f(path, std::ios::binary);
    const std::uint32_t magic = 0x46554747u;
    const std::uint32_t version = 2;  // v2 not supported
    const std::uint64_t tensor_count = 0;
    const std::uint64_t meta_count = 0;
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&tensor_count), 8);
    f.write(reinterpret_cast<const char*>(&meta_count), 8);
  }
  EXPECT_TRUE(esm::io::GgufFile::LooksLikeGguf(path));
  EXPECT_THROW(esm::io::GgufFile::Open(path), std::runtime_error);
}

TEST(GgufFile, EmptyTensorsAndMetadata) {
  const std::string path = TempPath("empty");
  esm::io::GgufFile::Write(path, {}, {});
  auto gf = esm::io::GgufFile::Open(path);
  EXPECT_TRUE(gf->Names().empty());
}
