#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace esm::io {

// Minimal safetensors reader.
// Format:
//   bytes [0, 8): little-endian uint64 header length N
//   bytes [8, 8+N): JSON header
//   bytes [8+N, EOF): raw tensor data (offsets in JSON are relative to this)
class SafetensorsFile {
 public:
  struct TensorInfo {
    std::string dtype;            // "F32", "F16", "I32", "I64", ...
    std::vector<int64_t> shape;
    const std::byte* data;
    std::size_t size_bytes;
  };

  // Opens the file at `path` and parses the header. Throws std::runtime_error
  // on I/O failure or malformed header. (Load-time API; not in the hot path.)
  static std::unique_ptr<SafetensorsFile> Open(const std::string& path);

  std::vector<std::string> Names() const;
  std::optional<TensorInfo> Get(std::string_view name) const;
  bool Has(std::string_view name) const { return Get(name).has_value(); }

 private:
  SafetensorsFile() = default;
  std::vector<std::byte> bytes_;
  std::unordered_map<std::string, TensorInfo> tensors_;
};

// GGUF v3 reader/writer. We support the strict subset llama.cpp uses
// for FP32 plus a custom Q8_ESM type for our per-channel symmetric
// INT8 weights (see esm_cpp/quant.h). The spec is pinned at:
//   https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
// rev as of v3 (commit recorded in src/io/gguf.cpp).
class GgufFile {
 public:
  // GGML tensor types we read/write. The standard GGML types come from
  // llama.cpp's ggml.h enum; Q8_ESM is our extension reserved at value
  // 100 (well above the current llama.cpp range as of 2026-05).
  enum class GgmlType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q8_ESM = 100,
  };

  struct TensorInfo {
    GgmlType dtype;
    std::vector<std::uint64_t> shape;
    const std::byte* data;
    std::size_t size_bytes;
  };

  // Metadata values are tagged unions. Strings live as std::string;
  // primitive numerics decay to int64 / double (we don't preserve the
  // exact wire type, but ESM-2 metadata is purely informational so
  // the loss is fine).
  using MetadataValue = std::variant<std::int64_t, double, std::string,
                                      bool, std::vector<std::int64_t>,
                                      std::vector<double>,
                                      std::vector<std::string>>;

  // Opens the file at `path`, validates magic + version, parses the
  // metadata + tensor info section, and mmaps the data region for
  // zero-copy tensor access. Throws std::runtime_error on any failure.
  static std::unique_ptr<GgufFile> Open(const std::string& path);

  std::vector<std::string> Names() const;
  std::optional<TensorInfo> Get(std::string_view name) const;
  bool Has(std::string_view name) const { return Get(name).has_value(); }
  std::optional<MetadataValue> Metadata(std::string_view key) const;

  // Static writer. Streams the GGUF file to `path` in two passes:
  // header + tensor info first, then aligned data. tensors maps from
  // tensor name to (dtype, shape, raw_bytes). metadata maps from key
  // to MetadataValue.
  struct WriteTensor {
    GgmlType dtype;
    std::vector<std::uint64_t> shape;
    const void* data;
    std::size_t size_bytes;
  };
  static void Write(const std::string& path,
                    const std::unordered_map<std::string, MetadataValue>&
                        metadata,
                    const std::vector<std::pair<std::string, WriteTensor>>&
                        tensors);

  // Sniff the magic bytes of `path`. Returns true if it starts with the
  // GGUF magic. Useful for Model::LoadFromAuto-style dispatch where the
  // caller doesn't know which format the file is in.
  static bool LooksLikeGguf(const std::string& path);

 private:
  GgufFile() = default;
  // Backing storage; data spans into bytes_ are stable for the lifetime
  // of the GgufFile (we own the buffer).
  std::vector<std::byte> bytes_;
  std::unordered_map<std::string, TensorInfo> tensors_;
  std::unordered_map<std::string, MetadataValue> metadata_;
};

}  // namespace esm::io
