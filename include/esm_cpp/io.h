#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
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

}  // namespace esm::io
