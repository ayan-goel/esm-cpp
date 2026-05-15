#include "esm_cpp/io.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace esm::io {

std::unique_ptr<SafetensorsFile> SafetensorsFile::Open(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("safetensors: cannot open " + path);
  std::streamsize file_size = f.tellg();
  if (file_size < 8) {
    throw std::runtime_error("safetensors: file too short: " + path);
  }
  f.seekg(0, std::ios::beg);

  auto file = std::unique_ptr<SafetensorsFile>(new SafetensorsFile());
  file->bytes_.resize(static_cast<std::size_t>(file_size));
  f.read(reinterpret_cast<char*>(file->bytes_.data()), file_size);
  if (!f) throw std::runtime_error("safetensors: short read: " + path);

  std::uint64_t header_len = 0;
  for (int i = 0; i < 8; ++i) {
    header_len |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(
                       file->bytes_[static_cast<std::size_t>(i)]))
                  << (8 * i);
  }
  if (8 + header_len > static_cast<std::uint64_t>(file_size)) {
    throw std::runtime_error("safetensors: header length out of range: " + path);
  }

  const char* header_begin = reinterpret_cast<const char*>(file->bytes_.data() + 8);
  std::string_view header_json(header_begin, static_cast<std::size_t>(header_len));
  nlohmann::json header = nlohmann::json::parse(header_json);

  const std::byte* data_base = file->bytes_.data() + 8 + header_len;
  std::size_t data_size = static_cast<std::size_t>(file_size) - 8 -
                          static_cast<std::size_t>(header_len);

  for (auto it = header.begin(); it != header.end(); ++it) {
    const std::string& name = it.key();
    if (name == "__metadata__") continue;
    const auto& entry = it.value();
    TensorInfo info;
    info.dtype = entry.at("dtype").get<std::string>();
    info.shape = entry.at("shape").get<std::vector<int64_t>>();
    auto offsets = entry.at("data_offsets").get<std::vector<std::uint64_t>>();
    if (offsets.size() != 2 || offsets[0] > offsets[1] ||
        offsets[1] > data_size) {
      throw std::runtime_error("safetensors: bad data_offsets for " + name);
    }
    info.data = data_base + offsets[0];
    info.size_bytes = static_cast<std::size_t>(offsets[1] - offsets[0]);
    file->tensors_.emplace(name, info);
  }

  return file;
}

std::vector<std::string> SafetensorsFile::Names() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto& [k, _] : tensors_) names.push_back(k);
  return names;
}

std::optional<SafetensorsFile::TensorInfo> SafetensorsFile::Get(
    std::string_view name) const {
  auto it = tensors_.find(std::string(name));
  if (it == tensors_.end()) return std::nullopt;
  return it->second;
}

}  // namespace esm::io
