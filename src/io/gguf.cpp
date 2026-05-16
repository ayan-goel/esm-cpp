// GGUF v3 reader/writer. Spec pinned at:
//   https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
// Reference revision: the v3 layout that llama.cpp shipped through 2025.
//
// File layout (little-endian throughout):
//   [0..4)   magic "GGUF"
//   [4..8)   uint32 version (= 3)
//   [8..16)  uint64 tensor_count
//   [16..24) uint64 metadata_kv_count
//   then `metadata_kv_count` entries: key:string + type:uint32 + value
//   then `tensor_count` entries: name:string + n_dims:uint32 +
//                                shape[n_dims]:uint64 + type:uint32 +
//                                offset:uint64
//   pad to `general.alignment` (default 32)
//   tensor data
//
// String wire format: uint64 length + raw bytes (no NUL terminator).

#include "esm_cpp/io.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace esm::io {

namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747u;  // "GGUF" little-endian
constexpr std::uint32_t kGgufVersion = 3;
constexpr std::uint64_t kDefaultAlignment = 32;

// gguf_metadata_value_type (mirrors llama.cpp's enum).
enum class MetaType : std::uint32_t {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12,
};

template <typename T>
T ReadLE(const std::byte*& p, const std::byte* end) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (p + sizeof(T) > end) {
    throw std::runtime_error("gguf: truncated read");
  }
  T v;
  std::memcpy(&v, p, sizeof(T));
  p += sizeof(T);
  return v;
}

std::string ReadString(const std::byte*& p, const std::byte* end) {
  const auto len = ReadLE<std::uint64_t>(p, end);
  if (p + len > end) {
    throw std::runtime_error("gguf: string overruns file");
  }
  std::string s(reinterpret_cast<const char*>(p), len);
  p += len;
  return s;
}

GgufFile::MetadataValue ReadMetaValue(const std::byte*& p,
                                       const std::byte* end);

template <typename T>
std::vector<std::int64_t> ReadIntArray(const std::byte*& p,
                                        const std::byte* end,
                                        std::uint64_t count) {
  std::vector<std::int64_t> out;
  out.reserve(count);
  for (std::uint64_t i = 0; i < count; ++i) {
    out.push_back(static_cast<std::int64_t>(ReadLE<T>(p, end)));
  }
  return out;
}

GgufFile::MetadataValue ReadArrayValue(const std::byte*& p,
                                        const std::byte* end) {
  const auto elem_type = ReadLE<std::uint32_t>(p, end);
  const auto count = ReadLE<std::uint64_t>(p, end);
  switch (static_cast<MetaType>(elem_type)) {
    case MetaType::Uint8:
      return ReadIntArray<std::uint8_t>(p, end, count);
    case MetaType::Int8:
      return ReadIntArray<std::int8_t>(p, end, count);
    case MetaType::Uint16:
      return ReadIntArray<std::uint16_t>(p, end, count);
    case MetaType::Int16:
      return ReadIntArray<std::int16_t>(p, end, count);
    case MetaType::Uint32:
      return ReadIntArray<std::uint32_t>(p, end, count);
    case MetaType::Int32:
      return ReadIntArray<std::int32_t>(p, end, count);
    case MetaType::Uint64:
      return ReadIntArray<std::uint64_t>(p, end, count);
    case MetaType::Int64:
      return ReadIntArray<std::int64_t>(p, end, count);
    case MetaType::Float32: {
      std::vector<double> out;
      out.reserve(count);
      for (std::uint64_t i = 0; i < count; ++i) {
        out.push_back(static_cast<double>(ReadLE<float>(p, end)));
      }
      return out;
    }
    case MetaType::Float64: {
      std::vector<double> out;
      out.reserve(count);
      for (std::uint64_t i = 0; i < count; ++i) out.push_back(ReadLE<double>(p, end));
      return out;
    }
    case MetaType::String: {
      std::vector<std::string> out;
      out.reserve(count);
      for (std::uint64_t i = 0; i < count; ++i) out.push_back(ReadString(p, end));
      return out;
    }
    default:
      throw std::runtime_error("gguf: unsupported array element type " +
                                std::to_string(elem_type));
  }
}

GgufFile::MetadataValue ReadMetaValue(const std::byte*& p,
                                       const std::byte* end) {
  const auto type = static_cast<MetaType>(ReadLE<std::uint32_t>(p, end));
  switch (type) {
    case MetaType::Uint8:
      return static_cast<std::int64_t>(ReadLE<std::uint8_t>(p, end));
    case MetaType::Int8:
      return static_cast<std::int64_t>(ReadLE<std::int8_t>(p, end));
    case MetaType::Uint16:
      return static_cast<std::int64_t>(ReadLE<std::uint16_t>(p, end));
    case MetaType::Int16:
      return static_cast<std::int64_t>(ReadLE<std::int16_t>(p, end));
    case MetaType::Uint32:
      return static_cast<std::int64_t>(ReadLE<std::uint32_t>(p, end));
    case MetaType::Int32:
      return static_cast<std::int64_t>(ReadLE<std::int32_t>(p, end));
    case MetaType::Uint64:
      return static_cast<std::int64_t>(ReadLE<std::uint64_t>(p, end));
    case MetaType::Int64:
      return ReadLE<std::int64_t>(p, end);
    case MetaType::Float32:
      return static_cast<double>(ReadLE<float>(p, end));
    case MetaType::Float64:
      return ReadLE<double>(p, end);
    case MetaType::Bool:
      return ReadLE<std::uint8_t>(p, end) != 0;
    case MetaType::String:
      return ReadString(p, end);
    case MetaType::Array:
      return ReadArrayValue(p, end);
  }
  throw std::runtime_error("gguf: unknown metadata type");
}

std::size_t TypeSizeBytes(GgufFile::GgmlType type,
                          const std::vector<std::uint64_t>& shape) {
  std::uint64_t n = 1;
  for (auto d : shape) n *= d;
  switch (type) {
    case GgufFile::GgmlType::F32:
      return n * 4;
    case GgufFile::GgmlType::F16:
      return n * 2;
    case GgufFile::GgmlType::Q8_ESM: {
      // [N, K] int8 packed + [N] float32 scales. Shape stores [K, N]
      // (column-major in llama.cpp convention) — flip to recover N.
      if (shape.size() != 2) {
        throw std::runtime_error(
            "gguf: Q8_ESM tensor must be 2-D");
      }
      const std::uint64_t K = shape[0];
      const std::uint64_t N = shape[1];
      return N * K + N * sizeof(float);
    }
  }
  throw std::runtime_error("gguf: unknown tensor type");
}

void WriteLE(std::ofstream& out, const void* data, std::size_t n) {
  out.write(static_cast<const char*>(data),
             static_cast<std::streamsize>(n));
  if (!out) throw std::runtime_error("gguf: write failed");
}

template <typename T>
void WriteLE(std::ofstream& out, T v) {
  WriteLE(out, &v, sizeof(T));
}

void WriteString(std::ofstream& out, std::string_view s) {
  WriteLE<std::uint64_t>(out, s.size());
  if (!s.empty()) WriteLE(out, s.data(), s.size());
}

void WriteMetaValue(std::ofstream& out,
                     const GgufFile::MetadataValue& v) {
  std::visit(
      [&out](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::int64_t>) {
          WriteLE<std::uint32_t>(
              out, static_cast<std::uint32_t>(MetaType::Int64));
          WriteLE<std::int64_t>(out, x);
        } else if constexpr (std::is_same_v<T, double>) {
          WriteLE<std::uint32_t>(
              out, static_cast<std::uint32_t>(MetaType::Float64));
          WriteLE<double>(out, x);
        } else if constexpr (std::is_same_v<T, std::string>) {
          WriteLE<std::uint32_t>(
              out, static_cast<std::uint32_t>(MetaType::String));
          WriteString(out, x);
        } else if constexpr (std::is_same_v<T, bool>) {
          WriteLE<std::uint32_t>(
              out, static_cast<std::uint32_t>(MetaType::Bool));
          WriteLE<std::uint8_t>(out, x ? 1 : 0);
        } else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) {
          WriteLE<std::uint32_t>(
              out, static_cast<std::uint32_t>(MetaType::Array));
          WriteLE<std::uint32_t>(
              out, static_cast<std::uint32_t>(MetaType::Int64));
          WriteLE<std::uint64_t>(out, x.size());
          for (auto v_ : x) WriteLE<std::int64_t>(out, v_);
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          WriteLE<std::uint32_t>(
              out, static_cast<std::uint32_t>(MetaType::Array));
          WriteLE<std::uint32_t>(
              out, static_cast<std::uint32_t>(MetaType::Float64));
          WriteLE<std::uint64_t>(out, x.size());
          for (auto v_ : x) WriteLE<double>(out, v_);
        } else if constexpr (std::is_same_v<T,
                                              std::vector<std::string>>) {
          WriteLE<std::uint32_t>(
              out, static_cast<std::uint32_t>(MetaType::Array));
          WriteLE<std::uint32_t>(
              out, static_cast<std::uint32_t>(MetaType::String));
          WriteLE<std::uint64_t>(out, x.size());
          for (const auto& v_ : x) WriteString(out, v_);
        }
      },
      v);
}

}  // namespace

std::unique_ptr<GgufFile> GgufFile::Open(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("gguf: open failed: " + path);
  }
  const auto size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<std::byte> bytes(size);
  f.read(reinterpret_cast<char*>(bytes.data()),
         static_cast<std::streamsize>(size));
  if (!f) {
    throw std::runtime_error("gguf: read failed: " + path);
  }
  if (bytes.size() < 24) {
    throw std::runtime_error("gguf: file too small to be GGUF: " + path);
  }
  const std::byte* p = bytes.data();
  const std::byte* end = bytes.data() + bytes.size();
  const auto magic = ReadLE<std::uint32_t>(p, end);
  if (magic != kGgufMagic) {
    throw std::runtime_error("gguf: bad magic in " + path);
  }
  const auto version = ReadLE<std::uint32_t>(p, end);
  if (version != kGgufVersion) {
    throw std::runtime_error("gguf: version " + std::to_string(version) +
                              " not supported (need v3)");
  }
  const auto tensor_count = ReadLE<std::uint64_t>(p, end);
  const auto meta_count = ReadLE<std::uint64_t>(p, end);

  auto out = std::unique_ptr<GgufFile>(new GgufFile());
  // Move bytes into the owning struct so all the `data` pointers remain
  // valid for the GgufFile's lifetime. We re-rebase p/end against the
  // moved buffer.
  out->bytes_ = std::move(bytes);
  const std::byte* base = out->bytes_.data();
  const std::byte* base_end = base + out->bytes_.size();
  // Recover the cursor offset after the header parse and continue.
  p = base + 24;
  end = base_end;

  for (std::uint64_t i = 0; i < meta_count; ++i) {
    auto key = ReadString(p, end);
    auto value = ReadMetaValue(p, end);
    out->metadata_.emplace(std::move(key), std::move(value));
  }

  struct PendingTensor {
    std::string name;
    GgmlType dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t data_offset;
  };
  std::vector<PendingTensor> pending;
  pending.reserve(tensor_count);
  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    PendingTensor t;
    t.name = ReadString(p, end);
    const auto n_dims = ReadLE<std::uint32_t>(p, end);
    t.shape.resize(n_dims);
    for (std::uint32_t d = 0; d < n_dims; ++d) {
      t.shape[d] = ReadLE<std::uint64_t>(p, end);
    }
    t.dtype = static_cast<GgmlType>(ReadLE<std::uint32_t>(p, end));
    t.data_offset = ReadLE<std::uint64_t>(p, end);
    pending.push_back(std::move(t));
  }

  // Align to general.alignment (default 32).
  std::uint64_t alignment = kDefaultAlignment;
  if (auto it = out->metadata_.find("general.alignment");
      it != out->metadata_.end()) {
    if (auto* v = std::get_if<std::int64_t>(&it->second)) {
      alignment = static_cast<std::uint64_t>(*v);
    }
  }
  const std::uint64_t header_end = static_cast<std::uint64_t>(p - base);
  const std::uint64_t data_section_start =
      (header_end + alignment - 1) / alignment * alignment;
  if (data_section_start > out->bytes_.size()) {
    throw std::runtime_error("gguf: data section start past EOF");
  }

  for (auto& t : pending) {
    const std::size_t sz = TypeSizeBytes(t.dtype, t.shape);
    const std::uint64_t off = data_section_start + t.data_offset;
    if (off + sz > out->bytes_.size()) {
      throw std::runtime_error("gguf: tensor data overruns file: " + t.name);
    }
    TensorInfo info;
    info.dtype = t.dtype;
    info.shape = std::move(t.shape);
    info.data = base + off;
    info.size_bytes = sz;
    out->tensors_.emplace(std::move(t.name), std::move(info));
  }

  return out;
}

std::vector<std::string> GgufFile::Names() const {
  std::vector<std::string> out;
  out.reserve(tensors_.size());
  for (const auto& kv : tensors_) out.push_back(kv.first);
  return out;
}

std::optional<GgufFile::TensorInfo> GgufFile::Get(
    std::string_view name) const {
  auto it = tensors_.find(std::string(name));
  if (it == tensors_.end()) return std::nullopt;
  return it->second;
}

std::optional<GgufFile::MetadataValue> GgufFile::Metadata(
    std::string_view key) const {
  auto it = metadata_.find(std::string(key));
  if (it == metadata_.end()) return std::nullopt;
  return it->second;
}

bool GgufFile::LooksLikeGguf(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[4];
  f.read(magic, 4);
  if (!f) return false;
  return magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' &&
          magic[3] == 'F';
}

void GgufFile::Write(
    const std::string& path,
    const std::unordered_map<std::string, MetadataValue>& metadata,
    const std::vector<std::pair<std::string, WriteTensor>>& tensors) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("gguf: open for write failed: " + path);
  }

  // Header.
  WriteLE<std::uint32_t>(out, kGgufMagic);
  WriteLE<std::uint32_t>(out, kGgufVersion);
  WriteLE<std::uint64_t>(out, tensors.size());
  WriteLE<std::uint64_t>(out, metadata.size());

  // Metadata.
  for (const auto& kv : metadata) {
    WriteString(out, kv.first);
    WriteMetaValue(out, kv.second);
  }

  // Tensor offsets are relative to the (aligned) data section start.
  // First compute the cumulative offsets so we can emit them in the
  // tensor-info section. Default alignment matches kDefaultAlignment;
  // a future override via metadata.general.alignment would need to
  // route through here too.
  std::uint64_t alignment = kDefaultAlignment;
  if (auto it = metadata.find("general.alignment");
      it != metadata.end()) {
    if (auto* v = std::get_if<std::int64_t>(&it->second)) {
      alignment = static_cast<std::uint64_t>(*v);
    }
  }
  std::vector<std::uint64_t> offsets(tensors.size(), 0);
  std::uint64_t cursor = 0;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    offsets[i] = cursor;
    const std::uint64_t pad =
        (alignment - (tensors[i].second.size_bytes % alignment)) % alignment;
    cursor += tensors[i].second.size_bytes + pad;
  }

  // Tensor info entries.
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& [name, t] = tensors[i];
    WriteString(out, name);
    WriteLE<std::uint32_t>(out, static_cast<std::uint32_t>(t.shape.size()));
    for (auto d : t.shape) WriteLE<std::uint64_t>(out, d);
    WriteLE<std::uint32_t>(out, static_cast<std::uint32_t>(t.dtype));
    WriteLE<std::uint64_t>(out, offsets[i]);
  }

  // Pad to alignment before the data section.
  const auto header_end = static_cast<std::uint64_t>(out.tellp());
  const auto data_section_start =
      (header_end + alignment - 1) / alignment * alignment;
  const auto header_pad = data_section_start - header_end;
  for (std::uint64_t i = 0; i < header_pad; ++i) {
    WriteLE<std::uint8_t>(out, 0);
  }

  // Tensor data, padded between entries to maintain alignment.
  for (const auto& [_, t] : tensors) {
    WriteLE(out, t.data, t.size_bytes);
    const std::uint64_t pad =
        (alignment - (t.size_bytes % alignment)) % alignment;
    for (std::uint64_t i = 0; i < pad; ++i) WriteLE<std::uint8_t>(out, 0);
  }

  if (!out) {
    throw std::runtime_error("gguf: write finalize failed: " + path);
  }
}

}  // namespace esm::io
