#include "esm_cpp/model.h"

#include "esm_cpp/artifact_cache.h"
#include "esm_cpp/artifact_trace_sha.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__F16C__)
#include <immintrin.h>
#endif

#include "esm_cpp/batch.h"
#include "esm_cpp/cpu_features.h"
#include "esm_cpp/io.h"
#include "esm_cpp/kernels.h"
#include "esm_cpp/profile.h"
#include "esm_cpp/smoothquant.h"
#include "esm_cpp/tokenizer.h"

namespace esm {

namespace {

constexpr float kTokenDropoutMaskRatioTrain = 0.15f * 0.8f;  // = 0.12

void CopyF32(const io::SafetensorsFile::TensorInfo& info, std::vector<float>& dst,
             const std::string& name) {
  if (info.dtype != "F32") {
    throw std::runtime_error("expected F32 for tensor " + name + " got " + info.dtype);
  }
  std::size_t n = info.size_bytes / sizeof(float);
  dst.resize(n);
  std::memcpy(dst.data(), info.data, info.size_bytes);
}

io::SafetensorsFile::TensorInfo Require(const io::SafetensorsFile& f,
                                         const std::string& name) {
  auto info = f.Get(name);
  if (!info) throw std::runtime_error("safetensors: missing tensor " + name);
  return *info;
}

void LoadLinear(const io::SafetensorsFile& f, const std::string& prefix,
                std::vector<float>& w, std::vector<float>& b) {
  CopyF32(Require(f, prefix + ".weight"), w, prefix + ".weight");
  CopyF32(Require(f, prefix + ".bias"), b, prefix + ".bias");
}

void LoadLayerNorm(const io::SafetensorsFile& f, const std::string& prefix,
                   std::vector<float>& w, std::vector<float>& b) {
  CopyF32(Require(f, prefix + ".weight"), w, prefix + ".weight");
  CopyF32(Require(f, prefix + ".bias"), b, prefix + ".bias");
}

}  // namespace

std::unique_ptr<Model> Model::LoadFromSafetensors(const std::string& path) {
  esm::MaybeLogIsaOnce();
  auto file = io::SafetensorsFile::Open(path);
  auto out = std::unique_ptr<Model>(new Model());

  // Embedding: [vocab_size, hidden_size]
  auto embed_info = Require(*file, "esm.embeddings.word_embeddings.weight");
  if (embed_info.shape.size() != 2) {
    throw std::runtime_error("unexpected embedding shape");
  }
  out->cfg_.vocab_size = static_cast<int>(embed_info.shape[0]);
  out->cfg_.hidden_size = static_cast<int>(embed_info.shape[1]);
  CopyF32(embed_info, out->embed_, "embeddings");

  if (out->cfg_.vocab_size != Tokenizer::kVocabSize) {
    throw std::runtime_error("vocab size mismatch: expected " +
                             std::to_string(Tokenizer::kVocabSize) + " got " +
                             std::to_string(out->cfg_.vocab_size));
  }

  // head_dim from rotary_embeddings.inv_freq[0] shape = [head_dim/2]
  auto inv_freq_info = Require(
      *file, "esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq");
  if (inv_freq_info.shape.size() != 1) {
    throw std::runtime_error("unexpected inv_freq shape");
  }
  out->cfg_.head_dim = 2 * static_cast<int>(inv_freq_info.shape[0]);
  if (out->cfg_.hidden_size % out->cfg_.head_dim != 0) {
    throw std::runtime_error("hidden_size not divisible by head_dim");
  }
  out->cfg_.num_attention_heads = out->cfg_.hidden_size / out->cfg_.head_dim;

  // intermediate_size from intermediate.dense.weight: [4d, d]
  auto inter_info = Require(
      *file, "esm.encoder.layer.0.intermediate.dense.weight");
  if (inter_info.shape.size() != 2) {
    throw std::runtime_error("unexpected intermediate shape");
  }
  out->cfg_.intermediate_size = static_cast<int>(inter_info.shape[0]);

  // Count layers.
  int num_layers = 0;
  while (file->Has("esm.encoder.layer." + std::to_string(num_layers) +
                   ".attention.LayerNorm.weight")) {
    ++num_layers;
  }
  if (num_layers == 0) throw std::runtime_error("no encoder layers found");
  out->cfg_.num_hidden_layers = num_layers;

  // ESM-2 always has these.
  out->cfg_.layer_norm_eps = 1e-5f;
  out->cfg_.token_dropout = true;
  out->cfg_.mask_token_id = Tokenizer::kMaskId;

  // Per-layer weights.
  out->layers_.resize(static_cast<std::size_t>(num_layers));
  for (int i = 0; i < num_layers; ++i) {
    const std::string p = "esm.encoder.layer." + std::to_string(i);
    auto& lw = out->layers_[static_cast<std::size_t>(i)];
    LoadLayerNorm(*file, p + ".attention.LayerNorm", lw.attn_ln_w, lw.attn_ln_b);
    LoadLinear(*file, p + ".attention.self.query", lw.q_w, lw.q_b);
    LoadLinear(*file, p + ".attention.self.key", lw.k_w, lw.k_b);
    LoadLinear(*file, p + ".attention.self.value", lw.v_w, lw.v_b);
    LoadLinear(*file, p + ".attention.output.dense", lw.out_w, lw.out_b);
    LoadLayerNorm(*file, p + ".LayerNorm", lw.ffn_ln_w, lw.ffn_ln_b);
    LoadLinear(*file, p + ".intermediate.dense", lw.fc1_w, lw.fc1_b);
    LoadLinear(*file, p + ".output.dense", lw.fc2_w, lw.fc2_b);
  }

  // Final LayerNorm (after the last encoder block).
  LoadLayerNorm(*file, "esm.encoder.emb_layer_norm_after",
                out->final_ln_w_, out->final_ln_b_);

  // LM head.
  LoadLinear(*file, "lm_head.dense", out->lm_dense_w_, out->lm_dense_b_);
  LoadLayerNorm(*file, "lm_head.layer_norm", out->lm_ln_w_, out->lm_ln_b_);
  CopyF32(Require(*file, "lm_head.bias"), out->lm_decoder_bias_, "lm_head.bias");
  // lm_head.decoder.weight is tied to esm.embeddings.word_embeddings.weight;
  // we reuse out->embed_ at logit time and don't load a separate copy.

  out->TryAutoLoadAppleArtifacts(path);
  return out;
}

namespace {

// HF safetensors <-> GGUF tensor name map. GGUF names mirror llama.cpp's
// convention where possible; ESM-specific tensors (lm_head dense + LN +
// per-vocab decoder bias) get descriptive names rather than llama.cpp's
// generic "output.*".
std::string GgufLayerName(int i, const std::string& suffix) {
  return "blk." + std::to_string(i) + "." + suffix;
}

void GgufLoadVector(const io::GgufFile& gf, const std::string& name,
                    std::vector<float>& dst) {
  auto info = gf.Get(name);
  if (!info) throw std::runtime_error("gguf: missing tensor " + name);
  if (info->dtype != io::GgufFile::GgmlType::F32) {
    throw std::runtime_error("gguf: expected F32 for " + name);
  }
  const std::size_t n = info->size_bytes / sizeof(float);
  dst.resize(n);
  std::memcpy(dst.data(), info->data, info->size_bytes);
}

// Slice 5: Q8_ESM tensor blob layout is [N, K] int8 weights followed by
// [N] float32 per-channel scales, contiguous. Shape on the wire is
// [K, N] (in, out) matching llama.cpp's natural axis order.
void GgufLoadQuant(const io::GgufFile& gf, const std::string& name,
                   esm::quant::QuantizedTensor& dst) {
  auto info = gf.Get(name);
  if (!info) throw std::runtime_error("gguf: missing tensor " + name);
  if (info->dtype != io::GgufFile::GgmlType::Q8_ESM) {
    throw std::runtime_error("gguf: expected Q8_ESM for " + name);
  }
  if (info->shape.size() != 2) {
    throw std::runtime_error("gguf: Q8_ESM " + name + " must be 2-D");
  }
  const std::size_t K = info->shape[0];
  const std::size_t N = info->shape[1];
  const std::size_t packed_bytes = N * K;
  dst.packed.assign(N * K, 0);
  std::memcpy(dst.packed.data(), info->data, packed_bytes);
  dst.per_channel_scales.assign(N, 0.0f);
  std::memcpy(dst.per_channel_scales.data(),
              info->data + packed_bytes, N * sizeof(float));
  dst.N = static_cast<int>(N);
  dst.K = static_cast<int>(K);
  esm::quant::BuildKernelCache(&dst);
}

std::int64_t GgufMetaInt(const io::GgufFile& gf, const std::string& key) {
  auto v = gf.Metadata(key);
  if (!v) throw std::runtime_error("gguf: missing metadata " + key);
  if (auto* i = std::get_if<std::int64_t>(&*v)) return *i;
  throw std::runtime_error("gguf: metadata " + key + " is not int");
}

}  // namespace

std::unique_ptr<Model> Model::LoadFromGguf(const std::string& path) {
  esm::MaybeLogIsaOnce();
  auto file = io::GgufFile::Open(path);
  auto arch = file->Metadata("general.architecture");
  if (!arch || std::get<std::string>(*arch) != "esm") {
    throw std::runtime_error(
        "gguf: not an esm-arch file (general.architecture must be 'esm')");
  }

  auto out = std::unique_ptr<Model>(new Model());
  out->cfg_.num_hidden_layers =
      static_cast<int>(GgufMetaInt(*file, "esm.block_count"));
  out->cfg_.hidden_size =
      static_cast<int>(GgufMetaInt(*file, "esm.embedding_length"));
  out->cfg_.intermediate_size =
      static_cast<int>(GgufMetaInt(*file, "esm.feed_forward_length"));
  out->cfg_.num_attention_heads =
      static_cast<int>(GgufMetaInt(*file, "esm.attention.head_count"));
  out->cfg_.head_dim =
      static_cast<int>(GgufMetaInt(*file, "esm.rope.dimension_count"));
  out->cfg_.vocab_size = Tokenizer::kVocabSize;
  out->cfg_.token_dropout = true;
  out->cfg_.mask_token_id = Tokenizer::kMaskId;
  // layer_norm_epsilon stored as float; fall back to the ESM default if
  // a writer forgot to include it.
  if (auto eps = file->Metadata("esm.attention.layer_norm_epsilon")) {
    if (auto* d = std::get_if<double>(&*eps)) {
      out->cfg_.layer_norm_eps = static_cast<float>(*d);
    } else {
      out->cfg_.layer_norm_eps = 1e-5f;
    }
  } else {
    out->cfg_.layer_norm_eps = 1e-5f;
  }

  GgufLoadVector(*file, "token_embd.weight", out->embed_);
  // Slice 5: detect quantized state via metadata flag. When set, per-layer
  // Linear weights are stored as Q8_ESM blobs; biases stay FP32. lm_head
  // and embed stay FP32 regardless (Slice 5 escape list).
  bool quantized = false;
  if (auto v = file->Metadata("esm.weights_quantized")) {
    if (auto* b = std::get_if<bool>(&*v)) quantized = *b;
  }
  out->cfg_.weights_quantized = quantized;
  if (auto v = file->Metadata("esm.first_block_fc1_fp16")) {
    if (auto* b = std::get_if<bool>(&*v)) {
      out->cfg_.first_block_fc1_fp16 = *b;
    }
  }
  out->layers_.resize(static_cast<std::size_t>(out->cfg_.num_hidden_layers));
  for (int i = 0; i < out->cfg_.num_hidden_layers; ++i) {
    auto& lw = out->layers_[static_cast<std::size_t>(i)];
    GgufLoadVector(*file, GgufLayerName(i, "attn_norm.weight"), lw.attn_ln_w);
    GgufLoadVector(*file, GgufLayerName(i, "attn_norm.bias"), lw.attn_ln_b);
    GgufLoadVector(*file, GgufLayerName(i, "attn_q.bias"), lw.q_b);
    GgufLoadVector(*file, GgufLayerName(i, "attn_k.bias"), lw.k_b);
    GgufLoadVector(*file, GgufLayerName(i, "attn_v.bias"), lw.v_b);
    GgufLoadVector(*file, GgufLayerName(i, "attn_output.bias"), lw.out_b);
    GgufLoadVector(*file, GgufLayerName(i, "ffn_norm.weight"), lw.ffn_ln_w);
    GgufLoadVector(*file, GgufLayerName(i, "ffn_norm.bias"), lw.ffn_ln_b);
    GgufLoadVector(*file, GgufLayerName(i, "ffn_up.bias"), lw.fc1_b);
    GgufLoadVector(*file, GgufLayerName(i, "ffn_down.bias"), lw.fc2_b);
    if (quantized) {
      GgufLoadQuant(*file, GgufLayerName(i, "attn_q.weight"), lw.q_w_int8);
      GgufLoadQuant(*file, GgufLayerName(i, "attn_k.weight"), lw.k_w_int8);
      GgufLoadQuant(*file, GgufLayerName(i, "attn_v.weight"), lw.v_w_int8);
      GgufLoadQuant(*file, GgufLayerName(i, "attn_output.weight"),
                     lw.out_w_int8);
      GgufLoadQuant(*file, GgufLayerName(i, "ffn_up.weight"),
                     lw.fc1_w_int8);
      GgufLoadQuant(*file, GgufLayerName(i, "ffn_down.weight"),
                     lw.fc2_w_int8);
    } else {
      GgufLoadVector(*file, GgufLayerName(i, "attn_q.weight"), lw.q_w);
      GgufLoadVector(*file, GgufLayerName(i, "attn_k.weight"), lw.k_w);
      GgufLoadVector(*file, GgufLayerName(i, "attn_v.weight"), lw.v_w);
      GgufLoadVector(*file, GgufLayerName(i, "attn_output.weight"),
                      lw.out_w);
      GgufLoadVector(*file, GgufLayerName(i, "ffn_up.weight"), lw.fc1_w);
      GgufLoadVector(*file, GgufLayerName(i, "ffn_down.weight"), lw.fc2_w);
    }
  }
  GgufLoadVector(*file, "enc_norm.weight", out->final_ln_w_);
  GgufLoadVector(*file, "enc_norm.bias", out->final_ln_b_);
  GgufLoadVector(*file, "lm_head.weight", out->lm_dense_w_);
  GgufLoadVector(*file, "lm_head.bias", out->lm_dense_b_);
  GgufLoadVector(*file, "lm_head_norm.weight", out->lm_ln_w_);
  GgufLoadVector(*file, "lm_head_norm.bias", out->lm_ln_b_);
  GgufLoadVector(*file, "output.bias", out->lm_decoder_bias_);

  out->TryAutoLoadAppleArtifacts(path);
  return out;
}

std::unique_ptr<Model> Model::Load(const std::string& path) {
  if (io::GgufFile::LooksLikeGguf(path)) return LoadFromGguf(path);
  return LoadFromSafetensors(path);
}

namespace {

// Walk a directory of M-<m>/ subdirs, sorted ascending by M.
std::vector<std::pair<int, std::string>> ListAneBuckets(const std::string& dir) {
  namespace fs = std::filesystem;
  std::vector<std::pair<int, std::string>> out;
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return out;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (!entry.is_directory()) continue;
    const std::string n = entry.path().filename().string();
    if (n.size() < 3 || n.substr(0, 2) != "M-") continue;
    int m = 0;
    try { m = std::stoi(n.substr(2)); } catch (...) { continue; }
    if (m > 0) out.emplace_back(m, entry.path().string());
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

std::size_t Model::LoadAneArtifacts(const std::string& dir) {
  const int d = cfg_.hidden_size;
  const int ffn = cfg_.intermediate_size;
  std::size_t loaded = 0;
  auto buckets = ListAneBuckets(dir);
  auto try_load_into = [&](std::vector<std::unique_ptr<AppleAneContext>>& vec,
                           const std::string& name, int K, int N) {
    vec.clear();
    for (const auto& [m, sub] : buckets) {
      auto p = sub + "/" + name + ".mlmodelc";
      auto ctx = AppleAneContext::LoadFromDir(p, m, K, N);
      if (ctx) { vec.push_back(std::move(ctx)); ++loaded; }
    }
    // Vec is already sorted by M (buckets sorted; we append in order).
  };
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    auto& lw = layers_[i];
    const std::string base = "esm.encoder.layer." + std::to_string(i);
    try_load_into(lw.ane_q,   base + ".attention.self.query",   d, d);
    try_load_into(lw.ane_k,   base + ".attention.self.key",     d, d);
    try_load_into(lw.ane_v,   base + ".attention.self.value",   d, d);
    try_load_into(lw.ane_out, base + ".attention.output.dense", d, d);
    try_load_into(lw.ane_fc1, base + ".intermediate.dense",     d, ffn);
    try_load_into(lw.ane_fc2, base + ".output.dense",           ffn, d);
  }
  try_load_into(ane_lm_dense_, "lm_head.dense", d, d);
  return loaded;
}

std::size_t Model::LoadAmxArtifacts(const std::string& dir) {
  // Per-Linear naming mirrors tools/build_amx_artifacts.py exactly: the
  // safetensors weight key minus the ".weight" suffix, plus ".mlmodelc".
  // Missing artifacts are not an error — the affected Linear silently falls
  // back to the default INT8/FP32 path at the next forward.
  const int d = cfg_.hidden_size;
  const int ffn = cfg_.intermediate_size;
  std::size_t loaded = 0;
  auto try_load = [&](const std::string& name, int K, int N)
      -> std::unique_ptr<AppleAmxContext> {
    auto p = dir + "/" + name + ".mlmodelc";
    auto ctx = AppleAmxContext::LoadFromDir(p, K, N);
    if (ctx) ++loaded;
    return ctx;
  };
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    auto& lw = layers_[i];
    const std::string base = "esm.encoder.layer." + std::to_string(i);
    lw.amx_q = try_load(base + ".attention.self.query", d, d);
    lw.amx_k = try_load(base + ".attention.self.key", d, d);
    lw.amx_v = try_load(base + ".attention.self.value", d, d);
    lw.amx_out = try_load(base + ".attention.output.dense", d, d);
    lw.amx_fc1 = try_load(base + ".intermediate.dense", d, ffn);
    lw.amx_fc2 = try_load(base + ".output.dense", ffn, d);
  }
  amx_lm_dense_ = try_load("lm_head.dense", d, d);
  return loaded;
}

namespace {
// Cheap JSON field reader for the artifact manifest. We only need a single
// string field (`trace_sha`), so pulling in nlohmann_json here is overkill.
// Returns empty string on any parse failure — caller treats that as "no
// manifest, no warning."
std::string ReadManifestTraceSha(const std::filesystem::path& manifest_path) {
  std::ifstream f(manifest_path);
  if (!f.is_open()) return "";
  std::string content((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
  // Look for "trace_sha": "<value>" — simple substring scan; manifest is
  // always written by our own helper so the format is stable.
  const std::string key = "\"trace_sha\"";
  auto kp = content.find(key);
  if (kp == std::string::npos) return "";
  auto colon = content.find(':', kp + key.size());
  if (colon == std::string::npos) return "";
  auto q1 = content.find('"', colon + 1);
  if (q1 == std::string::npos) return "";
  auto q2 = content.find('"', q1 + 1);
  if (q2 == std::string::npos) return "";
  return content.substr(q1 + 1, q2 - q1 - 1);
}

// Warn-once helper — same artifact path shouldn't double-log if the user
// somehow auto-loads twice.
void CheckManifestFreshness(const std::filesystem::path& manifest_path,
                            const char* kind_label) {
  const std::string artifact_sha = ReadManifestTraceSha(manifest_path);
  if (artifact_sha.empty()) return;  // no manifest or unparseable -> silent
  if (artifact_sha == kArtifactTraceSha) return;
  std::fprintf(stderr,
               "[esm_cpp] warning: %s artifact at %s was built against a "
               "different trace_sha (%s vs %s for this build). The artifact "
               "is still being used — refresh via "
               "`esm-cpp-fetch-artifacts --model <id>` if you see quality "
               "regressions.\n",
               kind_label, manifest_path.parent_path().c_str(),
               artifact_sha.substr(0, 12).c_str(),
               std::string(kArtifactTraceSha).substr(0, 12).c_str());
}
}  // namespace

void Model::TryAutoLoadAppleArtifacts(const std::string& weights_path) {
#ifdef ESM_APPLE_AMX_AVAILABLE
  namespace fs = std::filesystem;
  // ESM_APPLE_DEBUG_AUTOLOAD=1 dumps the candidate roots + per-root load
  // result to stderr. Useful when a user's auto-engage isn't firing and
  // they can't tell whether it's a missing artifact, a bad path, or a
  // failed compile.
  const bool debug = std::getenv("ESM_APPLE_DEBUG_AUTOLOAD") != nullptr;
  auto roots = ArtifactCache::CandidateRoots(weights_path);
  if (debug) {
    std::fprintf(stderr, "[autoload] weights=%s  candidates=%zu\n",
                 weights_path.c_str(), roots.size());
    for (const auto& r : roots) {
      std::fprintf(stderr, "[autoload]   candidate: %s\n", r.c_str());
    }
  }
  for (const auto& root : roots) {
    if (amx_artifacts_path_.empty()) {
      const fs::path amx_dir = root / "amx-fp16";
      std::error_code ec;
      if (fs::is_directory(amx_dir, ec)) {
        const std::size_t n = LoadAmxArtifacts(amx_dir.string());
        if (debug) {
          std::fprintf(stderr, "[autoload]   amx %s -> %zu contexts\n",
                       amx_dir.c_str(), n);
        }
        if (n > 0) {
          amx_artifacts_path_ = amx_dir.string();
          CheckManifestFreshness(amx_dir / "esm_cpp_artifact.json", "amx-fp16");
        }
      }
    }

    // Whole-graph: scan B-<B>_L-<L>/whole_graph.mlmodelc subdirs and register
    // each via the public LoadWholeGraphArtifact. Multiple shapes per root
    // are loaded together (each forward picks by (B, L)). First root that
    // yields any registration wins; we don't merge across roots.
    if (whole_graph_.empty()) {
      const fs::path wg_root = root / "whole-graph";
      std::error_code ec;
      if (fs::is_directory(wg_root, ec)) {
        std::size_t registered = 0;
        for (const auto& entry : fs::directory_iterator(wg_root, ec)) {
          if (!entry.is_directory()) continue;
          const std::string n = entry.path().filename().string();
          // Pattern: B-<B>_L-<L>
          if (n.size() < 6 || n.substr(0, 2) != "B-") continue;
          const auto underscore = n.find("_L-");
          if (underscore == std::string::npos) continue;
          int B = 0, L = 0;
          try {
            B = std::stoi(n.substr(2, underscore - 2));
            L = std::stoi(n.substr(underscore + 3));
          } catch (...) { continue; }
          if (B <= 0 || L <= 0) continue;
          const fs::path bundle = entry.path() / "whole_graph.mlmodelc";
          if (!fs::is_directory(bundle, ec)) continue;
          // Auto-load uses CPU_AND_NE as the default compute_units. Power
          // users who want CPU_ONLY / ALL go through the explicit
          // LoadWholeGraphArtifact API.
          const bool ok = LoadWholeGraphArtifact(
              bundle.string(), B, L,
              WholeGraphComputeUnits::kCpuAndNeuralEngine);
          if (debug) {
            std::fprintf(stderr, "[autoload]   wg %s (B=%d L=%d) -> %d\n",
                         bundle.c_str(), B, L, ok ? 1 : 0);
          }
          if (ok) {
            ++registered;
            CheckManifestFreshness(entry.path() / "esm_cpp_artifact.json",
                                    "whole-graph");
          }
        }
        if (registered > 0) {
          whole_graph_path_ = wg_root.string();
        }
      }
    }

    if (!amx_artifacts_path_.empty() && !whole_graph_.empty()) break;
  }
#else
  (void)weights_path;
#endif
}

bool Model::LoadWholeGraphArtifact(const std::string& dir, int B, int L,
                                   WholeGraphComputeUnits cu) {
  // Replace any existing registration at the same (B, L) — convenient when
  // a benchmark wants to A/B compute-units without restarting.
  for (auto& reg : whole_graph_) {
    if (reg.B == B && reg.L == L) {
      auto ctx = AppleWholeGraphContext::LoadFromDir(dir, B, L, cfg_.vocab_size, cu);
      if (!ctx) return false;
      reg.ctx = std::move(ctx);
      return true;
    }
  }
  auto ctx = AppleWholeGraphContext::LoadFromDir(dir, B, L, cfg_.vocab_size, cu);
  if (!ctx) return false;
  whole_graph_.push_back(WholeGraphReg{B, L, std::move(ctx)});
  return true;
}

std::vector<std::pair<int, int>> Model::whole_graph_shapes() const {
  std::vector<std::pair<int, int>> out;
  out.reserve(whole_graph_.size());
  for (const auto& reg : whole_graph_) out.emplace_back(reg.B, reg.L);
  return out;
}

std::vector<float> Model::ForwardWholeGraph(
    std::span<const std::int32_t> input_ids,
    std::span<const std::int32_t> attention_mask, int B, int L,
    bool* ok) const {
  auto report = [ok](bool good) { if (ok) *ok = good; };
  const std::size_t n = static_cast<std::size_t>(B) * L;
  if (input_ids.size() != n || attention_mask.size() != n) {
    report(false);
    return {};
  }
  const AppleWholeGraphContext* ctx = nullptr;
  for (const auto& reg : whole_graph_) {
    if (reg.B == B && reg.L == L) { ctx = reg.ctx.get(); break; }
  }
  if (ctx == nullptr) { report(false); return {}; }
  std::vector<float> logits(n * static_cast<std::size_t>(cfg_.vocab_size));
  auto* mut = const_cast<AppleWholeGraphContext*>(ctx);
  auto status = mut->Execute(input_ids.data(), attention_mask.data(), logits.data());
  if (!status.ok()) { report(false); return {}; }
  report(true);
  return logits;
}

void Model::SaveToGguf(const std::string& path) const {
  const int d = cfg_.hidden_size;
  const int ffn = cfg_.intermediate_size;
  const int L = cfg_.num_hidden_layers;
  const int V = cfg_.vocab_size;

  std::unordered_map<std::string, io::GgufFile::MetadataValue> meta;
  meta.emplace("general.architecture", std::string("esm"));
  meta.emplace("general.alignment", static_cast<std::int64_t>(32));
  // general.file_type: 1 = all F32; we use 2 to flag the Q8_ESM mixed
  // FP32+INT8 path (decimal codepoint, not standard). Phase 5 reader
  // keys off the esm.weights_quantized bool, not the file_type.
  meta.emplace("general.file_type",
                cfg_.weights_quantized ? static_cast<std::int64_t>(2)
                                       : static_cast<std::int64_t>(1));
  meta.emplace("esm.block_count", static_cast<std::int64_t>(L));
  meta.emplace("esm.embedding_length", static_cast<std::int64_t>(d));
  meta.emplace("esm.feed_forward_length", static_cast<std::int64_t>(ffn));
  meta.emplace("esm.attention.head_count",
                static_cast<std::int64_t>(cfg_.num_attention_heads));
  meta.emplace("esm.attention.head_count_kv",
                static_cast<std::int64_t>(cfg_.num_attention_heads));
  meta.emplace("esm.attention.layer_norm_epsilon",
                static_cast<double>(cfg_.layer_norm_eps));
  meta.emplace("esm.rope.dimension_count",
                static_cast<std::int64_t>(cfg_.head_dim));
  meta.emplace("esm.context_length",
                static_cast<std::int64_t>(Tokenizer::kModelMaxLength));
  meta.emplace("esm.gguf_writer", std::string("esm-cpp 0.1.0"));
  meta.emplace("esm.weights_quantized", cfg_.weights_quantized);
  meta.emplace("esm.first_block_fc1_fp16", cfg_.first_block_fc1_fp16);

  // Helper: build a WriteTensor for an FP32 vector with the given
  // logical [out, in] (or [N] for bias / LN params) shape.
  auto make_f32 = [](const std::vector<float>& w,
                      std::vector<std::uint64_t> shape)
      -> io::GgufFile::WriteTensor {
    io::GgufFile::WriteTensor t;
    t.dtype = io::GgufFile::GgmlType::F32;
    t.shape = std::move(shape);
    t.data = w.data();
    t.size_bytes = w.size() * sizeof(float);
    return t;
  };

  // Slice 5: per-quantized-Linear, build a contiguous blob of
  // [N*K int8 packed bytes][N float32 scales]. Blobs outlive the Write
  // call via this owning vector.
  std::vector<std::vector<std::byte>> q8_blobs;
  if (cfg_.weights_quantized) {
    q8_blobs.reserve(static_cast<std::size_t>(L) * 6);
  }
  auto make_q8 = [&q8_blobs](const esm::quant::QuantizedTensor& qt,
                              std::uint64_t K, std::uint64_t N)
      -> io::GgufFile::WriteTensor {
    const std::size_t packed_bytes = qt.packed.size();
    const std::size_t scale_bytes =
        qt.per_channel_scales.size() * sizeof(float);
    std::vector<std::byte> blob(packed_bytes + scale_bytes);
    std::memcpy(blob.data(), qt.packed.data(), packed_bytes);
    std::memcpy(blob.data() + packed_bytes,
                 qt.per_channel_scales.data(), scale_bytes);
    q8_blobs.push_back(std::move(blob));
    io::GgufFile::WriteTensor t;
    t.dtype = io::GgufFile::GgmlType::Q8_ESM;
    t.shape = {K, N};
    t.data = q8_blobs.back().data();
    t.size_bytes = q8_blobs.back().size();
    return t;
  };

  std::vector<std::pair<std::string, io::GgufFile::WriteTensor>> tensors;
  tensors.reserve(
      static_cast<std::size_t>(L) * 16 + 8);

  tensors.emplace_back(
      "token_embd.weight",
      make_f32(embed_,
                {static_cast<std::uint64_t>(d),
                 static_cast<std::uint64_t>(V)}));
  const std::uint64_t d_u = static_cast<std::uint64_t>(d);
  const std::uint64_t ffn_u = static_cast<std::uint64_t>(ffn);
  for (int i = 0; i < L; ++i) {
    const auto& lw = layers_[static_cast<std::size_t>(i)];
    tensors.emplace_back(
        GgufLayerName(i, "attn_norm.weight"),
        make_f32(lw.attn_ln_w, {d_u}));
    tensors.emplace_back(
        GgufLayerName(i, "attn_norm.bias"),
        make_f32(lw.attn_ln_b, {d_u}));
    if (cfg_.weights_quantized) {
      tensors.emplace_back(GgufLayerName(i, "attn_q.weight"),
                            make_q8(lw.q_w_int8, d_u, d_u));
      tensors.emplace_back(GgufLayerName(i, "attn_k.weight"),
                            make_q8(lw.k_w_int8, d_u, d_u));
      tensors.emplace_back(GgufLayerName(i, "attn_v.weight"),
                            make_q8(lw.v_w_int8, d_u, d_u));
      tensors.emplace_back(GgufLayerName(i, "attn_output.weight"),
                            make_q8(lw.out_w_int8, d_u, d_u));
    } else {
      const std::vector<std::uint64_t> d_by_d = {d_u, d_u};
      tensors.emplace_back(GgufLayerName(i, "attn_q.weight"),
                            make_f32(lw.q_w, d_by_d));
      tensors.emplace_back(GgufLayerName(i, "attn_k.weight"),
                            make_f32(lw.k_w, d_by_d));
      tensors.emplace_back(GgufLayerName(i, "attn_v.weight"),
                            make_f32(lw.v_w, d_by_d));
      tensors.emplace_back(GgufLayerName(i, "attn_output.weight"),
                            make_f32(lw.out_w, d_by_d));
    }
    tensors.emplace_back(GgufLayerName(i, "attn_q.bias"),
                          make_f32(lw.q_b, {d_u}));
    tensors.emplace_back(GgufLayerName(i, "attn_k.bias"),
                          make_f32(lw.k_b, {d_u}));
    tensors.emplace_back(GgufLayerName(i, "attn_v.bias"),
                          make_f32(lw.v_b, {d_u}));
    tensors.emplace_back(GgufLayerName(i, "attn_output.bias"),
                          make_f32(lw.out_b, {d_u}));
    tensors.emplace_back(GgufLayerName(i, "ffn_norm.weight"),
                          make_f32(lw.ffn_ln_w, {d_u}));
    tensors.emplace_back(GgufLayerName(i, "ffn_norm.bias"),
                          make_f32(lw.ffn_ln_b, {d_u}));
    if (cfg_.weights_quantized) {
      tensors.emplace_back(GgufLayerName(i, "ffn_up.weight"),
                            make_q8(lw.fc1_w_int8, d_u, ffn_u));
      tensors.emplace_back(GgufLayerName(i, "ffn_down.weight"),
                            make_q8(lw.fc2_w_int8, ffn_u, d_u));
    } else {
      tensors.emplace_back(GgufLayerName(i, "ffn_up.weight"),
                            make_f32(lw.fc1_w, {d_u, ffn_u}));
      tensors.emplace_back(GgufLayerName(i, "ffn_down.weight"),
                            make_f32(lw.fc2_w, {ffn_u, d_u}));
    }
    tensors.emplace_back(GgufLayerName(i, "ffn_up.bias"),
                          make_f32(lw.fc1_b, {ffn_u}));
    tensors.emplace_back(GgufLayerName(i, "ffn_down.bias"),
                          make_f32(lw.fc2_b, {d_u}));
  }
  tensors.emplace_back(
      "enc_norm.weight",
      make_f32(final_ln_w_, {static_cast<std::uint64_t>(d)}));
  tensors.emplace_back(
      "enc_norm.bias",
      make_f32(final_ln_b_, {static_cast<std::uint64_t>(d)}));
  tensors.emplace_back(
      "lm_head.weight",
      make_f32(lm_dense_w_,
                {static_cast<std::uint64_t>(d),
                 static_cast<std::uint64_t>(d)}));
  tensors.emplace_back(
      "lm_head.bias",
      make_f32(lm_dense_b_, {static_cast<std::uint64_t>(d)}));
  tensors.emplace_back(
      "lm_head_norm.weight",
      make_f32(lm_ln_w_, {static_cast<std::uint64_t>(d)}));
  tensors.emplace_back(
      "lm_head_norm.bias",
      make_f32(lm_ln_b_, {static_cast<std::uint64_t>(d)}));
  tensors.emplace_back(
      "output.bias",
      make_f32(lm_decoder_bias_, {static_cast<std::uint64_t>(V)}));

  io::GgufFile::Write(path, meta, tensors);
}

namespace {

// Embed: gather rows from [vocab_size, d] and apply the inference-time
// token_dropout rescale. ESM zeroes mask-token embeddings and rescales by
// (1 - 0.15*0.8) / (1 - observed_mask_fraction) = 0.88 / (1 - omf).
//
// Per-sequence rescale: each sequence b in [0, batch_size) computes its
// own mask_ratio over tokens in [cu_seqlens[b], cu_seqlens[b+1]), so the
// packed-batch path doesn't average mask densities across sequences. The
// B=1 path (Forward) collapses to the Phase 0 behavior bit-for-bit.
void Embed(const Config& cfg, const float* embed_w,
           std::span<const int32_t> ids, std::span<const int32_t> mask,
           const int* cu_seqlens, int batch_size, float* out) {
  const int d = cfg.hidden_size;
  for (int b = 0; b < batch_size; ++b) {
    const int start = cu_seqlens[b];
    const int end = cu_seqlens[b + 1];
    int src_len = 0;
    int observed_masks = 0;
    for (int t = start; t < end; ++t) {
      const std::size_t ti = static_cast<std::size_t>(t);
      if (mask.empty() || mask[ti] != 0) ++src_len;
      if (ids[ti] == cfg.mask_token_id) ++observed_masks;
    }
    const float mask_ratio_observed =
        (src_len > 0)
            ? static_cast<float>(observed_masks) / static_cast<float>(src_len)
            : 0.0f;
    // (1 - 0.12) / (1 - obs). When obs == 0 this is 0.88 — the well-known
    // "ESM scales everything by 0.88 at inference" gotcha.
    const float scale = (1.0f - kTokenDropoutMaskRatioTrain) /
                        std::max(1.0f - mask_ratio_observed, 1e-12f);
    for (int t = start; t < end; ++t) {
      const std::size_t ti = static_cast<std::size_t>(t);
      std::int32_t id = ids[ti];
      const float* row = embed_w + static_cast<long>(id) * d;
      float* out_row = out + static_cast<long>(t) * d;
      bool zero_row = (id == cfg.mask_token_id);
      bool is_pad = !mask.empty() && mask[ti] == 0;
      if (zero_row) {
        for (int i = 0; i < d; ++i) out_row[i] = 0.0f;
      } else {
        for (int i = 0; i < d; ++i) out_row[i] = row[i] * scale;
      }
      if (is_pad) {
        // HF multiplies the post-rescale embedding by attention_mask,
        // which zeroes pad rows entirely.
        for (int i = 0; i < d; ++i) out_row[i] = 0.0f;
      }
    }
  }
}

// Layout helpers: with the cu_seqlens-packed kernels, Q/K/V projections
// land directly in [L, H, head_dim] (= [L, H*head_dim] in memory). The
// old SplitHeads memcpy from [L, H*dh] -> [H, L, dh] is gone.
// Resolve the per-Linear AMX context (or nullptr if not loaded / non-Apple).
// Pointer-to-member lets us reuse one helper for all six per-layer Linears.
inline AppleAmxContext* AmxOf(
    const LayerWeights& w,
    std::unique_ptr<AppleAmxContext> LayerWeights::*field) {
  return (w.*field).get();
}

// Phase 12: ANE dispatch with pad-to-bucket + chunking when M > max_bucket.
// `vec` is per-Linear contexts sorted ascending by bucket M. Returns Ok on
// success; the LinearProj caller falls back on failure. Bias is baked into the
// artifact, so the caller does NOT add bias separately on success.
//
// Strategy:
//   1. If M <= max bucket: pick the smallest bucket >= M; if M == bucket_m,
//      Execute directly (zero-copy); else pad input to bucket_m in scratch_in
//      and copy the first M rows of scratch_out back into `out`.
//   2. If M > max bucket: chunk by max_bucket. Full chunks Execute directly
//      against (in + offset*K, out + offset*N); the partial tail goes through
//      scratch_in/out the same way as case 1.
inline Status DispatchAne(
    const std::vector<std::unique_ptr<AppleAneContext>>& vec,
    const float* in, float* out, int M, int K, int N) {
  if (vec.empty()) return {StatusCode::kNotFound, "ane: no contexts"};
  // Find smallest bucket >= M.
  AppleAneContext* fit = nullptr;
  for (const auto& c : vec) {
    if (c->bucket_m() >= M) { fit = c.get(); break; }
  }
  if (fit && fit->k() == K && fit->n() == N) {
    if (M == fit->bucket_m()) {
      return fit->Execute(in, out);  // zero-copy fast path
    }
    static thread_local std::vector<float> s_in, s_out;
    const std::size_t in_bytes  = static_cast<std::size_t>(fit->bucket_m()) * K;
    const std::size_t out_bytes = static_cast<std::size_t>(fit->bucket_m()) * N;
    if (s_in.size() < in_bytes) s_in.assign(in_bytes, 0.0f);
    if (s_out.size() < out_bytes) s_out.assign(out_bytes, 0.0f);
    std::memcpy(s_in.data(), in,
                static_cast<std::size_t>(M) * K * sizeof(float));
    std::memset(s_in.data() + static_cast<std::size_t>(M) * K, 0,
                (static_cast<std::size_t>(fit->bucket_m()) - M) * K *
                    sizeof(float));
    Status s = fit->Execute(s_in.data(), s_out.data());
    if (!s.ok()) return s;
    std::memcpy(out, s_out.data(),
                static_cast<std::size_t>(M) * N * sizeof(float));
    return Status::Ok();
  }
  // M > max bucket → chunk.
  AppleAneContext* max_ctx = vec.back().get();
  if (max_ctx->k() != K || max_ctx->n() != N) {
    return {StatusCode::kInternal, "ane: K/N mismatch on max bucket"};
  }
  const int M_b = max_ctx->bucket_m();
  int offset = 0;
  static thread_local std::vector<float> s_in, s_out;
  while (offset < M) {
    int chunk_M = std::min(M - offset, M_b);
    if (chunk_M == M_b) {
      Status s = max_ctx->Execute(in + static_cast<std::size_t>(offset) * K,
                                  out + static_cast<std::size_t>(offset) * N);
      if (!s.ok()) return s;
    } else {
      const std::size_t in_bytes  = static_cast<std::size_t>(M_b) * K;
      const std::size_t out_bytes = static_cast<std::size_t>(M_b) * N;
      if (s_in.size() < in_bytes) s_in.assign(in_bytes, 0.0f);
      if (s_out.size() < out_bytes) s_out.assign(out_bytes, 0.0f);
      std::memcpy(s_in.data(),
                  in + static_cast<std::size_t>(offset) * K,
                  static_cast<std::size_t>(chunk_M) * K * sizeof(float));
      std::memset(s_in.data() + static_cast<std::size_t>(chunk_M) * K, 0,
                  static_cast<std::size_t>(M_b - chunk_M) * K * sizeof(float));
      Status s = max_ctx->Execute(s_in.data(), s_out.data());
      if (!s.ok()) return s;
      std::memcpy(out + static_cast<std::size_t>(offset) * N, s_out.data(),
                  static_cast<std::size_t>(chunk_M) * N * sizeof(float));
    }
    offset += chunk_M;
  }
  return Status::Ok();
}

// Resolve the per-Linear ANE bucket vector (or empty if not loaded).
inline const std::vector<std::unique_ptr<AppleAneContext>>& AneOf(
    const LayerWeights& w,
    std::vector<std::unique_ptr<AppleAneContext>> LayerWeights::*field) {
  return w.*field;
}

// Branch helper: route through LinearInt8 when the model is quantized, else
// FP32 Linear — OR, if an Apple-AMX fp16 BNNSGraph context is loaded for this
// Linear AND ESM_APPLE_AMX=on, short-circuit through it. The AMX artifact has
// bias baked in, so we don't add it again on that path.
inline void LinearProj(
    const Config& cfg, const float* A, const float* W_fp32,
    const esm::quant::QuantizedTensor& W_int8, const float* bias, float* C,
    int M, int N, int K, AppleAmxContext* amx = nullptr,
    const std::vector<std::unique_ptr<AppleAneContext>>* ane = nullptr) {
#ifdef ESM_APPLE_ANE_AVAILABLE
  if (ane != nullptr && !ane->empty() && esm::ArmUseAppleAne()) {
    Status s = DispatchAne(*ane, A, C, M, K, N);
    if (s.ok()) return;
    // ANE dispatch failed (no matching bucket / runtime error) -> fall through
    // through AMX -> default. Each layer can mix.
  }
#else
  (void)ane;
#endif
#ifdef ESM_APPLE_AMX_AVAILABLE
  if (amx != nullptr && esm::ArmUseAppleAmx()) {
    Status s = amx->Execute(A, C, M);
    if (s.ok()) return;
  }
#else
  (void)amx;
#endif
  if (cfg.weights_quantized) {
    kernels::LinearInt8(A, W_int8, bias, C, M, N, K);
  } else {
    kernels::Linear(A, W_fp32, bias, C, M, N, K);
  }
}

// Phase 2 Slice 5 sensitivity escape: round an FP32 buffer to IEEE 754
// binary16 (FP16) precision and back. The activation keeps FP32 storage;
// we're simulating "as if the input arrived from an FP16-quantized site"
// so the downstream INT8 Linear sees lower-precision input (the cheapest
// outlier mitigation on layer 0).
inline void RoundToFp16Inplace(float* x, std::size_t n) {
#if defined(__aarch64__) || defined(_M_ARM64)
  // ARM: __fp16 is the C ABI IEEE 754 binary16 type, default-enabled.
  for (std::size_t i = 0; i < n; ++i) {
    __fp16 h = static_cast<__fp16>(x[i]);
    x[i] = static_cast<float>(h);
  }
#elif (defined(__x86_64__) || defined(_M_X64)) && defined(__F16C__)
  // x86 with F16C (part of -march=x86-64-v3): VCVTPS2PH + VCVTPH2PS.
  for (std::size_t i = 0; i < n; ++i) {
    unsigned short h =
        _cvtss_sh(x[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x[i] = _cvtsh_ss(h);
  }
#else
  (void)x;
  (void)n;  // No native FP16 path -> the escape is a silent no-op.
#endif
}

void TransformerBlock(const Config& cfg, const LayerWeights& w, float* hidden,
                      // scratch (all pulled from the per-Model arena)
                      float* scratch_ln, float* scratch_qkv_flat,
                      float* scratch_cos, float* scratch_sin,
                      float* scratch_attn_out, float* scratch_attn_proj,
                      float* scratch_inter, float* scratch_inter_gelu,
                      float* scratch_ffn_out, const int* cu_seqlens,
                      int batch_size, int L, int max_seqlen, int layer_index,
                      ActivationObserver* observer) {
  const int d = cfg.hidden_size;
  const int H = cfg.num_attention_heads;
  const int dh = cfg.head_dim;
  const int ffn = cfg.intermediate_size;

  // Pre-attention LayerNorm on `hidden` -> scratch_ln
  {
    esm::profile::ScopedTimer t("attn_ln");
    kernels::LayerNorm(hidden, w.attn_ln_w.data(), w.attn_ln_b.data(),
                       cfg.layer_norm_eps, scratch_ln, L, d);
  }
  if (observer) {
    observer->Observe("layer" + std::to_string(layer_index) + ".attn_ln_output",
                      scratch_ln, static_cast<std::size_t>(L) * d);
  }

  // Q, K, V projections write [L, d] = [L, H*dh] = [L, H, dh] directly.
  // Reuse scratch_qkv_flat in three chunks: q | k | v.
  float* q_packed = scratch_qkv_flat;
  float* k_packed = scratch_qkv_flat + static_cast<long>(L) * d;
  float* v_packed = scratch_qkv_flat + 2L * L * d;
  {
    esm::profile::ScopedTimer t("qkv_proj");
    LinearProj(cfg, scratch_ln, w.q_w.data(), w.q_w_int8, w.q_b.data(),
               q_packed, L, d, d, AmxOf(w, &LayerWeights::amx_q),
               &AneOf(w, &LayerWeights::ane_q));
    LinearProj(cfg, scratch_ln, w.k_w.data(), w.k_w_int8, w.k_b.data(),
               k_packed, L, d, d, AmxOf(w, &LayerWeights::amx_k),
               &AneOf(w, &LayerWeights::ane_k));
    LinearProj(cfg, scratch_ln, w.v_w.data(), w.v_w_int8, w.v_b.data(),
               v_packed, L, d, d, AmxOf(w, &LayerWeights::amx_v),
               &AneOf(w, &LayerWeights::ane_v));
  }

  // Q-scale BEFORE RoPE — ESM's load-bearing quirk; see CLAUDE.md.
  {
    esm::profile::ScopedTimer t("q_scale_rope");
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(dh));
    kernels::ScaleInplace(q_packed, static_cast<std::size_t>(L) * d, q_scale);

    // RoPE positions reset per sequence — table only needs max_seqlen rows
    // even when packed T = sum(L_b) is much larger.
    kernels::RopeBuildTables(max_seqlen, dh, scratch_cos, scratch_sin);
    kernels::RopeApplyVarlen(q_packed, scratch_cos, scratch_sin, cu_seqlens,
                              batch_size, H, dh);
    kernels::RopeApplyVarlen(k_packed, scratch_cos, scratch_sin, cu_seqlens,
                              batch_size, H, dh);
  }

  // Self-attention. Output is [L, H*dh] = [L, d] (heads concatenated).
  {
    esm::profile::ScopedTimer t("attention");
    kernels::AttentionVarlen(q_packed, k_packed, v_packed, cu_seqlens, batch_size,
                             H, dh, scratch_attn_out);
  }
  if (observer) {
    observer->Observe("layer" + std::to_string(layer_index) + ".attn_out",
                      scratch_attn_out, static_cast<std::size_t>(L) * d);
  }

  // out_proj
  {
    esm::profile::ScopedTimer t("attn_out_proj");
    LinearProj(cfg, scratch_attn_out, w.out_w.data(), w.out_w_int8,
               w.out_b.data(), scratch_attn_proj, L, d, d,
               AmxOf(w, &LayerWeights::amx_out),
               &AneOf(w, &LayerWeights::ane_out));
  }

  // Residual: hidden += attn_proj
  {
    esm::profile::ScopedTimer t("residual");
    kernels::ResidualAddInplace(hidden, scratch_attn_proj,
                                 static_cast<std::size_t>(L) * d);
  }

  // Pre-FFN LayerNorm on `hidden` -> scratch_ln
  {
    esm::profile::ScopedTimer t("ffn_ln");
    kernels::LayerNorm(hidden, w.ffn_ln_w.data(), w.ffn_ln_b.data(),
                       cfg.layer_norm_eps, scratch_ln, L, d);
  }
  if (observer) {
    observer->Observe("layer" + std::to_string(layer_index) + ".ffn_ln_output",
                      scratch_ln, static_cast<std::size_t>(L) * d);
  }

  // Slice 5 escape: on layer 0 only, optionally round the fc1 input to
  // FP16 precision. Soaks up activation outliers cheaply (10-bit mantissa
  // instead of 23) without changing the INT8 weight path.
  if (cfg.first_block_fc1_fp16 && layer_index == 0) {
    RoundToFp16Inplace(scratch_ln, static_cast<std::size_t>(L) * d);
  }

  // fc1: [L, d] -> [L, 4d]
  {
    esm::profile::ScopedTimer t("fc1");
    LinearProj(cfg, scratch_ln, w.fc1_w.data(), w.fc1_w_int8, w.fc1_b.data(),
               scratch_inter, L, ffn, d, AmxOf(w, &LayerWeights::amx_fc1),
               &AneOf(w, &LayerWeights::ane_fc1));
  }
  {
    esm::profile::ScopedTimer t("gelu");
    kernels::Gelu(scratch_inter, scratch_inter_gelu,
                  static_cast<std::size_t>(L) * ffn);
  }
  if (observer) {
    observer->Observe("layer" + std::to_string(layer_index) + ".inter_gelu",
                      scratch_inter_gelu, static_cast<std::size_t>(L) * ffn);
  }
  // fc2: [L, 4d] -> [L, d]
  {
    esm::profile::ScopedTimer t("fc2");
    LinearProj(cfg, scratch_inter_gelu, w.fc2_w.data(), w.fc2_w_int8,
               w.fc2_b.data(), scratch_ffn_out, L, d, ffn,
               AmxOf(w, &LayerWeights::amx_fc2),
               &AneOf(w, &LayerWeights::ane_fc2));
  }

  // Residual: hidden += ffn_out
  {
    esm::profile::ScopedTimer t("residual");
    kernels::ResidualAddInplace(hidden, scratch_ffn_out,
                                 static_cast<std::size_t>(L) * d);
  }
}

}  // namespace

std::vector<float> Model::Forward(std::span<const int32_t> input_ids,
                                  std::span<const int32_t> attention_mask) const {
  return ForwardWithHiddenStates(input_ids, attention_mask, nullptr);
}

std::vector<float> Model::ForwardWithHiddenStates(
    std::span<const int32_t> input_ids,
    std::span<const int32_t> attention_mask,
    std::vector<std::vector<float>>* hidden_states_out) const {
  std::vector<float> logits;
  ForwardInto(input_ids, attention_mask, ws_, &logits, hidden_states_out);
  return logits;
}

std::vector<float> Model::ForwardWithObserver(
    std::span<const std::int32_t> input_ids,
    std::span<const std::int32_t> attention_mask,
    ActivationObserver* observer) const {
  std::vector<float> logits;
  ForwardInto(input_ids, attention_mask, ws_, &logits, nullptr, observer);
  return logits;
}

void Model::ForwardPackedInto(
    const BatchView& batch, Workspace& ws,
    std::vector<std::vector<float>>* logits_per_seq_out,
    std::vector<std::vector<float>>* hidden_packed_out,
    ActivationObserver* observer) const {
  const int T = static_cast<int>(batch.total_tokens());
  const int B = batch.batch_size;
  const int d = cfg_.hidden_size;
  const int H = cfg_.num_attention_heads;
  const int dh = cfg_.head_dim;
  const int ffn = cfg_.intermediate_size;
  const int V = cfg_.vocab_size;
  if (T == 0) {
    if (logits_per_seq_out) logits_per_seq_out->clear();
    return;
  }

  // RAII activation: rewinds the arena cursor at entry and flags the
  // workspace in-use for the duration of the forward.
  auto ws_guard = ws.activate();

  const std::size_t T_sz = static_cast<std::size_t>(T);
  const std::size_t Td = T_sz * static_cast<std::size_t>(d);
  const std::size_t T3d = Td * 3;
  const std::size_t Tffn = T_sz * static_cast<std::size_t>(ffn);
  const std::size_t TV = T_sz * static_cast<std::size_t>(V);
  (void)H;

  // Max sequence length governs the RoPE table size; varlen RoPE indexes
  // positions [0, L_b) into the shared cos/sin tables per sequence.
  int max_seqlen = 0;
  for (int b = 0; b < B; ++b) {
    max_seqlen = std::max(max_seqlen, batch.sequence_length(b));
  }
  const std::size_t Mdh = static_cast<std::size_t>(max_seqlen) *
                          static_cast<std::size_t>(dh);

  // Reserve the worst-case workspace size BEFORE the first allocate.
  // Growing the arena mid-forward would reallocate the backing buffer
  // and invalidate every pointer we'd already handed out (Workspace::Grow
  // asserts against that path). Counts: hidden + scratch_ln + 3*qkv +
  //   2*cos/sin + attn_out + attn_proj + 2*inter/inter_gelu + ffn_out +
  //   final_ln + lm_dense + lm_gelu + lm_ln + packed_logits
  // = 15 buffers totalling 20*Td + 2*Mdh + T*V floats. One cache line of
  // slack per allocation covers Workspace::AlignUp padding.
  const std::size_t scratch_floats = 20 * Td + 2 * Mdh + TV;
  const std::size_t alignment_slack = 15 * 64;
  ws.reserve(scratch_floats * sizeof(float) + alignment_slack);

  float* hidden = ws.allocate<float>(Td);
  {
    esm::profile::ScopedTimer t("embed");
    Embed(cfg_, embed_.data(), batch.packed_ids, batch.packed_masks,
          batch.cu_seqlens.data(), B, hidden);
  }

  if (hidden_packed_out) {
    hidden_packed_out->clear();
    hidden_packed_out->reserve(
        static_cast<std::size_t>(cfg_.num_hidden_layers + 1));
    hidden_packed_out->emplace_back(hidden, hidden + Td);
  }

  float* scratch_ln = ws.allocate<float>(Td);
  float* scratch_qkv_flat = ws.allocate<float>(T3d);
  float* scratch_cos = ws.allocate<float>(Mdh);
  float* scratch_sin = ws.allocate<float>(Mdh);
  float* scratch_attn_out = ws.allocate<float>(Td);
  float* scratch_attn_proj = ws.allocate<float>(Td);
  float* scratch_inter = ws.allocate<float>(Tffn);
  float* scratch_inter_gelu = ws.allocate<float>(Tffn);
  float* scratch_ffn_out = ws.allocate<float>(Td);

  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    TransformerBlock(cfg_, layers_[static_cast<std::size_t>(i)], hidden,
                     scratch_ln, scratch_qkv_flat, scratch_cos, scratch_sin,
                     scratch_attn_out, scratch_attn_proj, scratch_inter,
                     scratch_inter_gelu, scratch_ffn_out,
                     batch.cu_seqlens.data(),
                     /*batch_size=*/B, T, max_seqlen,
                     /*layer_index=*/i, observer);
    if (hidden_packed_out) {
      hidden_packed_out->emplace_back(hidden, hidden + Td);
    }
  }

  // Final encoder LayerNorm (= HF emb_layer_norm_after); produces the
  // hidden state that lm_head consumes.
  float* final_ln = ws.allocate<float>(Td);
  {
    esm::profile::ScopedTimer t("final_ln");
    kernels::LayerNorm(hidden, final_ln_w_.data(), final_ln_b_.data(),
                       cfg_.layer_norm_eps, final_ln, T, d);
  }
  if (hidden_packed_out) {
    hidden_packed_out->back().assign(final_ln, final_ln + Td);
  }

  // lm_head: dense -> gelu -> layer_norm -> tied decoder + bias.
  float* lm_dense = ws.allocate<float>(Td);
  {
    esm::profile::ScopedTimer t("lm_dense");
    bool used_fast = false;
#ifdef ESM_APPLE_ANE_AVAILABLE
    if (!ane_lm_dense_.empty() && esm::ArmUseAppleAne()) {
      Status s = DispatchAne(ane_lm_dense_, final_ln, lm_dense, T, d, d);
      used_fast = s.ok();
    }
#endif
#ifdef ESM_APPLE_AMX_AVAILABLE
    if (!used_fast && amx_lm_dense_ && esm::ArmUseAppleAmx()) {
      Status s = amx_lm_dense_->Execute(final_ln, lm_dense, T);
      used_fast = s.ok();
    }
#endif
    if (!used_fast) {
      if (cfg_.lm_head_dense_quantized) {
        kernels::LinearInt8(final_ln, lm_dense_w_int8_, lm_dense_b_.data(),
                            lm_dense, T, d, d);
      } else {
        kernels::Linear(final_ln, lm_dense_w_.data(), lm_dense_b_.data(),
                        lm_dense, T, d, d);
      }
    }
  }
  float* lm_gelu = ws.allocate<float>(Td);
  {
    esm::profile::ScopedTimer t("lm_gelu");
    kernels::Gelu(lm_dense, lm_gelu, Td);
  }
  float* lm_ln = ws.allocate<float>(Td);
  {
    esm::profile::ScopedTimer t("lm_ln");
    kernels::LayerNorm(lm_gelu, lm_ln_w_.data(), lm_ln_b_.data(),
                       cfg_.layer_norm_eps, lm_ln, T, d);
  }

  // Tied decoder: packed [T, V] logits then split per-sequence. embed_
  // has shape [V, d] which is exactly the layout Linear expects for
  // W [N=V, K=d]. The packed logits live in the arena; per-sequence
  // outputs are the caller-visible boundary allocs.
  float* packed_logits = ws.allocate<float>(TV);
  {
    esm::profile::ScopedTimer t("lm_decoder");
    kernels::Linear(lm_ln, embed_.data(), lm_decoder_bias_.data(),
                    packed_logits, T, V, d);
  }
  if (logits_per_seq_out) {
    logits_per_seq_out->resize(static_cast<std::size_t>(B));
    for (int b = 0; b < B; ++b) {
      const int start = batch.cu_seqlens[static_cast<std::size_t>(b)];
      const int L_b = batch.sequence_length(b);
      auto& dst = (*logits_per_seq_out)[static_cast<std::size_t>(b)];
      dst.resize(static_cast<std::size_t>(L_b) *
                 static_cast<std::size_t>(V));
      std::memcpy(dst.data(), packed_logits + static_cast<long>(start) * V,
                  static_cast<std::size_t>(L_b) *
                      static_cast<std::size_t>(V) * sizeof(float));
    }
  }

  esm::profile::DumpAndReset();
}

void Model::ForwardInto(
    std::span<const std::int32_t> input_ids,
    std::span<const std::int32_t> attention_mask, Workspace& ws,
    std::vector<float>* logits_out,
    std::vector<std::vector<float>>* hidden_states_out,
    ActivationObserver* observer) const {
  const int L = static_cast<int>(input_ids.size());
  if (L == 0) {
    if (logits_out) logits_out->clear();
    if (hidden_states_out) hidden_states_out->clear();
    return;
  }
  if (!attention_mask.empty() &&
      attention_mask.size() != input_ids.size()) {
    throw std::runtime_error("attention_mask length mismatch");
  }

  // Build a B=1 BatchView over the caller's buffers. cu_seqlens is a
  // pair of locals; BatchView captures it by span so it must outlive the
  // call.
  const std::int32_t cu_arr[2] = {0, L};
  std::span<const std::int32_t> cu_span(cu_arr, 2);
  BatchView view(input_ids, attention_mask, cu_span, /*batch=*/1);

  std::vector<std::vector<float>> per_seq;
  ForwardPackedInto(view, ws, logits_out ? &per_seq : nullptr,
                    hidden_states_out, observer);
  if (logits_out && !per_seq.empty()) {
    *logits_out = std::move(per_seq[0]);
  } else if (logits_out) {
    logits_out->clear();
  }
}

std::vector<std::vector<float>> Model::ForwardPacked(
    const BatchView& batch) const {
  std::vector<std::vector<float>> outputs;
  ForwardPackedInto(batch, ws_, &outputs, nullptr, nullptr);
  return outputs;
}

std::vector<std::vector<float>> Model::ForwardScheduled(
    const std::vector<std::vector<std::int32_t>>& input_ids,
    const std::vector<std::vector<std::int32_t>>& attention_masks,
    const SchedulerConfig& cfg) const {
  const std::size_t N = input_ids.size();
  if (N == 0) return {};
  if (!attention_masks.empty() && attention_masks.size() != N) {
    throw std::runtime_error("attention_masks/input_ids batch size mismatch");
  }

  // Phase 13: whole-graph CoreML fast path. Activates when every input
  // sequence has the same length L AND we have a registered (N, L)
  // artifact AND the engine isn't told to opt out.
  // Phase 14: env semantics flipped to default-on — set
  // ESM_APPLE_ANE_GRAPH=off / 0 / false to disable. The whole_graph_
  // emptiness check above means a user with no artifacts gets exactly
  // the previous scheduled path; the flip only matters once an artifact
  // is registered (via auto-load or explicit LoadWholeGraphArtifact).
  auto wg_opted_out = []() {
    const char* e = std::getenv("ESM_APPLE_ANE_GRAPH");
    if (!e || *e == '\0') return false;
    const std::string_view s(e);
    return s == "off" || s == "0" || s == "false";
  };
  if (!whole_graph_.empty() && !wg_opted_out()) {
    bool uniform = true;
    const int L = static_cast<int>(input_ids[0].size());
    for (const auto& ids : input_ids) {
      if (static_cast<int>(ids.size()) != L) { uniform = false; break; }
    }
    if (uniform) {
      const int B = static_cast<int>(N);
      // Look up a matching shape registration first; only pack if found.
      bool have_shape = false;
      for (const auto& reg : whole_graph_) {
        if (reg.B == B && reg.L == L) { have_shape = true; break; }
      }
      if (have_shape) {
        std::vector<std::int32_t> ids_packed(static_cast<std::size_t>(B) * L);
        std::vector<std::int32_t> mask_packed(static_cast<std::size_t>(B) * L);
        for (int b = 0; b < B; ++b) {
          const auto& ids_b = input_ids[static_cast<std::size_t>(b)];
          const std::size_t off = static_cast<std::size_t>(b) * L;
          std::memcpy(ids_packed.data() + off, ids_b.data(),
                      static_cast<std::size_t>(L) * sizeof(std::int32_t));
          if (!attention_masks.empty() &&
              !attention_masks[static_cast<std::size_t>(b)].empty()) {
            const auto& m = attention_masks[static_cast<std::size_t>(b)];
            std::memcpy(mask_packed.data() + off, m.data(),
                        static_cast<std::size_t>(L) * sizeof(std::int32_t));
          } else {
            for (int t = 0; t < L; ++t) mask_packed[off + t] = 1;
          }
        }
        bool ok = false;
        auto logits_flat = ForwardWholeGraph(
            ids_packed, mask_packed, B, L, &ok);
        if (ok) {
          std::vector<std::vector<float>> outputs(N);
          const std::size_t per_seq = static_cast<std::size_t>(L) * cfg_.vocab_size;
          for (int b = 0; b < B; ++b) {
            outputs[static_cast<std::size_t>(b)].assign(
                logits_flat.begin() + static_cast<std::ptrdiff_t>(b * per_seq),
                logits_flat.begin() + static_cast<std::ptrdiff_t>((b + 1) * per_seq));
          }
          return outputs;
        }
        // Fall through on Execute failure (logged inside the bridge).
      }
    }
  }

  std::vector<int> lengths;
  lengths.reserve(N);
  for (const auto& ids : input_ids) {
    lengths.push_back(static_cast<int>(ids.size()));
  }
  auto plan = PlanBatches(lengths, cfg);

  std::vector<std::vector<float>> outputs(N);
  for (const auto& group : plan) {
    if (group.empty()) continue;
    // Pack the chosen sequences back-to-back. cu_seqlens drives per-
    // sequence isolation inside ForwardPacked.
    std::vector<std::int32_t> packed_ids;
    std::vector<std::int32_t> packed_masks;
    std::vector<std::int32_t> cu(group.size() + 1, 0);
    std::size_t total = 0;
    bool any_masks = false;
    for (int idx : group) {
      total += input_ids[static_cast<std::size_t>(idx)].size();
      if (!attention_masks.empty() &&
          !attention_masks[static_cast<std::size_t>(idx)].empty()) {
        any_masks = true;
      }
    }
    packed_ids.reserve(total);
    if (any_masks) packed_masks.reserve(total);
    for (std::size_t gi = 0; gi < group.size(); ++gi) {
      const int idx = group[gi];
      const auto& ids = input_ids[static_cast<std::size_t>(idx)];
      packed_ids.insert(packed_ids.end(), ids.begin(), ids.end());
      if (any_masks) {
        if (!attention_masks.empty() &&
            !attention_masks[static_cast<std::size_t>(idx)].empty()) {
          const auto& m =
              attention_masks[static_cast<std::size_t>(idx)];
          packed_masks.insert(packed_masks.end(), m.begin(), m.end());
        } else {
          packed_masks.insert(packed_masks.end(), ids.size(), 1);
        }
      }
      cu[gi + 1] = cu[gi] + static_cast<std::int32_t>(ids.size());
    }
    BatchView view(packed_ids, packed_masks, cu,
                    static_cast<int>(group.size()));
    auto group_out = ForwardPacked(view);
    // Place each per-sequence logits back in caller order.
    for (std::size_t gi = 0; gi < group.size(); ++gi) {
      outputs[static_cast<std::size_t>(group[gi])] = std::move(group_out[gi]);
    }
  }
  return outputs;
}

namespace {
// Thread-local Workspace owned by each pool worker. Re-used across
// every ForwardInto call from the same worker; sized on first call.
thread_local Workspace tls_batch_workspace;
}  // namespace

std::vector<std::vector<float>> Model::ForwardBatch(
    const std::vector<std::vector<std::int32_t>>& input_ids,
    const std::vector<std::vector<std::int32_t>>& attention_masks) const {
  const int B = static_cast<int>(input_ids.size());
  if (B == 0) return {};
  if (!attention_masks.empty() && attention_masks.size() != input_ids.size()) {
    throw std::runtime_error("attention_masks/input_ids batch size mismatch");
  }
  std::vector<std::vector<float>> outputs(static_cast<std::size_t>(B));
  GlobalPool().parallel_for(0, B, /*grain=*/1, [&](int begin, int end) {
    for (int b = begin; b < end; ++b) {
      std::span<const std::int32_t> ids(input_ids[b]);
      std::span<const std::int32_t> mask =
          attention_masks.empty()
              ? std::span<const std::int32_t>{}
              : std::span<const std::int32_t>(attention_masks[b]);
      ForwardInto(ids, mask, tls_batch_workspace, &outputs[b], nullptr);
    }
  });
  return outputs;
}

std::size_t Model::num_threads() { return GlobalPool().size(); }

namespace {

void QuantizeLinear(const std::vector<float>& w_fp32, int out_features,
                    int in_features, esm::quant::QuantizedTensor* out) {
  esm::quant::Quantize(w_fp32.data(), out_features, in_features, out);
}

}  // namespace

void Model::ApplySmoothQuant(
    const std::unordered_map<std::string, float>& act_stats, float alpha) {
  for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
    esm::quant::MigrateSmoothQuant(layers_[static_cast<std::size_t>(i)],
                                    act_stats, i, alpha);
  }
}

void Model::QuantizeWeights() {
  const int d = cfg_.hidden_size;
  const int ffn = cfg_.intermediate_size;
  for (auto& w : layers_) {
    QuantizeLinear(w.q_w, d, d, &w.q_w_int8);
    QuantizeLinear(w.k_w, d, d, &w.k_w_int8);
    QuantizeLinear(w.v_w, d, d, &w.v_w_int8);
    QuantizeLinear(w.out_w, d, d, &w.out_w_int8);
    QuantizeLinear(w.fc1_w, ffn, d, &w.fc1_w_int8);
    QuantizeLinear(w.fc2_w, d, ffn, &w.fc2_w_int8);
  }
  // Phase 7 Slice 8: optionally quantize lm_head.dense to INT8 (opt-in
  // via ESM_QUANTIZE_LM_HEAD=on). Default keeps FP32 — the original
  // Phase 2 Slice 5 escape list rationale stands until PPPL says
  // otherwise. lm_head.layer_norm and the tied decoder stay FP32
  // regardless: LN params don't quantize, and the decoder shares
  // embed_ which we don't touch.
  if (const char* env = std::getenv("ESM_QUANTIZE_LM_HEAD");
      env && *env && std::string(env) != "0" &&
      std::string(env) != "off") {
    QuantizeLinear(lm_dense_w_, d, d, &lm_dense_w_int8_);
    cfg_.lm_head_dense_quantized = true;
  }
  cfg_.weights_quantized = true;
}

}  // namespace esm
