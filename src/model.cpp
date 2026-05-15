#include "esm_cpp/model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "esm_cpp/cpu_features.h"
#include "esm_cpp/io.h"
#include "esm_cpp/kernels.h"
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

  return out;
}

namespace {

// Embed: gather rows from [vocab_size, d] and apply the inference-time
// token_dropout rescale. ESM zeroes mask-token embeddings and rescales by
// (1 - 0.15*0.8) / (1 - observed_mask_fraction) = 0.88 / (1 - omf).
void Embed(const Config& cfg, const float* embed_w,
           std::span<const int32_t> ids, std::span<const int32_t> mask,
           float* out) {
  const int L = static_cast<int>(ids.size());
  const int d = cfg.hidden_size;
  int src_len = 0;
  int observed_masks = 0;
  for (int t = 0; t < L; ++t) {
    if (mask.empty() || mask[static_cast<std::size_t>(t)] != 0) ++src_len;
    if (ids[static_cast<std::size_t>(t)] == cfg.mask_token_id) ++observed_masks;
  }
  const float mask_ratio_observed =
      (src_len > 0)
          ? static_cast<float>(observed_masks) / static_cast<float>(src_len)
          : 0.0f;
  // (1 - 0.12) / (1 - obs). When obs == 0 this is 0.88 — the well-known
  // "ESM scales everything by 0.88 at inference" gotcha.
  const float scale = (1.0f - kTokenDropoutMaskRatioTrain) /
                      std::max(1.0f - mask_ratio_observed, 1e-12f);

  for (int t = 0; t < L; ++t) {
    int32_t id = ids[static_cast<std::size_t>(t)];
    const float* row = embed_w + static_cast<long>(id) * d;
    float* out_row = out + static_cast<long>(t) * d;
    bool zero_row = (id == cfg.mask_token_id);
    bool is_pad = !mask.empty() && mask[static_cast<std::size_t>(t)] == 0;
    if (zero_row) {
      for (int i = 0; i < d; ++i) out_row[i] = 0.0f;
    } else {
      for (int i = 0; i < d; ++i) out_row[i] = row[i] * scale;
    }
    if (is_pad) {
      // HF multiplies the post-rescale embedding by attention_mask, which
      // zeroes pad rows entirely.
      for (int i = 0; i < d; ++i) out_row[i] = 0.0f;
    }
  }
}

// Layout helpers: with the cu_seqlens-packed kernels, Q/K/V projections
// land directly in [L, H, head_dim] (= [L, H*head_dim] in memory). The
// old SplitHeads memcpy from [L, H*dh] -> [H, L, dh] is gone.
// Branch helper: route through LinearInt8 when the model is quantized,
// else the FP32 Linear facade. Centralized so the routing logic lives
// in one place instead of being duplicated at every projection site.
inline void LinearProj(const Config& cfg, const float* A, const float* W_fp32,
                       const esm::quant::QuantizedTensor& W_int8,
                       const float* bias, float* C, int M, int N, int K) {
  if (cfg.weights_quantized) {
    kernels::LinearInt8(A, W_int8, bias, C, M, N, K);
  } else {
    kernels::Linear(A, W_fp32, bias, C, M, N, K);
  }
}

void TransformerBlock(const Config& cfg, const LayerWeights& w, float* hidden,
                      // scratch (all pulled from the per-Model arena)
                      float* scratch_ln, float* scratch_qkv_flat,
                      float* scratch_cos, float* scratch_sin,
                      float* scratch_attn_out, float* scratch_attn_proj,
                      float* scratch_inter, float* scratch_inter_gelu,
                      float* scratch_ffn_out, const int* cu_seqlens,
                      int batch_size, int L) {
  const int d = cfg.hidden_size;
  const int H = cfg.num_attention_heads;
  const int dh = cfg.head_dim;
  const int ffn = cfg.intermediate_size;

  // Pre-attention LayerNorm on `hidden` -> scratch_ln
  kernels::LayerNorm(hidden, w.attn_ln_w.data(), w.attn_ln_b.data(),
                     cfg.layer_norm_eps, scratch_ln, L, d);

  // Q, K, V projections write [L, d] = [L, H*dh] = [L, H, dh] directly.
  // Reuse scratch_qkv_flat in three chunks: q | k | v.
  float* q_packed = scratch_qkv_flat;
  float* k_packed = scratch_qkv_flat + static_cast<long>(L) * d;
  float* v_packed = scratch_qkv_flat + 2L * L * d;
  LinearProj(cfg, scratch_ln, w.q_w.data(), w.q_w_int8, w.q_b.data(),
             q_packed, L, d, d);
  LinearProj(cfg, scratch_ln, w.k_w.data(), w.k_w_int8, w.k_b.data(),
             k_packed, L, d, d);
  LinearProj(cfg, scratch_ln, w.v_w.data(), w.v_w_int8, w.v_b.data(),
             v_packed, L, d, d);

  // Q-scale BEFORE RoPE — ESM's load-bearing quirk; see CLAUDE.md.
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(dh));
  for (long i = 0; i < static_cast<long>(L) * d; ++i) q_packed[i] *= q_scale;

  kernels::RopeBuildTables(L, dh, scratch_cos, scratch_sin);
  kernels::RopeApplyVarlenRef(q_packed, scratch_cos, scratch_sin, cu_seqlens,
                              batch_size, H, dh);
  kernels::RopeApplyVarlenRef(k_packed, scratch_cos, scratch_sin, cu_seqlens,
                              batch_size, H, dh);

  // Self-attention. Output is [L, H*dh] = [L, d] (heads concatenated).
  kernels::AttentionVarlen(q_packed, k_packed, v_packed, cu_seqlens, batch_size,
                           H, dh, scratch_attn_out);

  // out_proj
  LinearProj(cfg, scratch_attn_out, w.out_w.data(), w.out_w_int8,
             w.out_b.data(), scratch_attn_proj, L, d, d);

  // Residual: hidden += attn_proj
  for (long i = 0; i < static_cast<long>(L) * d; ++i) {
    hidden[i] += scratch_attn_proj[i];
  }

  // Pre-FFN LayerNorm on `hidden` -> scratch_ln
  kernels::LayerNorm(hidden, w.ffn_ln_w.data(), w.ffn_ln_b.data(),
                     cfg.layer_norm_eps, scratch_ln, L, d);

  // fc1: [L, d] -> [L, 4d]
  LinearProj(cfg, scratch_ln, w.fc1_w.data(), w.fc1_w_int8, w.fc1_b.data(),
             scratch_inter, L, ffn, d);
  // GELU
  kernels::Gelu(scratch_inter, scratch_inter_gelu,
                static_cast<std::size_t>(L) * ffn);
  // fc2: [L, 4d] -> [L, d]
  LinearProj(cfg, scratch_inter_gelu, w.fc2_w.data(), w.fc2_w_int8,
             w.fc2_b.data(), scratch_ffn_out, L, d, ffn);

  // Residual: hidden += ffn_out
  for (long i = 0; i < static_cast<long>(L) * d; ++i) {
    hidden[i] += scratch_ffn_out[i];
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

void Model::ForwardInto(
    std::span<const std::int32_t> input_ids,
    std::span<const std::int32_t> attention_mask, Workspace& ws,
    std::vector<float>* logits_out,
    std::vector<std::vector<float>>* hidden_states_out) const {
  const int L = static_cast<int>(input_ids.size());
  const int d = cfg_.hidden_size;
  const int H = cfg_.num_attention_heads;
  const int dh = cfg_.head_dim;
  const int ffn = cfg_.intermediate_size;
  const int V = cfg_.vocab_size;
  if (L == 0) {
    if (logits_out) logits_out->clear();
    return;
  }

  if (!attention_mask.empty() &&
      attention_mask.size() != input_ids.size()) {
    throw std::runtime_error("attention_mask length mismatch");
  }

  // RAII activation: rewinds the arena cursor at entry and flags the
  // workspace in-use for the duration of the forward.
  auto ws_guard = ws.activate();

  const std::size_t L_sz = static_cast<std::size_t>(L);
  const std::size_t Ld = L_sz * static_cast<std::size_t>(d);
  const std::size_t L3d = Ld * 3;
  const std::size_t Ldh = L_sz * static_cast<std::size_t>(dh);
  const std::size_t Lffn = L_sz * static_cast<std::size_t>(ffn);
  const std::size_t LV = L_sz * static_cast<std::size_t>(V);
  (void)H;
  (void)LV;

  // Reserve the worst-case workspace size BEFORE the first allocate. Growing
  // the arena mid-forward would reallocate the backing buffer and invalidate
  // every pointer we'd already handed out — Workspace::Grow asserts against
  // that path, but only when we've correctly pre-sized here.
  // Slice 4: cu_seqlens layout drops the 3 head-major QKV reshape buffers.
  // Counts: hidden + scratch_ln + 3*qkv + 2*cos/sin +
  //         attn_out + attn_proj + 2*inter/inter_gelu + ffn_out +
  //         final_ln + lm_dense + lm_gelu + lm_ln
  //       = 14 buffers totalling 20*Ld + 2*Ldh floats. One cache line of
  //         slack per allocation covers Workspace::AlignUp padding.
  const std::size_t scratch_floats = 20 * Ld + 2 * Ldh;
  const std::size_t alignment_slack = 14 * 64;
  ws.reserve(scratch_floats * sizeof(float) + alignment_slack);

  float* hidden = ws.allocate<float>(Ld);
  Embed(cfg_, embed_.data(), input_ids, attention_mask, hidden);

  if (hidden_states_out) {
    hidden_states_out->clear();
    hidden_states_out->reserve(static_cast<std::size_t>(cfg_.num_hidden_layers + 1));
    hidden_states_out->emplace_back(hidden, hidden + Ld);
  }

  float* scratch_ln = ws.allocate<float>(Ld);
  float* scratch_qkv_flat = ws.allocate<float>(L3d);
  float* scratch_cos = ws.allocate<float>(Ldh);
  float* scratch_sin = ws.allocate<float>(Ldh);
  float* scratch_attn_out = ws.allocate<float>(Ld);
  float* scratch_attn_proj = ws.allocate<float>(Ld);
  float* scratch_inter = ws.allocate<float>(Lffn);
  float* scratch_inter_gelu = ws.allocate<float>(Lffn);
  float* scratch_ffn_out = ws.allocate<float>(Ld);

  // B=1 cu_seqlens for the Phase 1 single-sequence path. Phase 3 scheduler
  // will pack multiple sequences with cu_seqlens = {0, L_0, L_0+L_1, ...}.
  const int cu_seqlens_b1[2] = {0, L};

  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    TransformerBlock(cfg_, layers_[static_cast<std::size_t>(i)], hidden,
                     scratch_ln, scratch_qkv_flat, scratch_cos, scratch_sin,
                     scratch_attn_out, scratch_attn_proj, scratch_inter,
                     scratch_inter_gelu, scratch_ffn_out, cu_seqlens_b1,
                     /*batch_size=*/1, L);
    if (hidden_states_out) {
      hidden_states_out->emplace_back(hidden, hidden + Ld);
    }
  }

  // Final encoder LayerNorm (= HF emb_layer_norm_after); produces the
  // hidden state that lm_head consumes.
  float* final_ln = ws.allocate<float>(Ld);
  kernels::LayerNorm(hidden, final_ln_w_.data(), final_ln_b_.data(),
                     cfg_.layer_norm_eps, final_ln, L, d);
  if (hidden_states_out) {
    // Replace the last entry (which mirrored the pre-LN output) with the
    // post-final-LN tensor to match HF hidden_states[-1] semantics.
    hidden_states_out->back().assign(final_ln, final_ln + Ld);
  }

  // lm_head: dense -> gelu -> layer_norm -> tied decoder + bias.
  float* lm_dense = ws.allocate<float>(Ld);
  kernels::Linear(final_ln, lm_dense_w_.data(), lm_dense_b_.data(), lm_dense,
                  L, d, d);
  float* lm_gelu = ws.allocate<float>(Ld);
  kernels::Gelu(lm_dense, lm_gelu, Ld);
  float* lm_ln = ws.allocate<float>(Ld);
  kernels::LayerNorm(lm_gelu, lm_ln_w_.data(), lm_ln_b_.data(),
                     cfg_.layer_norm_eps, lm_ln, L, d);

  // Tied decoder: logits = lm_ln @ embed^T + lm_head.bias.
  // embed_ has shape [V, d] (out=V, in=d), which is exactly the layout
  // Linear expects for W [N=V, K=d]. The output vector is the caller-visible
  // boundary alloc; everything above is in the arena.
  if (logits_out) {
    logits_out->resize(LV);
    kernels::Linear(lm_ln, embed_.data(), lm_decoder_bias_.data(),
                    logits_out->data(), L, V, d);
  }
}

// Process-global thread pool. Lazy-init on first call (Model::load
// triggers it). Sized from ESM_NUM_THREADS at first construction; later
// env-var changes are not honored.
ThreadPool& GlobalPool() {
  static ThreadPool pool = ThreadPool::FromEnv();
  return pool;
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
  // lm_head.dense / lm_head.layer_norm stay FP32 (Slice 5 escape list).
  // The tied lm_head decoder uses embed_, which also stays FP32.
  cfg_.weights_quantized = true;
}

}  // namespace esm
