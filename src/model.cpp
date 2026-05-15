#include "esm_cpp/model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

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

// Reshape [L, num_heads*head_dim] in linear order into [num_heads, L, head_dim].
void SplitHeads(const float* flat, float* heads, int L, int num_heads,
                int head_dim) {
  for (int t = 0; t < L; ++t) {
    for (int h = 0; h < num_heads; ++h) {
      const float* src = flat + static_cast<long>(t) * num_heads * head_dim +
                         static_cast<long>(h) * head_dim;
      float* dst = heads + (static_cast<long>(h) * L + t) * head_dim;
      std::memcpy(dst, src, sizeof(float) * static_cast<std::size_t>(head_dim));
    }
  }
}

void TransformerBlock(const Config& cfg, const LayerWeights& w,
                      std::span<const int32_t> attention_mask, float* hidden,
                      // scratch
                      float* scratch_ln, float* scratch_qkv_flat,
                      float* scratch_q_heads, float* scratch_k_heads,
                      float* scratch_v_heads, float* scratch_cos,
                      float* scratch_sin, float* scratch_attn_out,
                      float* scratch_attn_proj, float* scratch_inter,
                      float* scratch_inter_gelu, float* scratch_ffn_out,
                      int L) {
  const int d = cfg.hidden_size;
  const int H = cfg.num_attention_heads;
  const int dh = cfg.head_dim;
  const int ffn = cfg.intermediate_size;
  const int* mask_ptr = attention_mask.empty() ? nullptr : attention_mask.data();

  // Pre-attention LayerNorm on `hidden` -> scratch_ln
  kernels::LayerNorm(hidden, w.attn_ln_w.data(), w.attn_ln_b.data(),
                        cfg.layer_norm_eps, scratch_ln, L, d);

  // Q, K, V projections: each is a Linear from [L, d] to [L, d].
  // Reuse scratch_qkv_flat in three chunks back-to-back: q | k | v.
  kernels::Linear(scratch_ln, w.q_w.data(), w.q_b.data(),
                     scratch_qkv_flat, L, d, d);
  kernels::Linear(scratch_ln, w.k_w.data(), w.k_b.data(),
                     scratch_qkv_flat + static_cast<long>(L) * d, L, d, d);
  kernels::Linear(scratch_ln, w.v_w.data(), w.v_b.data(),
                     scratch_qkv_flat + 2L * L * d, L, d, d);

  // Reshape into [H, L, head_dim].
  SplitHeads(scratch_qkv_flat, scratch_q_heads, L, H, dh);
  SplitHeads(scratch_qkv_flat + static_cast<long>(L) * d, scratch_k_heads,
             L, H, dh);
  SplitHeads(scratch_qkv_flat + 2L * L * d, scratch_v_heads, L, H, dh);

  // Q-scale BEFORE RoPE — ESM's load-bearing quirk; see CLAUDE.md.
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(dh));
  for (long i = 0; i < static_cast<long>(H) * L * dh; ++i) {
    scratch_q_heads[i] *= q_scale;
  }

  kernels::RopeBuildTables(L, dh, scratch_cos, scratch_sin);
  kernels::RopeApplyInplace(scratch_q_heads, scratch_cos, scratch_sin, H, L, dh);
  kernels::RopeApplyInplace(scratch_k_heads, scratch_cos, scratch_sin, H, L, dh);

  // Self-attention. Output is [L, H*dh] = [L, d] in heads-concatenated layout.
  kernels::Attention(scratch_q_heads, scratch_k_heads, scratch_v_heads,
                        mask_ptr, scratch_attn_out, H, L, dh);

  // out_proj
  kernels::Linear(scratch_attn_out, w.out_w.data(), w.out_b.data(),
                     scratch_attn_proj, L, d, d);

  // Residual: hidden += attn_proj
  for (long i = 0; i < static_cast<long>(L) * d; ++i) {
    hidden[i] += scratch_attn_proj[i];
  }

  // Pre-FFN LayerNorm on `hidden` -> scratch_ln
  kernels::LayerNorm(hidden, w.ffn_ln_w.data(), w.ffn_ln_b.data(),
                        cfg.layer_norm_eps, scratch_ln, L, d);

  // fc1: [L, d] -> [L, 4d]
  kernels::Linear(scratch_ln, w.fc1_w.data(), w.fc1_b.data(),
                     scratch_inter, L, ffn, d);
  // GELU
  kernels::Gelu(scratch_inter, scratch_inter_gelu,
                   static_cast<std::size_t>(L) * ffn);
  // fc2: [L, 4d] -> [L, d]
  kernels::Linear(scratch_inter_gelu, w.fc2_w.data(), w.fc2_b.data(),
                     scratch_ffn_out, L, d, ffn);

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
  const int L = static_cast<int>(input_ids.size());
  const int d = cfg_.hidden_size;
  const int H = cfg_.num_attention_heads;
  const int dh = cfg_.head_dim;
  const int ffn = cfg_.intermediate_size;
  const int V = cfg_.vocab_size;
  if (L == 0) return {};

  if (!attention_mask.empty() &&
      attention_mask.size() != input_ids.size()) {
    throw std::runtime_error("attention_mask length mismatch");
  }

  std::vector<float> hidden(static_cast<std::size_t>(L) * d);
  Embed(cfg_, embed_.data(), input_ids, attention_mask, hidden.data());

  if (hidden_states_out) {
    hidden_states_out->clear();
    hidden_states_out->reserve(static_cast<std::size_t>(cfg_.num_hidden_layers + 1));
    hidden_states_out->push_back(hidden);
  }

  std::vector<float> scratch_ln(static_cast<std::size_t>(L) * d);
  // QKV flat output buffer holds q | k | v concatenated along the row axis.
  std::vector<float> scratch_qkv_flat(static_cast<std::size_t>(L) * d * 3);
  std::vector<float> scratch_q_heads(static_cast<std::size_t>(H) * L * dh);
  std::vector<float> scratch_k_heads(static_cast<std::size_t>(H) * L * dh);
  std::vector<float> scratch_v_heads(static_cast<std::size_t>(H) * L * dh);
  std::vector<float> scratch_cos(static_cast<std::size_t>(L) * dh);
  std::vector<float> scratch_sin(static_cast<std::size_t>(L) * dh);
  std::vector<float> scratch_attn_out(static_cast<std::size_t>(L) * d);
  std::vector<float> scratch_attn_proj(static_cast<std::size_t>(L) * d);
  std::vector<float> scratch_inter(static_cast<std::size_t>(L) * ffn);
  std::vector<float> scratch_inter_gelu(static_cast<std::size_t>(L) * ffn);
  std::vector<float> scratch_ffn_out(static_cast<std::size_t>(L) * d);

  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    TransformerBlock(cfg_, layers_[static_cast<std::size_t>(i)], attention_mask,
                     hidden.data(), scratch_ln.data(), scratch_qkv_flat.data(),
                     scratch_q_heads.data(), scratch_k_heads.data(),
                     scratch_v_heads.data(), scratch_cos.data(),
                     scratch_sin.data(), scratch_attn_out.data(),
                     scratch_attn_proj.data(), scratch_inter.data(),
                     scratch_inter_gelu.data(), scratch_ffn_out.data(), L);
    if (hidden_states_out) hidden_states_out->push_back(hidden);
  }

  // Final encoder LayerNorm (= HF emb_layer_norm_after); produces the
  // hidden state that lm_head consumes.
  std::vector<float> final_ln(static_cast<std::size_t>(L) * d);
  kernels::LayerNorm(hidden.data(), final_ln_w_.data(), final_ln_b_.data(),
                        cfg_.layer_norm_eps, final_ln.data(), L, d);
  if (hidden_states_out) {
    // Replace the last entry (which mirrored the pre-LN output) with the
    // post-final-LN tensor to match HF hidden_states[-1] semantics.
    hidden_states_out->back() = final_ln;
  }

  // lm_head: dense -> gelu -> layer_norm -> tied decoder + bias.
  std::vector<float> lm_dense(static_cast<std::size_t>(L) * d);
  kernels::Linear(final_ln.data(), lm_dense_w_.data(), lm_dense_b_.data(),
                     lm_dense.data(), L, d, d);
  std::vector<float> lm_gelu(static_cast<std::size_t>(L) * d);
  kernels::Gelu(lm_dense.data(), lm_gelu.data(),
                   static_cast<std::size_t>(L) * d);
  std::vector<float> lm_ln(static_cast<std::size_t>(L) * d);
  kernels::LayerNorm(lm_gelu.data(), lm_ln_w_.data(), lm_ln_b_.data(),
                        cfg_.layer_norm_eps, lm_ln.data(), L, d);

  // Tied decoder: logits = lm_ln @ embed^T + lm_head.bias.
  // embed_ has shape [V, d] (out=V, in=d), which is exactly the layout
  // LinearRef expects for W [N=V, K=d].
  std::vector<float> logits(static_cast<std::size_t>(L) * V);
  kernels::Linear(lm_ln.data(), embed_.data(), lm_decoder_bias_.data(),
                     logits.data(), L, V, d);
  return logits;
}

}  // namespace esm
