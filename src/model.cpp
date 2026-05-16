#include "esm_cpp/model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__F16C__)
#include <immintrin.h>
#endif

#include "esm_cpp/batch.h"
#include "esm_cpp/cpu_features.h"
#include "esm_cpp/io.h"
#include "esm_cpp/kernels.h"
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

  return out;
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
  kernels::LayerNorm(hidden, w.attn_ln_w.data(), w.attn_ln_b.data(),
                     cfg.layer_norm_eps, scratch_ln, L, d);
  if (observer) {
    observer->Observe("layer" + std::to_string(layer_index) + ".attn_ln_output",
                      scratch_ln, static_cast<std::size_t>(L) * d);
  }

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

  // RoPE positions reset per sequence — table only needs max_seqlen rows
  // even when packed T = sum(L_b) is much larger.
  kernels::RopeBuildTables(max_seqlen, dh, scratch_cos, scratch_sin);
  kernels::RopeApplyVarlenRef(q_packed, scratch_cos, scratch_sin, cu_seqlens,
                              batch_size, H, dh);
  kernels::RopeApplyVarlenRef(k_packed, scratch_cos, scratch_sin, cu_seqlens,
                              batch_size, H, dh);

  // Self-attention. Output is [L, H*dh] = [L, d] (heads concatenated).
  kernels::AttentionVarlen(q_packed, k_packed, v_packed, cu_seqlens, batch_size,
                           H, dh, scratch_attn_out);
  if (observer) {
    observer->Observe("layer" + std::to_string(layer_index) + ".attn_out",
                      scratch_attn_out, static_cast<std::size_t>(L) * d);
  }

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
  LinearProj(cfg, scratch_ln, w.fc1_w.data(), w.fc1_w_int8, w.fc1_b.data(),
             scratch_inter, L, ffn, d);
  // GELU
  kernels::Gelu(scratch_inter, scratch_inter_gelu,
                static_cast<std::size_t>(L) * ffn);
  if (observer) {
    observer->Observe("layer" + std::to_string(layer_index) + ".inter_gelu",
                      scratch_inter_gelu, static_cast<std::size_t>(L) * ffn);
  }
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
  Embed(cfg_, embed_.data(), batch.packed_ids, batch.packed_masks,
        batch.cu_seqlens.data(), B, hidden);

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
  kernels::LayerNorm(hidden, final_ln_w_.data(), final_ln_b_.data(),
                     cfg_.layer_norm_eps, final_ln, T, d);
  if (hidden_packed_out) {
    hidden_packed_out->back().assign(final_ln, final_ln + Td);
  }

  // lm_head: dense -> gelu -> layer_norm -> tied decoder + bias.
  float* lm_dense = ws.allocate<float>(Td);
  kernels::Linear(final_ln, lm_dense_w_.data(), lm_dense_b_.data(), lm_dense,
                  T, d, d);
  float* lm_gelu = ws.allocate<float>(Td);
  kernels::Gelu(lm_dense, lm_gelu, Td);
  float* lm_ln = ws.allocate<float>(Td);
  kernels::LayerNorm(lm_gelu, lm_ln_w_.data(), lm_ln_b_.data(),
                     cfg_.layer_norm_eps, lm_ln, T, d);

  // Tied decoder: packed [T, V] logits then split per-sequence. embed_
  // has shape [V, d] which is exactly the layout Linear expects for
  // W [N=V, K=d]. The packed logits live in the arena; per-sequence
  // outputs are the caller-visible boundary allocs.
  float* packed_logits = ws.allocate<float>(TV);
  kernels::Linear(lm_ln, embed_.data(), lm_decoder_bias_.data(),
                  packed_logits, T, V, d);
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
  // lm_head.dense / lm_head.layer_norm stay FP32 (Slice 5 escape list).
  // The tied lm_head decoder uses embed_, which also stays FP32.
  cfg_.weights_quantized = true;
}

}  // namespace esm
