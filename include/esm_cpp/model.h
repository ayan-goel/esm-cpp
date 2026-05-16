#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "esm_cpp/batch.h"
#include "esm_cpp/observer.h"
#include "esm_cpp/quant.h"
#include "esm_cpp/thread_pool.h"
#include "esm_cpp/workspace.h"

namespace esm {

struct Config {
  int num_hidden_layers;
  int hidden_size;
  int num_attention_heads;
  int head_dim;
  int intermediate_size;
  int vocab_size;
  float layer_norm_eps;
  bool token_dropout;
  int mask_token_id;
  // Phase 2: when true, the per-layer Linear projections route through
  // LinearInt8 using the INT8 weight tensors populated on each
  // LayerWeights. Set by Model::QuantizeWeights; default false.
  bool weights_quantized = false;
  // Phase 2 Slice 5 sensitivity escape: when true, the activation
  // feeding layer 0's fc1 is rounded to FP16 precision (FP32 -> half
  // -> FP32 round-trip) before the fc1 Linear. Cheapest outlier
  // mitigation per research-report §"Recommended INT8 quantization
  // recipe" when SmoothQuant alone doesn't close the PPPL gap.
  bool first_block_fc1_fp16 = false;
};

struct LayerWeights {
  // Pre-attention LayerNorm (sits in attention block, before the projections).
  std::vector<float> attn_ln_w, attn_ln_b;
  // Self-attention projections; weight shape [d, d] (out, in) row-major.
  std::vector<float> q_w, q_b;
  std::vector<float> k_w, k_b;
  std::vector<float> v_w, v_b;
  std::vector<float> out_w, out_b;
  // Pre-FFN LayerNorm.
  std::vector<float> ffn_ln_w, ffn_ln_b;
  // FFN: fc1 [4d, d], fc2 [d, 4d].
  std::vector<float> fc1_w, fc1_b;
  std::vector<float> fc2_w, fc2_b;

  // Phase 2: per-channel INT8 versions, populated by Model::QuantizeWeights.
  // When the parent Config's weights_quantized is true, the forward path
  // reads these instead of the FP32 vectors above. Biases stay FP32.
  esm::quant::QuantizedTensor q_w_int8, k_w_int8, v_w_int8;
  esm::quant::QuantizedTensor out_w_int8, fc1_w_int8, fc2_w_int8;
};

// Forward output. `hidden_states` semantics match HF EsmModel:
//   hidden_states[0]          = post-embed (after token_dropout rescale)
//   hidden_states[i] (1..N-1) = output of layer i-1
//   hidden_states[N]          = output of layer N-1, after final LN
class Model {
 public:
  // Load ESM-2 weights from an HF safetensors file. Infers architecture
  // (num_layers, hidden, heads, FFN) from tensor shapes alone — no
  // config.json required.
  static std::unique_ptr<Model> LoadFromSafetensors(const std::string& path);

  const Config& config() const { return cfg_; }

  // Run forward. attention_mask uses 1 for real tokens, 0 for pad.
  // Returns logits as a row-major [seq_len, vocab_size] vector.
  //
  // NOT thread-safe. A single Model instance owns one Workspace and one
  // forward may be in flight at a time. Phase 3's scheduler will introduce
  // per-thread Workspaces for concurrent forwards.
  std::vector<float> Forward(std::span<const int32_t> input_ids,
                             std::span<const int32_t> attention_mask) const;

  // Same as Forward, but also collects every hidden state. If
  // hidden_states_out is not null it is resized to num_hidden_layers + 1
  // entries (post-embed, per-layer outputs, final post-LN).
  std::vector<float> ForwardWithHiddenStates(
      std::span<const int32_t> input_ids,
      std::span<const int32_t> attention_mask,
      std::vector<std::vector<float>>* hidden_states_out) const;

  // Bytes the per-forward arena currently holds. Useful for the zero-alloc
  // regression test: after the first forward at a given length, the arena
  // capacity should stay constant on subsequent calls at the same length.
  std::size_t workspace_capacity_bytes() const { return ws_.bytes_capacity(); }

  // Run a batch of independent sequences in parallel. Each sequence runs
  // through its own Workspace from a per-thread pool; the dispatch axis
  // is the batch dimension. ForwardPacked is the cu_seqlens-packed sibling
  // that runs all sequences as one packed forward through the encoder.
  std::vector<std::vector<float>> ForwardBatch(
      const std::vector<std::vector<std::int32_t>>& input_ids,
      const std::vector<std::vector<std::int32_t>>& attention_masks) const;

  // Phase 3 Slice 1: run a packed batch as a single concatenated forward.
  // The encoder sees one [T, d] tensor where T = sum of per-sequence lengths;
  // per-sequence attention is enforced via cu_seqlens inside AttentionVarlen.
  // Returns one logits buffer per sequence, each shaped [L_b, vocab_size].
  // NOT thread-safe: the single Model instance owns one Workspace.
  std::vector<std::vector<float>> ForwardPacked(const BatchView& batch) const;

  // Number of threads used by ForwardBatch (process-global pool sized
  // from ESM_NUM_THREADS at first Model::load, default physical-core).
  static std::size_t num_threads();

  // Phase 2 Slice 5: toggle the first-block FP16 escape on/off.
  void SetFirstBlockFc1Fp16(bool enabled) {
    cfg_.first_block_fc1_fp16 = enabled;
  }

  // Phase 2: quantize every per-layer Linear weight in place to
  // per-channel symmetric INT8 and flag the model as quantized.
  // Biases stay FP32; lm_head stays FP32 (Slice 5 escape list).
  // After this call, Forward / ForwardBatch route the per-layer
  // projections through LinearInt8 instead of Linear.
  void QuantizeWeights();

  // Phase 2: apply the SmoothQuant migration in place using the per-site
  // activation stats from a calibration run. Identity-preserving to FP32
  // round-off; the load-bearing checkpoint for Phase 2 is verifying
  // forward(post-migration) ≈ forward(pre-migration).
  void ApplySmoothQuant(
      const std::unordered_map<std::string, float>& act_stats, float alpha);

  // Phase 2: run the forward pass and feed every Linear-input activation
  // into the observer at well-known site keys for SmoothQuant calibration:
  //   layer<i>.attn_ln_output     (input to Q/K/V projections)
  //   layer<i>.attn_out           (input to out_proj)
  //   layer<i>.ffn_ln_output      (input to fc1)
  //   layer<i>.inter_gelu         (input to fc2)
  // Returns the same logits as Forward().
  std::vector<float> ForwardWithObserver(
      std::span<const std::int32_t> input_ids,
      std::span<const std::int32_t> attention_mask,
      ActivationObserver* observer) const;

 private:
  Model() = default;
  // Single-sequence forward (B=1 wrapper around ForwardPackedInto). Writes
  // a flat [L * vocab_size] logits vector. Forward / ForwardWithHiddenStates
  // / ForwardWithObserver are the public entry points.
  void ForwardInto(
      std::span<const std::int32_t> input_ids,
      std::span<const std::int32_t> attention_mask, Workspace& ws,
      std::vector<float>* logits_out,
      std::vector<std::vector<float>>* hidden_states_out,
      ActivationObserver* observer = nullptr) const;

  // Packed-batch forward core: writes one logits buffer per sequence into
  // logits_per_seq_out (each [L_b * V]). If hidden_packed_out is non-null,
  // it receives the packed [T, d] hidden state at every layer + the final
  // post-LN tensor (semantics match ForwardWithHiddenStates but on T not L).
  void ForwardPackedInto(
      const BatchView& batch, Workspace& ws,
      std::vector<std::vector<float>>* logits_per_seq_out,
      std::vector<std::vector<float>>* hidden_packed_out = nullptr,
      ActivationObserver* observer = nullptr) const;

  Config cfg_{};
  std::vector<float> embed_;            // [vocab_size, hidden_size]
  std::vector<LayerWeights> layers_;
  std::vector<float> final_ln_w_, final_ln_b_;
  // lm_head
  std::vector<float> lm_dense_w_, lm_dense_b_;
  std::vector<float> lm_ln_w_, lm_ln_b_;
  std::vector<float> lm_decoder_bias_;  // [vocab_size]
  // lm_head.decoder.weight is tied to embed_ — we reuse embed_ directly.

  // Per-Model scratch arena. Mutable because Forward is logically const
  // (the model state doesn't change), but the arena bumps and resets.
  mutable Workspace ws_;
};

}  // namespace esm
