#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

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

 private:
  Model() = default;

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
