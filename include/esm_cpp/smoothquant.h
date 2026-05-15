#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace esm {
struct LayerWeights;
}

namespace esm::quant {

// Compute the per-input-channel SmoothQuant scale vector
//   s[k] = (P_X[k])^alpha / (P_W[k])^(1 - alpha)
// where P_X[k] is the activation magnitude percentile for channel k
// and P_W[k] is the weight max-abs for channel k across the consumer
// matrix's output axis. Channels where either P_X or P_W is zero are
// pinned to s[k] = 1 (no-op for that channel). The result is clamped
// to [1e-3, 1e3] so a pathological channel doesn't destabilize the
// activation rescale at INT8 time.
std::vector<float> SmoothQuantScales(const std::vector<float>& P_X,
                                      const std::vector<float>& P_W,
                                      float alpha);

// Rewrite one LayerWeights to migrate activation outliers into the
// per-input-channel weight scales. Operates on three sites per layer:
//   1. attn_ln_output -> {Q, K, V}: divides attn_ln.{w,b}[k] by s[k]
//      and multiplies {q,k,v}_w[n, k] by s[k]. P_W is the joint per-
//      channel max-abs across q_w, k_w, v_w rows.
//   2. attn_out -> out_proj: divides v_w[n, :] by s[n] and v_b[n] by
//      s[n]; multiplies out_w[m, k] by s[k]. P_W is per-column max-abs
//      across out_w's rows.
//   3. ffn_ln_output -> fc1: divides ffn_ln.{w,b}[k] by s[k] and
//      multiplies fc1_w[n, k] by s[k].
//
// The inter_gelu -> fc2 site is NOT migrated; the GELU non-linearity
// breaks the diagonal-rescale identity. Slice 6 may add an approximate
// channel-wise smoothing here if PPPL budget allows.
//
// `act_stats` keys: "layer<i>.{attn_ln_output,attn_out,ffn_ln_output}".
// `layer_index` selects which entries to look up. Identity-preserving
// to FP32 round-off when applied to a model and its forward.
void MigrateSmoothQuant(esm::LayerWeights& layer,
                        const std::unordered_map<std::string, float>& act_stats,
                        int layer_index, float alpha);

}  // namespace esm::quant
