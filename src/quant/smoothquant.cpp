#include "esm_cpp/smoothquant.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "esm_cpp/model.h"

namespace esm::quant {

namespace {

constexpr float kScaleMin = 1e-3f;
constexpr float kScaleMax = 1e3f;

// Per-column max-abs of a row-major [N, K] matrix.
std::vector<float> ColumnMaxAbs(const std::vector<float>& W, int N, int K) {
  std::vector<float> out(K, 0.0f);
  for (int n = 0; n < N; ++n) {
    const float* row = W.data() + static_cast<long>(n) * K;
    for (int k = 0; k < K; ++k) {
      out[k] = std::max(out[k], std::fabs(row[k]));
    }
  }
  return out;
}

// Per-row max-abs of a row-major [N, K] matrix.
std::vector<float> RowMaxAbs(const std::vector<float>& W, int N, int K) {
  std::vector<float> out(N, 0.0f);
  for (int n = 0; n < N; ++n) {
    const float* row = W.data() + static_cast<long>(n) * K;
    float m = 0.0f;
    for (int k = 0; k < K; ++k) m = std::max(m, std::fabs(row[k]));
    out[n] = m;
  }
  return out;
}

float LookupOrZero(const std::unordered_map<std::string, float>& m,
                   const std::string& key) {
  auto it = m.find(key);
  return it == m.end() ? 0.0f : it->second;
}

}  // namespace

std::vector<float> SmoothQuantScales(const std::vector<float>& P_X,
                                      const std::vector<float>& P_W,
                                      float alpha) {
  if (P_X.size() != P_W.size()) {
    throw std::runtime_error("SmoothQuantScales: P_X and P_W size mismatch");
  }
  const std::size_t K = P_X.size();
  std::vector<float> s(K, 1.0f);
  for (std::size_t k = 0; k < K; ++k) {
    if (P_X[k] <= 0.0f || P_W[k] <= 0.0f) {
      s[k] = 1.0f;
      continue;
    }
    const float numerator = std::pow(P_X[k], alpha);
    const float denominator = std::pow(P_W[k], 1.0f - alpha);
    float val = numerator / denominator;
    if (!std::isfinite(val) || val <= 0.0f) {
      s[k] = 1.0f;
      continue;
    }
    if (val < kScaleMin) val = kScaleMin;
    if (val > kScaleMax) val = kScaleMax;
    s[k] = val;
  }
  return s;
}

namespace {

// "Channel k has P_X[k] = act_stat[k]" — but the observer returns a
// single per-tensor scalar per site, not per-channel. To bridge: for
// activation X with shape [B, K], the observer's percentile is the
// percentile of |X| flattened across all (b, k). That's a SINGLE
// number, not a per-channel vector.
//
// Per-channel SmoothQuant nominally wants per-channel activation maxes.
// Since the SPEC mandates per-tensor activation observer (cheaper, more
// stable), we treat all channels as having the same P_X. The per-
// channel asymmetry then comes entirely from P_W. That degenerates the
// smoothing to "uniformly scale activation down by sqrt(P_X / mean(P_W))"
// — less powerful than per-channel-activation SmoothQuant but still
// helps where weights are spiky (Bondarenko/Dettmers outlier channels).
//
// If we ever need per-channel activation stats, add a sibling observer
// that tracks per-channel reservoirs and feed into this helper.
std::vector<float> BroadcastScalarToChannels(float P_X_scalar, std::size_t K) {
  return std::vector<float>(K, P_X_scalar);
}

}  // namespace

void MigrateSmoothQuant(
    esm::LayerWeights& layer,
    const std::unordered_map<std::string, float>& act_stats, int layer_index,
    float alpha) {
  const std::string prefix = "layer" + std::to_string(layer_index);

  // ------------------------------------------------------------------
  // Site 1: attn_ln_output -> {Q, K, V}.
  // P_X = act_stats[layer<i>.attn_ln_output]  (scalar, broadcast).
  // P_W[k] = max(|q_w[:, k]|, |k_w[:, k]|, |v_w[:, k]|).
  // attn_ln_w[k] /= s[k]; attn_ln_b[k] /= s[k].
  // {q,k,v}_w[n, k] *= s[k].
  // ------------------------------------------------------------------
  if (!layer.attn_ln_w.empty()) {
    const int K = static_cast<int>(layer.attn_ln_w.size());
    const int Nq = static_cast<int>(layer.q_w.size()) / K;
    auto P_W_q = ColumnMaxAbs(layer.q_w, Nq, K);
    auto P_W_k = ColumnMaxAbs(layer.k_w, Nq, K);
    auto P_W_v = ColumnMaxAbs(layer.v_w, Nq, K);
    std::vector<float> P_W(K, 0.0f);
    for (int k = 0; k < K; ++k) {
      P_W[k] = std::max({P_W_q[k], P_W_k[k], P_W_v[k]});
    }
    const float P_X_scalar = LookupOrZero(act_stats, prefix + ".attn_ln_output");
    auto P_X = BroadcastScalarToChannels(P_X_scalar, K);
    auto s = SmoothQuantScales(P_X, P_W, alpha);
    for (int k = 0; k < K; ++k) {
      layer.attn_ln_w[k] /= s[k];
      layer.attn_ln_b[k] /= s[k];
    }
    for (auto* w_ptr : {&layer.q_w, &layer.k_w, &layer.v_w}) {
      auto& W = *w_ptr;
      const int N = static_cast<int>(W.size()) / K;
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) W[n * K + k] *= s[k];
      }
    }
  }

  // ------------------------------------------------------------------
  // Site 2: attn_out -> out_proj.
  // P_X = act_stats[layer<i>.attn_out]; P_W[n] = max(|out_w[:, n]|).
  // V's output column n (= attn_out column n) scaled by 1/s[n]:
  //   v_w[n, :] /= s[n]; v_b[n] /= s[n].
  // out_w[m, n] *= s[n].
  // ------------------------------------------------------------------
  if (!layer.v_b.empty() && !layer.out_w.empty()) {
    const int N_v = static_cast<int>(layer.v_b.size());
    const int K_out = N_v;  // out_proj's input axis equals V's output axis
    const int N_out = static_cast<int>(layer.out_w.size()) / K_out;
    auto P_W_out = ColumnMaxAbs(layer.out_w, N_out, K_out);
    const float P_X_scalar = LookupOrZero(act_stats, prefix + ".attn_out");
    auto P_X = BroadcastScalarToChannels(P_X_scalar, K_out);
    auto s = SmoothQuantScales(P_X, P_W_out, alpha);
    const int K_v = static_cast<int>(layer.v_w.size()) / N_v;
    for (int n = 0; n < N_v; ++n) {
      layer.v_b[n] /= s[n];
      for (int k = 0; k < K_v; ++k) layer.v_w[n * K_v + k] /= s[n];
    }
    for (int m = 0; m < N_out; ++m) {
      for (int k = 0; k < K_out; ++k) layer.out_w[m * K_out + k] *= s[k];
    }
  }

  // ------------------------------------------------------------------
  // Site 3: ffn_ln_output -> fc1.
  // ------------------------------------------------------------------
  if (!layer.ffn_ln_w.empty() && !layer.fc1_w.empty()) {
    const int K_ffn = static_cast<int>(layer.ffn_ln_w.size());
    const int N_fc1 = static_cast<int>(layer.fc1_w.size()) / K_ffn;
    auto P_W_fc1 = ColumnMaxAbs(layer.fc1_w, N_fc1, K_ffn);
    const float P_X_scalar = LookupOrZero(act_stats, prefix + ".ffn_ln_output");
    auto P_X = BroadcastScalarToChannels(P_X_scalar, K_ffn);
    auto s = SmoothQuantScales(P_X, P_W_fc1, alpha);
    for (int k = 0; k < K_ffn; ++k) {
      layer.ffn_ln_w[k] /= s[k];
      layer.ffn_ln_b[k] /= s[k];
    }
    for (int n = 0; n < N_fc1; ++n) {
      for (int k = 0; k < K_ffn; ++k) layer.fc1_w[n * K_ffn + k] *= s[k];
    }
  }

  // inter_gelu -> fc2 deliberately NOT migrated; GELU non-linearity
  // breaks the diagonal-rescale identity.
}

}  // namespace esm::quant
