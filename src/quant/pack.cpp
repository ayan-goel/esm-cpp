#include "esm_cpp/quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "esm_cpp/cpu_features.h"

namespace esm::quant {

void BuildVnniCache(QuantizedTensor* out) {
  const int N = out->N;
  const int K = out->K;
  const int K_pad = (K + 3) & ~3;
  // Each N-panel is laid out as 16 rows × K_pad bytes worth of VPDPBUSD
  // tiles, regardless of how many real rows the panel covers. Round N up
  // to a multiple of 16 so the last (possibly partial) panel has the full
  // 16-row footprint reserved. Production ESM dims (320/480/640/1280/2560)
  // are already multiples of 16 so this is identity in practice; the
  // round-up matters for unit-test shapes that exercise tails.
  const int N_pad = (N + 15) & ~15;
  out->packed_vnni.assign(static_cast<std::size_t>(N_pad) *
                              static_cast<std::size_t>(K_pad),
                          std::int8_t{0});
  for (int nb = 0; nb < N; nb += 16) {
    const int n_block = std::min(16, N - nb);
    std::int8_t* panel =
        out->packed_vnni.data() + static_cast<long>(nb) * K_pad;
    for (int kb = 0; kb < K; kb += 4) {
      std::int8_t* tile = panel + static_cast<long>(kb) * 16;
      const int k_step = std::min(4, K - kb);
      for (int nn = 0; nn < n_block; ++nn) {
        for (int kk = 0; kk < k_step; ++kk) {
          tile[nn * 4 + kk] =
              out->packed[static_cast<std::size_t>(nb + nn) * K + (kb + kk)];
        }
      }
    }
  }
  out->col_sum.assign(static_cast<std::size_t>(N), 0);
  for (int n = 0; n < N; ++n) {
    std::int32_t s = 0;
    const std::int8_t* row =
        out->packed.data() + static_cast<long>(n) * K;
    for (int k = 0; k < K; ++k) s += row[k];
    out->col_sum[static_cast<std::size_t>(n)] = s;
  }
}

void BuildArmCache(QuantizedTensor* out) {
  const int N = out->N;
  const int K = out->K;
  const int K_pad = (K + 3) & ~3;
  const int N_pad = (N + 3) & ~3;
  out->packed_arm.assign(static_cast<std::size_t>(N_pad) *
                             static_cast<std::size_t>(K_pad),
                         std::int8_t{0});
  // Each 4-N panel holds K_pad/4 tiles of 16 bytes; panel stride is K_pad*4
  // bytes = nb*K_pad from panel index nb/4. Tile t (k in [4t, 4t+4)) is
  // [n0:k0..k3, n1:k0..k3, n2:.., n3:..]; broadcasting a[m][4t..4t+3] and
  // vdotq against it produces 4 column dot-products in one int32x4.
  for (int nb = 0; nb < N; nb += 4) {
    const int n_block = std::min(4, N - nb);
    std::int8_t* panel = out->packed_arm.data() + static_cast<long>(nb) * K_pad;
    for (int kb = 0; kb < K; kb += 4) {
      std::int8_t* tile = panel + static_cast<long>(kb) * 4;
      const int k_step = std::min(4, K - kb);
      for (int nn = 0; nn < n_block; ++nn) {
        for (int kk = 0; kk < k_step; ++kk) {
          tile[nn * 4 + kk] =
              out->packed[static_cast<std::size_t>(nb + nn) * K + (kb + kk)];
        }
      }
    }
  }
}

void BuildI8mmCache(QuantizedTensor* out) {
  const int N = out->N;
  const int K = out->K;
  const int K_pad8 = (K + 7) & ~7;
  const int N_pad2 = (N + 1) & ~1;
  out->packed_arm_i8mm.assign(static_cast<std::size_t>(N_pad2) *
                                  static_cast<std::size_t>(K_pad8),
                              std::int8_t{0});
  // Each 2-column pair holds K_pad8/8 tiles of 16 bytes; the col-pair index
  // nb/2 sits at byte offset nb*K_pad8. Tile t (k in [8t, 8t+8)) is
  // [n0:k0..k7, n1:k0..k7] — the 2x8 SMMLA right-hand operand.
  for (int nb = 0; nb < N; nb += 2) {
    const int n_block = std::min(2, N - nb);
    std::int8_t* pair =
        out->packed_arm_i8mm.data() + static_cast<long>(nb) * K_pad8;
    for (int kb = 0; kb < K; kb += 8) {
      std::int8_t* tile = pair + static_cast<long>(kb) * 2;
      const int k_step = std::min(8, K - kb);
      for (int nn = 0; nn < n_block; ++nn) {
        for (int kk = 0; kk < k_step; ++kk) {
          tile[nn * 8 + kk] =
              out->packed[static_cast<std::size_t>(nb + nn) * K + (kb + kk)];
        }
      }
    }
  }
}

void BuildKernelCache(QuantizedTensor* out) {
#if defined(__aarch64__) || defined(_M_ARM64)
  if (esm::CurrentIsa() == esm::Isa::NeonI8mm && esm::ArmUseSmmla()) {
    BuildI8mmCache(out);
  } else {
    BuildArmCache(out);
  }
#else
  BuildVnniCache(out);
#endif
}

void QuantizeActivationRef(const float* x, std::size_t n, float scale,
                            std::uint8_t* q) {
  if (scale == 0.0f) {
    for (std::size_t i = 0; i < n; ++i) q[i] = 128;
    return;
  }
  const float inv_scale = 1.0f / scale;
  for (std::size_t i = 0; i < n; ++i) {
    float v = std::nearbyint(x[i] * inv_scale);
    if (v > 127.0f) v = 127.0f;
    if (v < -127.0f) v = -127.0f;
    q[i] = static_cast<std::uint8_t>(static_cast<int>(v) + 128);
  }
}

void QuantizeActivationSymmetricRef(const float* x, std::size_t n, float scale,
                                    std::int8_t* q) {
  if (scale == 0.0f) {
    for (std::size_t i = 0; i < n; ++i) q[i] = 0;
    return;
  }
  const float inv_scale = 1.0f / scale;
  for (std::size_t i = 0; i < n; ++i) {
    float v = std::nearbyint(x[i] * inv_scale);
    if (v > 127.0f) v = 127.0f;
    if (v < -127.0f) v = -127.0f;
    q[i] = static_cast<std::int8_t>(static_cast<int>(v));
  }
}

void Quantize(const float* W_fp32, int N, int K, QuantizedTensor* out) {
  out->N = N;
  out->K = K;
  out->packed.assign(static_cast<std::size_t>(N) * K, 0);
  out->per_channel_scales.assign(N, 0.0f);
  for (int n = 0; n < N; ++n) {
    float row_max = 0.0f;
    const float* row = W_fp32 + static_cast<long>(n) * K;
    for (int k = 0; k < K; ++k) {
      row_max = std::max(row_max, std::fabs(row[k]));
    }
    if (row_max == 0.0f) {
      // Already zero-initialized; nothing to round.
      out->per_channel_scales[n] = 0.0f;
      continue;
    }
    const float scale = row_max / 127.0f;
    out->per_channel_scales[n] = scale;
    const float inv_scale = 1.0f / scale;
    for (int k = 0; k < K; ++k) {
      float q = std::nearbyint(row[k] * inv_scale);
      if (q > 127.0f) q = 127.0f;
      if (q < -127.0f) q = -127.0f;
      out->packed[static_cast<std::size_t>(n) * K + k] =
          static_cast<std::int8_t>(q);
    }
  }
  BuildKernelCache(out);
}

}  // namespace esm::quant
