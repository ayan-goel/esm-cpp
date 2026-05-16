#include "esm_cpp/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// AVX-512 path uses intrinsics + thread_local scratch. Headers stay at
// file scope (per the gemm_int8.cpp lesson — system headers inside the
// namespace block leak C-stdlib symbols into esm::kernels).
#ifdef ESM_KERNEL_AVX512
#include <immintrin.h>
#endif

namespace esm::kernels {

#ifdef ESM_KERNEL_REFERENCE

// Scaled-dot self-attention; Q must already be scaled by 1/sqrt(head_dim)
// per the ESM convention (scale before RoPE, not the score after).
//   Q, K, V: [num_heads, seq_len, head_dim]
//   attention_mask: [seq_len] with 1 for real tokens and 0 for pad,
//                   or nullptr to treat everything as real.
//   out: [seq_len, num_heads * head_dim] with heads concatenated along
//        the last dimension (matches HF .transpose(1,2).reshape(L, -1)).
// Softmax accumulator is FP32 — kept in double for numerical safety on
// long sequences, since this is the reference path.
void AttentionRef(const float* Q, const float* K, const float* V,
                  const int* attention_mask, float* out, int num_heads,
                  int seq_len, int head_dim) {
  std::vector<float> scores(static_cast<std::size_t>(seq_len));
  for (int h = 0; h < num_heads; ++h) {
    const float* Qh = Q + static_cast<long>(h) * seq_len * head_dim;
    const float* Kh = K + static_cast<long>(h) * seq_len * head_dim;
    const float* Vh = V + static_cast<long>(h) * seq_len * head_dim;
    for (int i = 0; i < seq_len; ++i) {
      const float* qi = Qh + static_cast<long>(i) * head_dim;
      float max_score = -std::numeric_limits<float>::infinity();
      for (int j = 0; j < seq_len; ++j) {
        const float* kj = Kh + static_cast<long>(j) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
        if (attention_mask && attention_mask[j] == 0) {
          dot = -std::numeric_limits<float>::infinity();
        }
        scores[static_cast<std::size_t>(j)] = dot;
        if (dot > max_score) max_score = dot;
      }
      double sum = 0.0;
      for (int j = 0; j < seq_len; ++j) {
        float e = (scores[static_cast<std::size_t>(j)] == -std::numeric_limits<float>::infinity())
                      ? 0.0f
                      : std::exp(scores[static_cast<std::size_t>(j)] - max_score);
        scores[static_cast<std::size_t>(j)] = e;
        sum += e;
      }
      float inv_sum = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
      float* out_row = out + (static_cast<long>(i) * num_heads + h) * head_dim;
      for (int d = 0; d < head_dim; ++d) out_row[d] = 0.0f;
      for (int j = 0; j < seq_len; ++j) {
        float w = scores[static_cast<std::size_t>(j)] * inv_sum;
        if (w == 0.0f) continue;
        const float* vj = Vh + static_cast<long>(j) * head_dim;
        for (int d = 0; d < head_dim; ++d) out_row[d] += w * vj[d];
      }
    }
  }
}

// Packed-varlen scaled-dot attention with FP32 softmax accumulator.
// Layout differs from AttentionRef: Q/K/V are token-major [T, H, dh]
// rather than head-major [H, L, dh]; output is still [T, H*dh] so it
// drops in to model.cpp without further rearrangement. cu_seqlens
// isolates sequences — each query at token t in sequence b only attends
// to keys/values from positions [cu_seqlens[b], cu_seqlens[b+1]).
//
// This is the scalar reference; tile size is 1. Slice 4.3 will add the
// FlashAttention-style tiled AVX-512 path against this oracle.
void AttentionVarlenRef(const float* q, const float* k, const float* v,
                        const int* cu_seqlens, int batch_size, int num_heads,
                        int head_dim, float* out) {
  for (int b = 0; b < batch_size; ++b) {
    const int seq_start = cu_seqlens[b];
    const int seq_end = cu_seqlens[b + 1];
    const int seq_len = seq_end - seq_start;
    if (seq_len <= 0) continue;
    std::vector<float> scores(static_cast<std::size_t>(seq_len));
    for (int h = 0; h < num_heads; ++h) {
      for (int i = 0; i < seq_len; ++i) {
        const int t_q = seq_start + i;
        const float* qi =
            q + (static_cast<long>(t_q) * num_heads + h) * head_dim;
        float max_score = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < seq_len; ++j) {
          const int t_k = seq_start + j;
          const float* kj =
              k + (static_cast<long>(t_k) * num_heads + h) * head_dim;
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
          scores[static_cast<std::size_t>(j)] = dot;
          if (dot > max_score) max_score = dot;
        }
        double sum = 0.0;
        for (int j = 0; j < seq_len; ++j) {
          float e = std::exp(scores[static_cast<std::size_t>(j)] - max_score);
          scores[static_cast<std::size_t>(j)] = e;
          sum += e;
        }
        float inv_sum = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
        float* out_row =
            out + (static_cast<long>(t_q) * num_heads + h) * head_dim;
        for (int d = 0; d < head_dim; ++d) out_row[d] = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
          float w = scores[static_cast<std::size_t>(j)] * inv_sum;
          if (w == 0.0f) continue;
          const int t_k = seq_start + j;
          const float* vj =
              v + (static_cast<long>(t_k) * num_heads + h) * head_dim;
          for (int d = 0; d < head_dim; ++d) out_row[d] += w * vj[d];
        }
      }
    }
  }
}

#endif  // ESM_KERNEL_REFERENCE

#ifdef ESM_KERNEL_AVX512

namespace {

// Polynomial exp(x) helper, shared shape with the GELU/LayerNorm path.
// 5-term Horner of the Taylor series in r = x - n*ln(2), with 2^n built
// via the integer-cast bit-hack. FP32 accurate over the input range we
// see here (softmax shifts scores into (-∞, 0], so |r| ≤ ln(2)/2).
inline __m512 ExpAvx512Attn(__m512 x) {
  const __m512 log2e = _mm512_set1_ps(1.44269504088896340736f);
  const __m512 ln2_hi = _mm512_set1_ps(6.93145752e-1f);
  const __m512 ln2_lo = _mm512_set1_ps(1.42860677e-6f);
  __m512 n_f = _mm512_roundscale_ps(
      _mm512_mul_ps(x, log2e),
      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_sub_ps(_mm512_sub_ps(x, _mm512_mul_ps(n_f, ln2_hi)),
                            _mm512_mul_ps(n_f, ln2_lo));
  __m512 p = _mm512_set1_ps(1.0f / 120.0f);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
  __m512i ni = _mm512_cvtps_epi32(n_f);
  __m512i e_bits = _mm512_slli_epi32(
      _mm512_add_epi32(ni, _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(p, _mm512_castsi512_ps(e_bits));
}

// Scores scratch buffer per worker thread. Reused across queries; grows
// monotonically. Reserve 16 trailing slack floats so the masked-tail
// load/store on the final partial vector never reads out of the
// allocation (we leave those slots as zero).
thread_local std::vector<float> g_attn_scores;

inline float ReduceMaxPs(__m512 v) {
  return _mm512_reduce_max_ps(v);
}

inline float ReduceAddPs(__m512 v) {
  return _mm512_reduce_add_ps(v);
}

// AVX-512 dot product of two head_dim-long FP32 vectors. head_dim ∈
// {16, 24, 32, 64} for ESM-2 (8M / 35M / 150M / 650M); none are vector
// multiples but all fit in 1–4 zmm with a masked tail.
inline float DotAvx512(const float* a, const float* b, int n) {
  __m512 acc = _mm512_setzero_ps();
  int d = 0;
  for (; d + 16 <= n; d += 16) {
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + d), _mm512_loadu_ps(b + d), acc);
  }
  if (d < n) {
    const int tail = n - d;
    const __mmask16 mask = static_cast<__mmask16>((1u << tail) - 1u);
    __m512 av = _mm512_maskz_loadu_ps(mask, a + d);
    __m512 bv = _mm512_maskz_loadu_ps(mask, b + d);
    acc = _mm512_fmadd_ps(av, bv, acc);
  }
  return ReduceAddPs(acc);
}

}  // namespace

void AttentionVarlenAvx512(const float* q, const float* k, const float* v,
                            const int* cu_seqlens, int batch_size,
                            int num_heads, int head_dim, float* out) {
  for (int b = 0; b < batch_size; ++b) {
    const int seq_start = cu_seqlens[b];
    const int seq_end = cu_seqlens[b + 1];
    const int seq_len = seq_end - seq_start;
    if (seq_len <= 0) continue;
    if (g_attn_scores.size() <
        static_cast<std::size_t>(seq_len + 16)) {
      g_attn_scores.assign(static_cast<std::size_t>(seq_len + 16), 0.0f);
    } else {
      std::fill(g_attn_scores.begin() + seq_len,
                g_attn_scores.begin() + seq_len + 16, 0.0f);
    }
    float* scores = g_attn_scores.data();

    for (int h = 0; h < num_heads; ++h) {
      for (int i = 0; i < seq_len; ++i) {
        const int t_q = seq_start + i;
        const float* qi =
            q + (static_cast<long>(t_q) * num_heads + h) * head_dim;

        // Pass 1: scores[j] = qi · kj for each j; track running max.
        __m512 v_max = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        int j = 0;
        for (; j + 16 <= seq_len; j += 16) {
          // Compute 16 scores; only the dot itself is vectorized inside
          // DotAvx512 — we still do 16 separate dot products per j-chunk
          // because each one depends on a different K row.
          for (int jj = 0; jj < 16; ++jj) {
            const int t_k = seq_start + j + jj;
            const float* kj =
                k + (static_cast<long>(t_k) * num_heads + h) * head_dim;
            scores[j + jj] = DotAvx512(qi, kj, head_dim);
          }
          v_max = _mm512_max_ps(v_max,
                                _mm512_loadu_ps(scores + j));
        }
        for (; j < seq_len; ++j) {
          const int t_k = seq_start + j;
          const float* kj =
              k + (static_cast<long>(t_k) * num_heads + h) * head_dim;
          scores[j] = DotAvx512(qi, kj, head_dim);
        }
        float max_score = ReduceMaxPs(v_max);
        for (int jj = (seq_len & ~15); jj < seq_len; ++jj) {
          if (scores[jj] > max_score) max_score = scores[jj];
        }

        // Pass 2: scores[j] = exp(scores[j] - max); sum.
        const __m512 v_max_bcast = _mm512_set1_ps(max_score);
        __m512 v_sum = _mm512_setzero_ps();
        j = 0;
        for (; j + 16 <= seq_len; j += 16) {
          __m512 s = _mm512_sub_ps(_mm512_loadu_ps(scores + j), v_max_bcast);
          __m512 e = ExpAvx512Attn(s);
          _mm512_storeu_ps(scores + j, e);
          v_sum = _mm512_add_ps(v_sum, e);
        }
        float sum = ReduceAddPs(v_sum);
        for (; j < seq_len; ++j) {
          const float e = std::exp(scores[j] - max_score);
          scores[j] = e;
          sum += e;
        }
        const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;

        // Pass 3: out_row = sum_j (scores[j] * inv_sum) * V[j].
        float* out_row =
            out + (static_cast<long>(t_q) * num_heads + h) * head_dim;
        // Zero out_row.
        {
          int d = 0;
          for (; d + 16 <= head_dim; d += 16) {
            _mm512_storeu_ps(out_row + d, _mm512_setzero_ps());
          }
          if (d < head_dim) {
            const int tail = head_dim - d;
            const __mmask16 mask = static_cast<__mmask16>((1u << tail) - 1u);
            _mm512_mask_storeu_ps(out_row + d, mask, _mm512_setzero_ps());
          }
        }
        for (int jj = 0; jj < seq_len; ++jj) {
          const float w = scores[jj] * inv_sum;
          if (w == 0.0f) continue;
          const int t_k = seq_start + jj;
          const float* vj =
              v + (static_cast<long>(t_k) * num_heads + h) * head_dim;
          const __m512 wv = _mm512_set1_ps(w);
          int d = 0;
          for (; d + 16 <= head_dim; d += 16) {
            __m512 vv = _mm512_loadu_ps(vj + d);
            __m512 ov = _mm512_loadu_ps(out_row + d);
            _mm512_storeu_ps(out_row + d, _mm512_fmadd_ps(vv, wv, ov));
          }
          if (d < head_dim) {
            const int tail = head_dim - d;
            const __mmask16 mask = static_cast<__mmask16>((1u << tail) - 1u);
            __m512 vv = _mm512_maskz_loadu_ps(mask, vj + d);
            __m512 ov = _mm512_maskz_loadu_ps(mask, out_row + d);
            _mm512_mask_storeu_ps(out_row + d, mask,
                                   _mm512_fmadd_ps(vv, wv, ov));
          }
        }
      }
    }
  }
}

#endif  // ESM_KERNEL_AVX512

}  // namespace esm::kernels
