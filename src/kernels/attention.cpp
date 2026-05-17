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

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "esm_cpp/thread_pool.h"
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

namespace {

// One (batch, head) attention pass. Two structural improvements over
// the earlier per-query implementation:
//
//   Pass 1: 16 independent zmm accumulators per j-chunk. The previous
//   code called DotAvx512 sixteen times sequentially, producing a
//   serial chain of horizontal reductions even though each j is fully
//   independent. The new inner head_dim loop loads a single q_chunk
//   and FMAs it against 16 different K-row chunks into 16 parallel
//   accumulators — same op count, full FMA-pipeline utilization.
//
//   Pass 3: O accumulated in zmm registers across the entire seq_len
//   loop, then stored once at end. The previous code reloaded
//   out_row from L1 every jj iteration (4 zmm reads + 4 zmm writes
//   for head_dim=64) — ~2 KB of L1 traffic per query that's now zero.
//
// FP32 accumulator throughout (CLAUDE.md mandatory under long-seq INT8 KV).
// Max head_dim = 64 (650M / 3B) → 4 zmm output chunks; we size the O
// register array at 8 for headroom past v0.1.
//
// The body is templated on head_dim where useful: 16/32/64 are the
// production paths (multiples of 16, no masked tail) and they let the
// compiler fully unroll the inner d-loop and elide the runtime tail
// branch. 24 (35M) and dynamic head_dim fall through to the dynamic
// dispatch.
constexpr int kOMaxChunks = 8;

template <int HEAD_DIM_T>
inline void RunOneHeadAvx512Impl(const float* q, const float* k,
                                  const float* v, int seq_start, int seq_len,
                                  int h, int num_heads, int head_dim_arg,
                                  float* out, float* scores) {
  // HEAD_DIM_T = -1 means "dynamic"; otherwise it's a compile-time
  // constant that lets the compiler unroll. Hot paths (16/32/64) use
  // the constant version, which also lets us static-assert the no-
  // tail invariant.
  const int head_dim = HEAD_DIM_T >= 0 ? HEAD_DIM_T : head_dim_arg;
  const int head_dim_full = head_dim & ~15;
  const int head_dim_tail = head_dim - head_dim_full;
  const __mmask16 head_dim_tail_mask =
      head_dim_tail
          ? static_cast<__mmask16>((1u << head_dim_tail) - 1u)
          : static_cast<__mmask16>(0);

  for (int i = 0; i < seq_len; ++i) {
    const int t_q = seq_start + i;
    const float* qi =
        q + (static_cast<long>(t_q) * num_heads + h) * head_dim;

    // Pass 1: scores[j] = qi · kj for each j; track running max.
    // 16 independent zmm accumulators per j-chunk. Inner head_dim loop
    // loads q_chunk once and reuses it across the 16 K-row chunks.
    __m512 v_max = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    int j = 0;
    for (; j + 16 <= seq_len; j += 16) {
      __m512 a0  = _mm512_setzero_ps();
      __m512 a1  = _mm512_setzero_ps();
      __m512 a2  = _mm512_setzero_ps();
      __m512 a3  = _mm512_setzero_ps();
      __m512 a4  = _mm512_setzero_ps();
      __m512 a5  = _mm512_setzero_ps();
      __m512 a6  = _mm512_setzero_ps();
      __m512 a7  = _mm512_setzero_ps();
      __m512 a8  = _mm512_setzero_ps();
      __m512 a9  = _mm512_setzero_ps();
      __m512 a10 = _mm512_setzero_ps();
      __m512 a11 = _mm512_setzero_ps();
      __m512 a12 = _mm512_setzero_ps();
      __m512 a13 = _mm512_setzero_ps();
      __m512 a14 = _mm512_setzero_ps();
      __m512 a15 = _mm512_setzero_ps();
      const long row_stride =
          static_cast<long>(num_heads) * head_dim;
      const float* kj_base =
          k + (static_cast<long>(seq_start + j) * num_heads + h) * head_dim;

      int d = 0;
      for (; d + 16 <= head_dim; d += 16) {
        __m512 qc = _mm512_loadu_ps(qi + d);
        a0  = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base +  0 * row_stride + d), a0);
        a1  = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base +  1 * row_stride + d), a1);
        a2  = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base +  2 * row_stride + d), a2);
        a3  = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base +  3 * row_stride + d), a3);
        a4  = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base +  4 * row_stride + d), a4);
        a5  = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base +  5 * row_stride + d), a5);
        a6  = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base +  6 * row_stride + d), a6);
        a7  = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base +  7 * row_stride + d), a7);
        a8  = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base +  8 * row_stride + d), a8);
        a9  = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base +  9 * row_stride + d), a9);
        a10 = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base + 10 * row_stride + d), a10);
        a11 = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base + 11 * row_stride + d), a11);
        a12 = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base + 12 * row_stride + d), a12);
        a13 = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base + 13 * row_stride + d), a13);
        a14 = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base + 14 * row_stride + d), a14);
        a15 = _mm512_fmadd_ps(qc, _mm512_loadu_ps(kj_base + 15 * row_stride + d), a15);
      }
      if (head_dim_tail) {
        __m512 qc = _mm512_maskz_loadu_ps(head_dim_tail_mask, qi + d);
        a0  = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base +  0 * row_stride + d), a0);
        a1  = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base +  1 * row_stride + d), a1);
        a2  = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base +  2 * row_stride + d), a2);
        a3  = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base +  3 * row_stride + d), a3);
        a4  = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base +  4 * row_stride + d), a4);
        a5  = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base +  5 * row_stride + d), a5);
        a6  = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base +  6 * row_stride + d), a6);
        a7  = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base +  7 * row_stride + d), a7);
        a8  = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base +  8 * row_stride + d), a8);
        a9  = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base +  9 * row_stride + d), a9);
        a10 = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base + 10 * row_stride + d), a10);
        a11 = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base + 11 * row_stride + d), a11);
        a12 = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base + 12 * row_stride + d), a12);
        a13 = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base + 13 * row_stride + d), a13);
        a14 = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base + 14 * row_stride + d), a14);
        a15 = _mm512_fmadd_ps(qc, _mm512_maskz_loadu_ps(head_dim_tail_mask, kj_base + 15 * row_stride + d), a15);
      }
      // Horizontally reduce each accumulator and pack 16 scalars into
      // one zmm so we update v_max with a single op (and so the scores
      // store doesn't round-trip through L1 before the v_max read).
      __m512 scores_chunk = _mm512_set_ps(
          _mm512_reduce_add_ps(a15), _mm512_reduce_add_ps(a14),
          _mm512_reduce_add_ps(a13), _mm512_reduce_add_ps(a12),
          _mm512_reduce_add_ps(a11), _mm512_reduce_add_ps(a10),
          _mm512_reduce_add_ps(a9),  _mm512_reduce_add_ps(a8),
          _mm512_reduce_add_ps(a7),  _mm512_reduce_add_ps(a6),
          _mm512_reduce_add_ps(a5),  _mm512_reduce_add_ps(a4),
          _mm512_reduce_add_ps(a3),  _mm512_reduce_add_ps(a2),
          _mm512_reduce_add_ps(a1),  _mm512_reduce_add_ps(a0));
      _mm512_storeu_ps(scores + j, scores_chunk);
      v_max = _mm512_max_ps(v_max, scores_chunk);
    }
    // j-tail: < 16 K-rows remain. Per-position scalar dot product.
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
    // Output O held in zmm registers across the entire seq_len loop;
    // out_row is touched exactly once (at the end). Max head_dim is 64
    // (= 4 chunks); we size for 8 as headroom and ignore unused entries.
    __m512 o0 = _mm512_setzero_ps();
    __m512 o1 = _mm512_setzero_ps();
    __m512 o2 = _mm512_setzero_ps();
    __m512 o3 = _mm512_setzero_ps();
    __m512 o4 = _mm512_setzero_ps();
    __m512 o5 = _mm512_setzero_ps();
    __m512 o6 = _mm512_setzero_ps();
    __m512 o7 = _mm512_setzero_ps();
    __m512 o_tail = _mm512_setzero_ps();
    const int o_full_chunks = head_dim_full / 16;
    static_assert(kOMaxChunks == 8,
                  "If you increase kOMaxChunks, add more O registers below");

    for (int jj = 0; jj < seq_len; ++jj) {
      const float w_scalar = scores[jj] * inv_sum;
      if (w_scalar == 0.0f) continue;
      const __m512 w = _mm512_set1_ps(w_scalar);
      const int t_k = seq_start + jj;
      const float* vj =
          v + (static_cast<long>(t_k) * num_heads + h) * head_dim;
      // Unrolled FMA into the O registers — compiler picks up which ones
      // it actually needs based on o_full_chunks at compile time only when
      // inlined; we branch explicitly to avoid the unused-FMA stalls.
      if (o_full_chunks > 0) o0 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vj +  0), o0);
      if (o_full_chunks > 1) o1 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vj + 16), o1);
      if (o_full_chunks > 2) o2 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vj + 32), o2);
      if (o_full_chunks > 3) o3 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vj + 48), o3);
      if (o_full_chunks > 4) o4 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vj + 64), o4);
      if (o_full_chunks > 5) o5 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vj + 80), o5);
      if (o_full_chunks > 6) o6 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vj + 96), o6);
      if (o_full_chunks > 7) o7 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vj + 112), o7);
      if (head_dim_tail) {
        __m512 vv = _mm512_maskz_loadu_ps(head_dim_tail_mask,
                                           vj + head_dim_full);
        o_tail = _mm512_fmadd_ps(w, vv, o_tail);
      }
    }

    float* out_row =
        out + (static_cast<long>(t_q) * num_heads + h) * head_dim;
    if (o_full_chunks > 0) _mm512_storeu_ps(out_row +  0, o0);
    if (o_full_chunks > 1) _mm512_storeu_ps(out_row + 16, o1);
    if (o_full_chunks > 2) _mm512_storeu_ps(out_row + 32, o2);
    if (o_full_chunks > 3) _mm512_storeu_ps(out_row + 48, o3);
    if (o_full_chunks > 4) _mm512_storeu_ps(out_row + 64, o4);
    if (o_full_chunks > 5) _mm512_storeu_ps(out_row + 80, o5);
    if (o_full_chunks > 6) _mm512_storeu_ps(out_row + 96, o6);
    if (o_full_chunks > 7) _mm512_storeu_ps(out_row + 112, o7);
    if (head_dim_tail) {
      _mm512_mask_storeu_ps(out_row + head_dim_full,
                             head_dim_tail_mask, o_tail);
    }
  }
}

inline void RunOneHeadAvx512(const float* q, const float* k, const float* v,
                              int seq_start, int seq_len, int h, int num_heads,
                              int head_dim, float* out, float* scores) {
  // Compile-time dispatch by head_dim for the ESM-2 production sizes.
  // Dispatch overhead is one untaken branch; the body is fully inlined.
  switch (head_dim) {
    case 16:
      return RunOneHeadAvx512Impl<16>(q, k, v, seq_start, seq_len, h,
                                       num_heads, head_dim, out, scores);
    case 32:
      return RunOneHeadAvx512Impl<32>(q, k, v, seq_start, seq_len, h,
                                       num_heads, head_dim, out, scores);
    case 64:
      return RunOneHeadAvx512Impl<64>(q, k, v, seq_start, seq_len, h,
                                       num_heads, head_dim, out, scores);
    default:
      // 24 (35M), 17, 3, ... — anything that doesn't have a fast
      // specialization. The dynamic path handles masked tails too.
      return RunOneHeadAvx512Impl<-1>(q, k, v, seq_start, seq_len, h,
                                       num_heads, head_dim, out, scores);
  }
}

// BF16 variant of the attention kernel — uses VDPBF16PS for Pass 1,
// FP32 fallback for Pass 2 (softmax) and Pass 3 (S · V). The plan
// section S7 specifies the AMX TDPBF16PS path; this simpler AVX-512
// BF16 variant captures the same throughput gain (32 BF16 muls per
// instruction = 2× FP32 FMA throughput, half the K-tile bytes) without
// the AMX tile-state complexity. PPPL gate still required before this
// becomes default-on; for now ESM_AMX_ATTENTION=on opts in per the
// plan's safety mitigation.
//
// Per-(b, h) plan:
//   1. Pre-convert Q[L, h, :] and K[L, h, :] to BF16 in thread_local
//      scratch buffers (each [L * head_dim] BF16 = half the FP32 size).
//      The conversion uses _mm512_cvtne2ps_pbh which packs 32 FP32
//      into 32 BF16 in one zmm.
//   2. Pass 1: same 16-zmm-accumulator structure as Slice 4/6 but the
//      inner head_dim chunk is 32 (BF16) instead of 16 (FP32). Each
//      _mm512_dpbf16_ps does 32 BF16 mul-adds → 16 FP32 accumulator
//      lanes (DEST[i] += SRC1.bf16[2i]*SRC2.bf16[2i] +
//      SRC1.bf16[2i+1]*SRC2.bf16[2i+1]) which is exactly the partial
//      dot of two head_dim values.
//   3. Pass 2 (softmax) unchanged — FP32 throughout.
//   4. Pass 3 (S · V) unchanged — the per-output FMA is FP32 scalar
//      weight × FP32 V, no BF16 throughput win available.
//
// CLAUDE.md: FP32 accumulator preserved inside attention (the _ps
// accumulator type in _mm512_dpbf16_ps is exactly that). These helpers
// live in the SAME anonymous namespace as the FP32 attention helpers
// above — opening a nested anonymous would shadow `AttentionVarlenAvx512`
// when we forward-call it as the BF16 fallback path.

thread_local std::vector<std::uint16_t> g_attn_q_bf16;
thread_local std::vector<std::uint16_t> g_attn_k_bf16;

inline void ConvertPackedToBf16(const float* src, std::uint16_t* dst,
                                 int n) {
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    __m512 lo = _mm512_loadu_ps(src + i);
    __m512 hi = _mm512_loadu_ps(src + i + 16);
    __m512bh bh = _mm512_cvtne2ps_pbh(hi, lo);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i),
                         reinterpret_cast<__m512i>(bh));
  }
  if (i + 16 <= n) {
    __m512 v = _mm512_loadu_ps(src + i);
    __m256bh half = _mm512_cvtneps_pbh(v);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                         reinterpret_cast<__m256i>(half));
    i += 16;
  }
  if (i < n) {
    // Scalar tail. Round-to-nearest-even FP32 → BF16: shift bit 16
    // off the top, with a +0.5-ulp bias.
    for (; i < n; ++i) {
      std::uint32_t bits;
      const float f = src[i];
      std::memcpy(&bits, &f, sizeof(bits));
      const std::uint32_t bias = 0x7FFFu + ((bits >> 16) & 1u);
      bits += bias;
      dst[i] = static_cast<std::uint16_t>(bits >> 16);
    }
  }
}

template <int HEAD_DIM_T>
inline void RunOneHeadAvx512Bf16Impl(const std::uint16_t* q_bf16,
                                      const std::uint16_t* k_bf16,
                                      const float* v, int seq_start,
                                      int seq_len, int h, int num_heads,
                                      int head_dim_arg, float* out,
                                      float* scores) {
  const int head_dim = HEAD_DIM_T >= 0 ? HEAD_DIM_T : head_dim_arg;
  // BF16 path supports head_dim multiples of 32 only — the BF16 zmm
  // holds 32 values per chunk. Inner-loop tail handling for
  // non-multiples isn't worth the complexity in the BF16 path; the
  // caller routes such heads to the FP32 kernel.
  const int head_dim_chunks_bf16 = head_dim / 32;
  for (int i = 0; i < seq_len; ++i) {
    const int t_q = seq_start + i;
    // Q for this query is contiguous head_dim BF16 in q_bf16 scratch
    // (we pre-packed it as [L, head_dim] so per-h indexing is just
    // i * head_dim).
    const std::uint16_t* qi_bf16 = q_bf16 + static_cast<long>(i) * head_dim;

    // Pass 1: scores[j] = qi · kj for each j, via VDPBF16PS.
    __m512 v_max = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    int j = 0;
    for (; j + 16 <= seq_len; j += 16) {
      __m512 a0  = _mm512_setzero_ps();
      __m512 a1  = _mm512_setzero_ps();
      __m512 a2  = _mm512_setzero_ps();
      __m512 a3  = _mm512_setzero_ps();
      __m512 a4  = _mm512_setzero_ps();
      __m512 a5  = _mm512_setzero_ps();
      __m512 a6  = _mm512_setzero_ps();
      __m512 a7  = _mm512_setzero_ps();
      __m512 a8  = _mm512_setzero_ps();
      __m512 a9  = _mm512_setzero_ps();
      __m512 a10 = _mm512_setzero_ps();
      __m512 a11 = _mm512_setzero_ps();
      __m512 a12 = _mm512_setzero_ps();
      __m512 a13 = _mm512_setzero_ps();
      __m512 a14 = _mm512_setzero_ps();
      __m512 a15 = _mm512_setzero_ps();
      const long k_row_stride = static_cast<long>(head_dim);
      const std::uint16_t* kj_base =
          k_bf16 + static_cast<long>(j) * k_row_stride;

      for (int d = 0; d < head_dim_chunks_bf16; ++d) {
        const int d_off = d * 32;
        __m512bh qc = reinterpret_cast<__m512bh>(
            _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(qi_bf16 + d_off)));
#define LOAD_KJ(N)                                                         \
  __m512bh kc##N = reinterpret_cast<__m512bh>(_mm512_loadu_si512(           \
      reinterpret_cast<const __m512i*>(kj_base + N * k_row_stride + d_off)))
        LOAD_KJ(0);  LOAD_KJ(1);  LOAD_KJ(2);  LOAD_KJ(3);
        LOAD_KJ(4);  LOAD_KJ(5);  LOAD_KJ(6);  LOAD_KJ(7);
        LOAD_KJ(8);  LOAD_KJ(9);  LOAD_KJ(10); LOAD_KJ(11);
        LOAD_KJ(12); LOAD_KJ(13); LOAD_KJ(14); LOAD_KJ(15);
#undef LOAD_KJ
        a0  = _mm512_dpbf16_ps(a0,  qc, kc0);
        a1  = _mm512_dpbf16_ps(a1,  qc, kc1);
        a2  = _mm512_dpbf16_ps(a2,  qc, kc2);
        a3  = _mm512_dpbf16_ps(a3,  qc, kc3);
        a4  = _mm512_dpbf16_ps(a4,  qc, kc4);
        a5  = _mm512_dpbf16_ps(a5,  qc, kc5);
        a6  = _mm512_dpbf16_ps(a6,  qc, kc6);
        a7  = _mm512_dpbf16_ps(a7,  qc, kc7);
        a8  = _mm512_dpbf16_ps(a8,  qc, kc8);
        a9  = _mm512_dpbf16_ps(a9,  qc, kc9);
        a10 = _mm512_dpbf16_ps(a10, qc, kc10);
        a11 = _mm512_dpbf16_ps(a11, qc, kc11);
        a12 = _mm512_dpbf16_ps(a12, qc, kc12);
        a13 = _mm512_dpbf16_ps(a13, qc, kc13);
        a14 = _mm512_dpbf16_ps(a14, qc, kc14);
        a15 = _mm512_dpbf16_ps(a15, qc, kc15);
      }
      __m512 scores_chunk = _mm512_set_ps(
          _mm512_reduce_add_ps(a15), _mm512_reduce_add_ps(a14),
          _mm512_reduce_add_ps(a13), _mm512_reduce_add_ps(a12),
          _mm512_reduce_add_ps(a11), _mm512_reduce_add_ps(a10),
          _mm512_reduce_add_ps(a9),  _mm512_reduce_add_ps(a8),
          _mm512_reduce_add_ps(a7),  _mm512_reduce_add_ps(a6),
          _mm512_reduce_add_ps(a5),  _mm512_reduce_add_ps(a4),
          _mm512_reduce_add_ps(a3),  _mm512_reduce_add_ps(a2),
          _mm512_reduce_add_ps(a1),  _mm512_reduce_add_ps(a0));
      _mm512_storeu_ps(scores + j, scores_chunk);
      v_max = _mm512_max_ps(v_max, scores_chunk);
    }
    // j-tail: < 16 K-rows. Use scalar BF16 dot (slow but rare; called
    // only on the seq_len tail, which for ESM-2 typical L=256 is empty).
    for (; j < seq_len; ++j) {
      const std::uint16_t* kj =
          k_bf16 + static_cast<long>(j) * static_cast<long>(head_dim);
      __m512 acc = _mm512_setzero_ps();
      for (int d = 0; d < head_dim_chunks_bf16; ++d) {
        const int d_off = d * 32;
        __m512bh qc = reinterpret_cast<__m512bh>(_mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(qi_bf16 + d_off)));
        __m512bh kc = reinterpret_cast<__m512bh>(_mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(kj + d_off)));
        acc = _mm512_dpbf16_ps(acc, qc, kc);
      }
      scores[j] = _mm512_reduce_add_ps(acc);
    }
    float max_score = _mm512_reduce_max_ps(v_max);
    for (int jj = (seq_len & ~15); jj < seq_len; ++jj) {
      if (scores[jj] > max_score) max_score = scores[jj];
    }

    // Pass 2: softmax — same as the FP32 kernel.
    const __m512 v_max_bcast = _mm512_set1_ps(max_score);
    __m512 v_sum = _mm512_setzero_ps();
    j = 0;
    for (; j + 16 <= seq_len; j += 16) {
      __m512 s = _mm512_sub_ps(_mm512_loadu_ps(scores + j), v_max_bcast);
      __m512 e = ExpAvx512Attn(s);
      _mm512_storeu_ps(scores + j, e);
      v_sum = _mm512_add_ps(v_sum, e);
    }
    float sum = _mm512_reduce_add_ps(v_sum);
    for (; j < seq_len; ++j) {
      const float e = std::exp(scores[j] - max_score);
      scores[j] = e;
      sum += e;
    }
    const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;

    // Pass 3: out_row = sum_j (scores[j] * inv_sum) * V[j]. FP32 V.
    __m512 o0 = _mm512_setzero_ps();
    __m512 o1 = _mm512_setzero_ps();
    __m512 o2 = _mm512_setzero_ps();
    __m512 o3 = _mm512_setzero_ps();
    const int o_full_chunks = head_dim / 16;
    for (int jj = 0; jj < seq_len; ++jj) {
      const float w_scalar = scores[jj] * inv_sum;
      if (w_scalar == 0.0f) continue;
      const __m512 w = _mm512_set1_ps(w_scalar);
      const int t_k = seq_start + jj;
      const float* vj =
          v + (static_cast<long>(t_k) * num_heads + h) * head_dim;
      if (o_full_chunks > 0) o0 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vj +  0), o0);
      if (o_full_chunks > 1) o1 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vj + 16), o1);
      if (o_full_chunks > 2) o2 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vj + 32), o2);
      if (o_full_chunks > 3) o3 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vj + 48), o3);
    }
    float* out_row =
        out + (static_cast<long>(t_q) * num_heads + h) * head_dim;
    if (o_full_chunks > 0) _mm512_storeu_ps(out_row +  0, o0);
    if (o_full_chunks > 1) _mm512_storeu_ps(out_row + 16, o1);
    if (o_full_chunks > 2) _mm512_storeu_ps(out_row + 32, o2);
    if (o_full_chunks > 3) _mm512_storeu_ps(out_row + 48, o3);
  }
}

inline void RunOneHeadAvx512Bf16(const std::uint16_t* q_bf16,
                                  const std::uint16_t* k_bf16,
                                  const float* v, int seq_start,
                                  int seq_len, int h, int num_heads,
                                  int head_dim, float* out, float* scores) {
  switch (head_dim) {
    case 32:
      return RunOneHeadAvx512Bf16Impl<32>(q_bf16, k_bf16, v, seq_start,
                                           seq_len, h, num_heads, head_dim,
                                           out, scores);
    case 64:
      return RunOneHeadAvx512Bf16Impl<64>(q_bf16, k_bf16, v, seq_start,
                                           seq_len, h, num_heads, head_dim,
                                           out, scores);
    default:
      return RunOneHeadAvx512Bf16Impl<-1>(q_bf16, k_bf16, v, seq_start,
                                           seq_len, h, num_heads, head_dim,
                                           out, scores);
  }
}

bool ReadBf16AttentionEnvOnce() {
  const char* v = std::getenv("ESM_AMX_ATTENTION");
  if (!v || !*v) return false;
  const std::string s = v;
  return s == "on" || s == "1" || s == "true";
}

}  // namespace

// Forward declaration: AttentionVarlenAvx512Bf16's head_dim-not-multiple-
// of-32 fallback delegates to the FP32 path, which is defined later in
// this TU.
void AttentionVarlenAvx512(const float* q, const float* k, const float* v,
                            const int* cu_seqlens, int batch_size,
                            int num_heads, int head_dim, float* out);

void AttentionVarlenAvx512Bf16(const float* q, const float* k, const float* v,
                                const int* cu_seqlens, int batch_size,
                                int num_heads, int head_dim, float* out) {
  // BF16 path only supports head_dim multiples of 32 (the BF16 zmm
  // holds 32 values per chunk). For other sizes (e.g. 35M's dh=24),
  // fall back to the FP32 path.
  if ((head_dim & 31) != 0) {
    AttentionVarlenAvx512(q, k, v, cu_seqlens, batch_size, num_heads,
                          head_dim, out);
    return;
  }
  const int total_bh = batch_size * num_heads;
  int max_seq = 0;
  for (int b = 0; b < batch_size; ++b) {
    max_seq = std::max(max_seq, cu_seqlens[b + 1] - cu_seqlens[b]);
  }
  auto run = [&](int bh_begin, int bh_end) {
    if (g_attn_scores.size() <
        static_cast<std::size_t>(max_seq + 16)) {
      g_attn_scores.assign(static_cast<std::size_t>(max_seq + 16), 0.0f);
    }
    const std::size_t qk_scratch =
        static_cast<std::size_t>(max_seq) * head_dim;
    if (g_attn_q_bf16.size() < qk_scratch)
      g_attn_q_bf16.assign(qk_scratch, 0);
    if (g_attn_k_bf16.size() < qk_scratch)
      g_attn_k_bf16.assign(qk_scratch, 0);
    float* scores = g_attn_scores.data();
    for (int bh = bh_begin; bh < bh_end; ++bh) {
      const int b = bh / num_heads;
      const int h = bh % num_heads;
      const int seq_start = cu_seqlens[b];
      const int seq_end = cu_seqlens[b + 1];
      const int seq_len = seq_end - seq_start;
      if (seq_len <= 0) continue;

      // Pack Q[seq_start..seq_end, h, :] and K[same] to contiguous
      // BF16 scratch [seq_len, head_dim]. Strided FP32 → packed BF16.
      for (int i = 0; i < seq_len; ++i) {
        const int t = seq_start + i;
        ConvertPackedToBf16(
            q + (static_cast<long>(t) * num_heads + h) * head_dim,
            g_attn_q_bf16.data() + static_cast<long>(i) * head_dim,
            head_dim);
        ConvertPackedToBf16(
            k + (static_cast<long>(t) * num_heads + h) * head_dim,
            g_attn_k_bf16.data() + static_cast<long>(i) * head_dim,
            head_dim);
      }

      RunOneHeadAvx512Bf16(g_attn_q_bf16.data(), g_attn_k_bf16.data(), v,
                            seq_start, seq_len, h, num_heads, head_dim,
                            out, scores);
    }
  };
  if (total_bh > 1 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(0, total_bh, /*grain=*/1, run);
  } else {
    run(0, total_bh);
  }
}

void AttentionVarlenAvx512(const float* q, const float* k, const float* v,
                            const int* cu_seqlens, int batch_size,
                            int num_heads, int head_dim, float* out) {
  static const bool bf16_attention_on = ReadBf16AttentionEnvOnce();
  if (bf16_attention_on && (head_dim & 31) == 0) {
    return AttentionVarlenAvx512Bf16(q, k, v, cu_seqlens, batch_size,
                                      num_heads, head_dim, out);
  }
  // Parallelize across (batch × head) — 33-layer ESM-2 at B=8 H=20 gives
  // 160 work items, enough to keep all 22 cores fed. The previous
  // batch-only parallelism left 14 of 22 cores idle every layer; on the
  // gate-machine profile this section was 30 % of the 650M forward.
  const int total_bh = batch_size * num_heads;
  // Compute max_seq across batches once; each worker's scores scratch
  // needs to fit the largest sequence.
  int max_seq = 0;
  for (int b = 0; b < batch_size; ++b) {
    max_seq = std::max(max_seq, cu_seqlens[b + 1] - cu_seqlens[b]);
  }
  auto run = [&](int bh_begin, int bh_end) {
    if (g_attn_scores.size() <
        static_cast<std::size_t>(max_seq + 16)) {
      g_attn_scores.assign(static_cast<std::size_t>(max_seq + 16), 0.0f);
    }
    float* scores = g_attn_scores.data();
    for (int bh = bh_begin; bh < bh_end; ++bh) {
      const int b = bh / num_heads;
      const int h = bh % num_heads;
      const int seq_start = cu_seqlens[b];
      const int seq_end = cu_seqlens[b + 1];
      const int seq_len = seq_end - seq_start;
      if (seq_len <= 0) continue;
      RunOneHeadAvx512(q, k, v, seq_start, seq_len, h, num_heads, head_dim,
                       out, scores);
    }
  };
  if (total_bh > 1 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(0, total_bh, /*grain=*/1, run);
  } else {
    run(0, total_bh);
  }
}

#endif  // ESM_KERNEL_AVX512

}  // namespace esm::kernels
