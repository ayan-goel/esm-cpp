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

}  // namespace

void AttentionVarlenAvx512(const float* q, const float* k, const float* v,
                            const int* cu_seqlens, int batch_size,
                            int num_heads, int head_dim, float* out) {
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
