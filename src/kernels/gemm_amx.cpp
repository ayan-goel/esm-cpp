#include "esm_cpp/kernels.h"
#include "esm_cpp/profile.h"
#include "esm_cpp/quant.h"

// AMX-INT8 path uses TDPBUSD with on-chip 16-row × 64-byte tiles for
// W8A8 GEMM. SPR's AMX-INT8 issues TDPBUSD at one tile pair per cycle
// (16 m × 16 n × 64 k = 16K macs in a single instruction), ~67 TOPS/socket
// peak — roughly 8× the AVX-512 VPDPBUSD peak.
//
// System headers at file scope (per the gemm_int8.cpp lesson — leaks of
// C-stdlib symbols into namespace esm::kernels broke libstdc++).
#ifdef ESM_KERNEL_AMX
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
// arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) — gates per-thread
// XSAVE-permitted state to include the 8 KB of AMX tile data. Required on
// Linux ≥ 5.16; earlier kernels reject the request and LinearAmx falls
// back to LinearVnni at runtime.
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#endif  // __linux__

#include "esm_cpp/thread_pool.h"
#endif  // ESM_KERNEL_AMX

namespace esm::kernels {

#ifdef ESM_KERNEL_AMX

// Forward decl: AMX falls back to the AVX-512 VNNI path for any shape
// the tile-based microkernel can't handle (M < 32, K not multiple of 64,
// N not multiple of 32, or AMX permission denied at process start). The
// VNNI symbol lives in the sibling esm_cpp_kernels_avx512 OBJECT lib;
// linkage resolves through esm_cpp_core.
void LinearVnni(const float* A, const esm::quant::QuantizedTensor& W,
                const float* bias, float* C, int M, int N, int K);

// Shared parallel activation prefix (absmax + quantize). Defined in
// gemm_int8.cpp's AVX-512 TU and reused here so both INT8 paths get the
// same parallelization without duplicating the kernel body.
void QuantizeActPrefixAvx512(const float* A, std::size_t MK,
                              std::uint8_t* a_u8, float* act_scale_out);

namespace {

// AMX tile config layout, fixed by Intel ISA. palette_id=1 enables INT8
// tile semantics; rows[i] and colsb[i] dimension each of 8 tiles. We
// configure all 8 tiles to the maximum 16×64 footprint for our 32×32
// microkernel (2 A tiles + 2 B tiles + 4 C tiles).
struct alignas(64) TileConfig {
  std::uint8_t palette_id;
  std::uint8_t start_row;
  std::uint8_t reserved_0[14];
  std::uint16_t colsb[16];
  std::uint8_t rows[16];
};

#if defined(__linux__)
bool RequestAmxXcompPerm() {
  const long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
                           XFEATURE_XTILEDATA);
  return rc == 0;
}
#else
bool RequestAmxXcompPerm() { return false; }
#endif

// Once-per-process: ask the kernel for AMX tile-data state. If granted,
// LinearAmx is enabled. Cached via call_once so subsequent forward calls
// pay nothing.
bool AmxProcessReady() {
  static std::once_flag flag;
  static std::atomic<bool> ok{false};
  std::call_once(flag, []() { ok.store(RequestAmxXcompPerm()); });
  return ok.load();
}

// Once-per-thread: load tile config so this worker can issue AMX
// instructions. Per the ISA, every thread that uses AMX must call
// `_tile_loadconfig` before the first TDPBUSD; the state is otherwise
// undefined and the instruction faults.
inline void EnsureThreadTileConfig() {
  thread_local bool configured = false;
  if (configured) return;
  alignas(64) TileConfig cfg{};
  cfg.palette_id = 1;  // INT8
  for (int i = 0; i < 8; ++i) {
    cfg.rows[i] = 16;
    cfg.colsb[i] = 64;
  }
  _tile_loadconfig(&cfg);
  configured = true;
}

// Bottom half of the GEMM: corrects the s32 accumulator by the zero-
// point shift (subtract 128 × col_sum[n]), folds in the per-channel
// weight scale and the per-tensor activation scale, adds bias, stores
// 16 FP32 outputs. Shared shape with the AVX-512 VNNI finalize.
inline void FinalizeStore16Amx(__m512i acc_raw, const std::int32_t* col_sum,
                                const float* w_scale, const float* bias,
                                float act_scale, float* out_row) {
  const __m512i cs = _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(col_sum));
  const __m512i corrected =
      _mm512_sub_epi32(acc_raw, _mm512_slli_epi32(cs, 7));
  __m512 fp = _mm512_cvtepi32_ps(corrected);
  __m512 ws = _mm512_loadu_ps(w_scale);
  __m512 combined = _mm512_mul_ps(ws, _mm512_set1_ps(act_scale));
  __m512 b = _mm512_loadu_ps(bias);
  _mm512_storeu_ps(out_row, _mm512_fmadd_ps(fp, combined, b));
}

// Per-thread scratch for the 32×32 int32 accumulator block written by
// _tile_stored. Sized for one microkernel call; reused across calls.
// alignas must lead the decl-specifiers — clang and gcc both reject
// `thread_local alignas(...)` ordering.
alignas(64) thread_local std::int32_t g_amx_c_buf[32 * 32];
// Per-thread scratch for the quantized activations.
thread_local std::vector<std::uint8_t> g_amx_a_u8;

// 32×32 microkernel: 4 C tiles, 2 A tiles, 2 B tiles. K must be a
// multiple of 64 (one full tile-K). Caller has gated all of that. K_pad
// equals K here (we only run when K % 64 == 0) so packed_vnni reads
// straight through with stride 64.
void Kernel32x32(const std::uint8_t* a_block, const std::int8_t* w_panel0,
                  const std::int8_t* w_panel1, const float* w_scale,
                  const std::int32_t* col_sum, const float* bias,
                  float act_scale, int K, float* c_base, int N) {
  _tile_zero(4);
  _tile_zero(5);
  _tile_zero(6);
  _tile_zero(7);
  for (int kb = 0; kb < K; kb += 64) {
    // A tiles: rows [0, 16) and [16, 32) of a_block, each 64 K-bytes
    // wide. Stride is K (the activation matrix row stride).
    _tile_loadd(0, a_block + 0 * static_cast<long>(K) + kb,
                 static_cast<long>(K));
    _tile_loadd(1, a_block + 16 * static_cast<long>(K) + kb,
                 static_cast<long>(K));
    // B tiles: each 16-N panel's 64-K slice. packed_vnni stores 16
    // K-groups of 64 bytes per panel, so the AMX tile stride (col-bytes
    // per row) is exactly 64.
    _tile_loadd(2,
                 w_panel0 + static_cast<long>(kb) * 16,
                 /*stride=*/64);
    _tile_loadd(3,
                 w_panel1 + static_cast<long>(kb) * 16,
                 /*stride=*/64);
    _tile_dpbusd(4, 0, 2);
    _tile_dpbusd(5, 0, 3);
    _tile_dpbusd(6, 1, 2);
    _tile_dpbusd(7, 1, 3);
  }
  // Drain s32 accumulators to per-thread scratch (4 quadrants of 16×16).
  std::int32_t* buf = g_amx_c_buf;
  _tile_stored(4, buf + 0 * 32 + 0, 32 * sizeof(std::int32_t));
  _tile_stored(5, buf + 0 * 32 + 16, 32 * sizeof(std::int32_t));
  _tile_stored(6, buf + 16 * 32 + 0, 32 * sizeof(std::int32_t));
  _tile_stored(7, buf + 16 * 32 + 16, 32 * sizeof(std::int32_t));

  alignas(64) static const float kZeroBias[16] = {0};
  const float* b_lo = bias ? bias : kZeroBias;
  const float* b_hi = bias ? bias + 16 : kZeroBias;
  for (int r = 0; r < 32; ++r) {
    const __m512i acc_lo = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(buf + r * 32));
    const __m512i acc_hi = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(buf + r * 32 + 16));
    FinalizeStore16Amx(acc_lo, col_sum, w_scale, b_lo, act_scale,
                        c_base + r * N);
    FinalizeStore16Amx(acc_hi, col_sum + 16, w_scale + 16, b_hi, act_scale,
                        c_base + r * N + 16);
  }
}

}  // namespace

void LinearAmx(const float* A, const esm::quant::QuantizedTensor& W,
                const float* bias, float* C, int M, int N, int K) {
  if (K <= 0 || M <= 0 || N <= 0) {
    if (M > 0 && N > 0 && C) {
      std::memset(C, 0,
                  static_cast<std::size_t>(M) * static_cast<std::size_t>(N) *
                      sizeof(float));
    }
    return;
  }
  // AMX gates: full 32×32 microkernel needs M ≥ 32, N % 32 == 0, K % 64
  // == 0, and the OS-granted XSAVE permission. Anything outside that goes
  // through LinearVnni (which itself routes to the multi-accumulator
  // Goto kernel for the same hot shapes).
  if (M < 32 || (N & 31) != 0 || (K & 63) != 0 || !AmxProcessReady()) {
    esm::profile::BumpCounter("amx_fallback_to_vnni");
    return LinearVnni(A, W, bias, C, M, N, K);
  }
  esm::profile::BumpCounter("amx_engaged");

  // Activation prefix: shared parallel absmax + quantize helper. Both the
  // VNNI and AMX paths consume the same u8 layout, so the prefix is
  // factored to QuantizeActPrefixAvx512 (defined in gemm_int8.cpp).
  const std::size_t MK = static_cast<std::size_t>(M) *
                          static_cast<std::size_t>(K);
  if (g_amx_a_u8.size() < MK) g_amx_a_u8.resize(MK);
  std::uint8_t* a_u8 = g_amx_a_u8.data();
  float act_scale = 1.0f;
  QuantizeActPrefixAvx512(A, MK, a_u8, &act_scale);

  const std::int8_t* w_packed = W.packed_vnni.data();
  const std::int32_t* col_sum = W.col_sum.data();
  const float* w_scale = W.per_channel_scales.data();

  // M main rectangle: rows [0, M_main) processed as M_main/32 row-blocks.
  // parallel_for's `grain` is a *minimum* chunk size, not an alignment —
  // its chunks can split mid-row-block. Parallelize over the row-block
  // count instead so every task always sees full 32-row blocks; this
  // avoids past-end A-tile loads on misaligned chunk boundaries (caused
  // SIGSEGV at M_main=2048 when 22 workers chunked into 94-row pieces).
  const int M_main = M & ~31;
  const int num_blocks = M_main / 32;
  auto run_blocks = [&](int block_begin, int block_end) {
    EnsureThreadTileConfig();
    for (int b = block_begin; b < block_end; ++b) {
      const int m = b * 32;
      const std::uint8_t* a_block =
          a_u8 + static_cast<long>(m) * static_cast<long>(K);
      float* c_row = C + static_cast<long>(m) * static_cast<long>(N);
      for (int nb = 0; nb < N; nb += 32) {
        const std::int8_t* w_panel0 =
            w_packed + static_cast<long>(nb) * static_cast<long>(K);
        const std::int8_t* w_panel1 =
            w_packed + static_cast<long>(nb + 16) * static_cast<long>(K);
        Kernel32x32(a_block, w_panel0, w_panel1, w_scale + nb, col_sum + nb,
                     bias ? bias + nb : nullptr, act_scale, K, c_row + nb, N);
      }
    }
  };
  if (num_blocks > 0 && !esm::InGlobalPoolWorker()) {
    esm::GlobalPool().parallel_for(0, num_blocks, /*grain=*/1, run_blocks);
  } else if (num_blocks > 0) {
    run_blocks(0, num_blocks);
  }

  // M-tail (< 32 rows remaining): hand to LinearVnni against the SAME
  // already-quantized activations. We can't call the public LinearVnni
  // because it re-quantizes; instead we model the tail as a separate
  // Linear call against rows [M_main, M). The re-quantize cost is small
  // (at most 31 rows of activations) compared to skipping AMX entirely.
  if (M_main < M) {
    LinearVnni(A + static_cast<long>(M_main) * static_cast<long>(K),
                W, bias, C + static_cast<long>(M_main) * static_cast<long>(N),
                M - M_main, N, K);
  }
}

#endif  // ESM_KERNEL_AMX

}  // namespace esm::kernels
