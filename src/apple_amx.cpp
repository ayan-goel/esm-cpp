#include "esm_cpp/apple_amx.h"

#ifdef ESM_APPLE_AMX_AVAILABLE

#include <Accelerate/Accelerate.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Per-Linear BNNSGraph runtime: load a compiled .mlmodelc, cache the context +
// workspace at Model::load, execute per forward via SetDynamicShapes(M) +
// ContextExecute. Allocate-once discipline (CLAUDE.md): zero new allocations
// in the forward path. Deprecation warnings for BNNSMatMul are suppressed at
// the call sites that need them; this TU itself uses only the BNNSGraph API
// surface (the non-deprecated modern path, macOS 15+).

namespace esm {

namespace {

inline bnns_graph_context_t MakeCtx(void* data, std::size_t size) {
  bnns_graph_context_t c{};
  c.data = data;
  c.size = size;
  return c;
}

// One workspace buffer per thread, shared across all AppleAmxContext::Execute
// calls. Grows lazily to fit the largest workspace any context demands at the
// current M; subsequent calls are zero-alloc once it has converged. Same
// pattern as gemm_int8.cpp's g_a_s8.
thread_local std::vector<char> g_workspace;

}  // namespace

std::unique_ptr<AppleAmxContext> AppleAmxContext::LoadFromDir(
    const std::string& dir, int K, int N) {
  if (K <= 0 || N <= 0) return nullptr;
  auto opts = BNNSGraphCompileOptionsMakeDefault();
  bnns_graph_t g = BNNSGraphCompileFromFile(dir.c_str(), nullptr, opts);
  BNNSGraphCompileOptionsDestroy(opts);
  if (g.data == nullptr) return nullptr;

  bnns_graph_context_t ctx = BNNSGraphContextMake(g);
  if (ctx.data == nullptr) {
    std::free(g.data);
    return nullptr;
  }

  const std::size_t argc = BNNSGraphGetArgumentCount(g, nullptr);
  const std::size_t xi = BNNSGraphGetArgumentPosition(g, nullptr, "x");
  const std::size_t oi = BNNSGraphGetArgumentPosition(g, nullptr, "out");
  constexpr std::size_t kBad = static_cast<std::size_t>(-1);
  if (argc == kBad || xi == kBad || oi == kBad || argc < 2) {
    BNNSGraphContextDestroy(ctx);
    std::free(g.data);
    return nullptr;
  }

  auto out = std::unique_ptr<AppleAmxContext>(new AppleAmxContext());
  out->graph_data_ = g.data;
  out->graph_size_ = g.size;
  out->ctx_data_ = ctx.data;
  out->ctx_size_ = ctx.size;
  out->x_pos_ = xi;
  out->out_pos_ = oi;
  out->arg_count_ = argc;
  out->k_ = K;
  out->n_ = N;
  return out;
}

AppleAmxContext::~AppleAmxContext() {
  if (ctx_data_ != nullptr) {
    bnns_graph_context_t c = MakeCtx(ctx_data_, ctx_size_);
    BNNSGraphContextDestroy(c);
  }
  if (graph_data_ != nullptr) {
    std::free(graph_data_);
  }
}

Status AppleAmxContext::Execute(const float* in, float* out, int M) {
  if (in == nullptr || out == nullptr || M <= 0)
    return {StatusCode::kInvalidArgument, "apple_amx: bad Execute args"};

  // Set the input's full [M, K] shape — passing 0 for the static K does NOT
  // work in practice; BNNS needs the explicit value (verified by spike). The
  // output's shape is inferred (its M follows the input).
  std::array<std::uint64_t, 2> x_shape = {static_cast<std::uint64_t>(M),
                                           static_cast<std::uint64_t>(k_)};
  std::vector<bnns_graph_shape_t> shapes(arg_count_);
  for (auto& s : shapes) {
    s.rank = 0;
    s.shape = nullptr;
  }
  shapes[x_pos_].rank = 2;
  shapes[x_pos_].shape = x_shape.data();

  bnns_graph_context_t ctx = MakeCtx(ctx_data_, ctx_size_);
  int rc = BNNSGraphContextSetDynamicShapes(ctx, nullptr, arg_count_,
                                            shapes.data());
  if (rc != 0)
    return {StatusCode::kInternal, "apple_amx: SetDynamicShapes failed"};

  // Workspace requirement is M-dependent and grows after SetDynamicShapes — it
  // started at the M=1 default. Grow the thread_local buffer if needed; once
  // the program has seen its max M, this is a no-op.
  const std::size_t ws = BNNSGraphContextGetWorkspaceSize(ctx, nullptr);
  if (g_workspace.size() < ws) g_workspace.resize(ws);

  std::vector<bnns_graph_argument_t> args(arg_count_);
  for (auto& a : args) {
    a.data_ptr = nullptr;
    a.data_ptr_size = 0;
  }
  args[x_pos_].data_ptr = const_cast<float*>(in);
  args[x_pos_].data_ptr_size =
      static_cast<std::size_t>(M) * k_ * sizeof(float);
  args[out_pos_].data_ptr = out;
  args[out_pos_].data_ptr_size =
      static_cast<std::size_t>(M) * n_ * sizeof(float);

  rc = BNNSGraphContextExecute(ctx, nullptr, arg_count_, args.data(),
                               ws, g_workspace.data());
  return rc == 0 ? Status::Ok()
                 : Status{StatusCode::kInternal,
                          "apple_amx: BNNSGraphContextExecute failed"};
}

}  // namespace esm

#endif  // ESM_APPLE_AMX_AVAILABLE
