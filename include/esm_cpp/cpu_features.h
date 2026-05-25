#pragma once

#include <optional>
#include <string_view>

namespace esm {

enum class Isa : int {
  Ref = 0,
  Neon = 1,
  Avx2 = 2,
  Avx512 = 3,
  Avx512Vnni = 4,
  Amx = 5,
  // ARM INT8 tiers (v0.2). Appended to keep the x86 values stable.
  // NeonDotProd: ARMv8.2 FEAT_DotProd (SDOT) — the VNNI analog.
  // NeonI8mm:    ARMv8.6 FEAT_I8MM (SMMLA) — the AMX analog.
  NeonDotProd = 6,
  NeonI8mm = 7,
};

// Best ISA available on the current host, probed once via __builtin_cpu_supports
// on x86 and via build-time macros on ARM. Idempotent and cheap on repeat calls.
Isa HostIsa();

// HostIsa() unless overridden by the ESM_FORCE_ISA env var. The env var is
// re-read on every call so tests can flip it without restarting the process.
// Unknown values are ignored (fall through to HostIsa()).
Isa CurrentIsa();

// Stable lowercase names: "ref", "neon", "neondotprod", "neoni8mm", "avx2",
// "avx512", "avx512vnni", "amx".
std::string_view IsaToString(Isa isa);

// Parses the names emitted by IsaToString. Case-sensitive; nullopt otherwise.
std::optional<Isa> StringToIsa(std::string_view s);

// Whether to route FP32 GEMM through Apple's Accelerate (AMX coprocessor)
// instead of the hand-written NEON FMLA kernel. Apple-only, opt-in via
// ESM_APPLE_AMX=on (Accelerate is a closed lib with no scalar cross-check;
// the hand-written NEON path stays the portable default and the only path on
// Linux ARM / Graviton). Measured ~3.75-6.2x faster than NEON FMLA on M3.
// Re-read on each call; returns false on non-Apple builds.
bool ArmUseAppleAmx();

// Whether to route the dense GEMMs through CoreML's Apple Neural Engine (ANE)
// path. Apple-only, opt-in via ESM_APPLE_ANE=on. ANE delivers 2-4x over our
// fp16-AMX path on the M values our forward uses (Phase 12 T1 characterization),
// but requires static-shape mlmodelc artifacts built at convert time. Re-read
// on each call; returns false on non-Apple builds. Independent of ArmUseAppleAmx
// — if both are on AND the relevant context is loaded, ANE wins; if ANE has no
// matching bucket for the runtime M, AMX (or SDOT default) takes over.
bool ArmUseAppleAne();

// Whether the ARM i8mm (SMMLA) INT8 kernel should engage. SMMLA is opt-in:
// it does not beat the SDOT kernel on Apple M3 (measured), so an auto-detected
// NeonI8mm host uses SDOT by default. SMMLA engages only on an explicit request
// — ESM_FORCE_ISA=neoni8mm or ESM_NEON_I8MM=on — and is expected to win on
// hardware with stronger SMMLA throughput (e.g. AWS Graviton3). Re-read on
// each call. Returns false on non-ARM builds.
bool ArmUseSmmla();

// Log "esm.cpp: ISA = <name>\n" to stderr at most once per process when
// ESM_LOG_ISA=1. Intended to be called by Model::load; safe to call repeatedly.
void MaybeLogIsaOnce();

}  // namespace esm
