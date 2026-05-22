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

// Log "esm.cpp: ISA = <name>\n" to stderr at most once per process when
// ESM_LOG_ISA=1. Intended to be called by Model::load; safe to call repeatedly.
void MaybeLogIsaOnce();

}  // namespace esm
