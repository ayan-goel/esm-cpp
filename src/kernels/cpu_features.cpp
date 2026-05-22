#include "esm_cpp/cpu_features.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string_view>

#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

namespace esm {

namespace {

#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
// Raw CPUID probe for AMX-INT8 (leaf 7, subleaf 0, EDX bit 25). The
// __builtin_cpu_supports("amx-int8") string isn't recognized by Clang < 15
// (Ubuntu 22.04 ships Clang 14), so we sidestep the builtin and read the
// feature bit directly. AMX as a family also requires XSAVE permission via
// arch_prctl at first tile use; that's gated separately in the kernel.
bool HasAmxInt8() {
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
  return (edx & (1u << 25)) != 0;
}
#endif

Isa DetectHostIsa() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  if (HasAmxInt8()) return Isa::Amx;
  if (__builtin_cpu_supports("avx512vnni")) return Isa::Avx512Vnni;
  if (__builtin_cpu_supports("avx512f")) return Isa::Avx512;
  if (__builtin_cpu_supports("avx2")) return Isa::Avx2;
#endif
  return Isa::Ref;
#elif defined(__aarch64__) || defined(_M_ARM64)
  return Isa::Neon;
#else
  return Isa::Ref;
#endif
}

}  // namespace

Isa HostIsa() {
  static const Isa cached = DetectHostIsa();
  return cached;
}

std::string_view IsaToString(Isa isa) {
  switch (isa) {
    case Isa::Ref:
      return "ref";
    case Isa::Neon:
      return "neon";
    case Isa::NeonDotProd:
      return "neondotprod";
    case Isa::NeonI8mm:
      return "neoni8mm";
    case Isa::Avx2:
      return "avx2";
    case Isa::Avx512:
      return "avx512";
    case Isa::Avx512Vnni:
      return "avx512vnni";
    case Isa::Amx:
      return "amx";
  }
  return "ref";
}

std::optional<Isa> StringToIsa(std::string_view s) {
  if (s == "ref") return Isa::Ref;
  if (s == "neon") return Isa::Neon;
  if (s == "neondotprod") return Isa::NeonDotProd;
  if (s == "neoni8mm") return Isa::NeonI8mm;
  if (s == "avx2") return Isa::Avx2;
  if (s == "avx512") return Isa::Avx512;
  if (s == "avx512vnni") return Isa::Avx512Vnni;
  if (s == "amx") return Isa::Amx;
  return std::nullopt;
}

Isa CurrentIsa() {
  if (const char* env = std::getenv("ESM_FORCE_ISA"); env && *env != '\0') {
    if (auto parsed = StringToIsa(env)) return *parsed;
  }
  return HostIsa();
}

void MaybeLogIsaOnce() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    const char* env = std::getenv("ESM_LOG_ISA");
    if (env == nullptr) return;
    if (std::string_view(env) != "1") return;
    const auto name = IsaToString(CurrentIsa());
    std::fprintf(stderr, "esm.cpp: ISA = %.*s\n", static_cast<int>(name.size()),
                 name.data());
  });
}

}  // namespace esm
