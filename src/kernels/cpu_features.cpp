#include "esm_cpp/cpu_features.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string_view>

#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
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

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__APPLE__)
bool SysctlArmFeature(const char* name) {
  int value = 0;
  std::size_t size = sizeof(value);
  if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) return false;
  return value != 0;
}
bool HasNeonDotProd() { return SysctlArmFeature("hw.optional.arm.FEAT_DotProd"); }
bool HasNeonI8mm() { return SysctlArmFeature("hw.optional.arm.FEAT_I8MM"); }
#elif defined(__linux__)
// HWCAP bits are stable in the Linux ABI; define them if the toolchain's
// <asm/hwcap.h> predates FEAT_DotProd / FEAT_I8MM.
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1u << 20)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1u << 13)
#endif
bool HasNeonDotProd() {
  return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
}
bool HasNeonI8mm() { return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0; }
#else
bool HasNeonDotProd() { return false; }
bool HasNeonI8mm() { return false; }
#endif
#endif  // aarch64

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
  // i8mm (ARMv8.6) implies dotprod (ARMv8.2), so probe the top tier first.
  if (HasNeonI8mm()) return Isa::NeonI8mm;
  if (HasNeonDotProd()) return Isa::NeonDotProd;
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

bool ArmUseAppleAmx() {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
  // Phase 14 default-on flip: AMX engages unless the user explicitly opts
  // out. The `amx != nullptr` guard at the call site means this is a no-op
  // when no artifacts have been loaded — so the FP32/INT8 path still runs
  // for a user who never installed artifacts.
  const char* e = std::getenv("ESM_APPLE_AMX");
  if (e && *e != '\0') {
    const std::string_view s(e);
    if (s == "off" || s == "0" || s == "false") return false;
  }
  return true;
#else
  return false;
#endif
}

bool ArmUseAppleAne() {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
  const char* e = std::getenv("ESM_APPLE_ANE");
  if (!e || *e == '\0') return false;
  const std::string_view s(e);
  return s == "on" || s == "1" || s == "true";
#else
  return false;
#endif
}

bool ArmUseSmmla() {
  if (const char* f = std::getenv("ESM_FORCE_ISA");
      f && std::string_view(f) == "neoni8mm") {
    return true;
  }
  if (const char* e = std::getenv("ESM_NEON_I8MM");
      e && std::string_view(e) == "on") {
    return true;
  }
  return false;
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
