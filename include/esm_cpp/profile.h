#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

// Lightweight per-section timer. Runtime-gated by the ESM_PROFILE env var
// (read once at first call); when off, AddNs() and BumpCounter() do
// nothing observable beyond an atomic add to a per-thread accumulator
// that no one reads. When on, DumpAndReset() prints a section breakdown
// to stderr at the end of each ForwardPackedInto.
//
// Sections are identified by a string literal. We aggregate by name
// across all worker threads using a process-global mutex-guarded map at
// dump time only — the hot path uses thread_local storage so kernels
// never touch a shared cache line per call.

namespace esm::profile {

bool Enabled();

// Add wall-time nanoseconds to the named section on the current thread.
// Safe (and free) when Enabled() is false.
void AddNs(const char* section, std::int64_t ns);

// Increment a named counter on the current thread.
void BumpCounter(const char* name, std::int64_t by = 1);

// Aggregate all thread-local accumulators across threads, print to
// stderr, and reset to zero for the next forward.
void DumpAndReset();

class ScopedTimer {
 public:
  explicit ScopedTimer(const char* name) : name_(name) {
    if (Enabled()) start_ = std::chrono::steady_clock::now();
  }
  ~ScopedTimer() {
    if (!Enabled()) return;
    const auto end = std::chrono::steady_clock::now();
    AddNs(name_, std::chrono::duration_cast<std::chrono::nanoseconds>(
                      end - start_)
                      .count());
  }

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  const char* name_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace esm::profile
