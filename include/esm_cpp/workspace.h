#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <vector>

namespace esm {

// Bump (monotonic) allocator owned by Model. The forward path pulls every
// scratch buffer from here, then calls reset() at the top of the next
// forward to rewind the cursor. Buffer capacity grows only on overflow,
// so after the first call at the max sequence length subsequent calls
// allocate zero new bytes.
//
// Not thread-safe. Model::Forward is documented non-reentrant; concurrent
// forwards need a per-caller Workspace (Phase 3 scheduler will own that).
//
// Only trivially-destructible POD types may be allocated; the arena does
// not call destructors on reset() or destruction.
class Workspace {
 public:
  Workspace() = default;
  explicit Workspace(std::size_t initial_bytes) { reserve(initial_bytes); }

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  Workspace& operator=(Workspace&&) = default;

  void reserve(std::size_t bytes) {
    if (bytes > buffer_.size()) buffer_.resize(bytes);
  }

  template <typename T>
  T* allocate(std::size_t n, std::size_t align = alignof(T)) {
    static_assert(std::is_trivially_destructible_v<T>,
                  "Workspace requires trivially-destructible types");
    const std::size_t bytes = n * sizeof(T);
    const std::size_t aligned_offset = AlignUp(offset_, align);
    if (aligned_offset + bytes > buffer_.size()) {
      Grow(aligned_offset + bytes);
    }
    void* p = buffer_.data() + aligned_offset;
    offset_ = aligned_offset + bytes;
    return static_cast<T*>(p);
  }

  void reset() { offset_ = 0; }

  std::size_t bytes_used() const { return offset_; }
  std::size_t bytes_capacity() const { return buffer_.size(); }

  bool in_use() const { return in_use_; }

  // RAII guard: top of Forward calls ws.activate() to flag the workspace
  // as in-use. Debug builds assert() on re-entry; release builds just
  // reset() the cursor.
  class Activation {
   public:
    explicit Activation(Workspace& ws) : ws_(&ws) {
      ws_->in_use_ = true;
      ws_->reset();
    }
    ~Activation() {
      if (ws_) ws_->in_use_ = false;
    }
    Activation(const Activation&) = delete;
    Activation& operator=(const Activation&) = delete;
    Activation(Activation&& other) noexcept : ws_(other.ws_) {
      other.ws_ = nullptr;
    }
    Activation& operator=(Activation&&) = delete;

   private:
    Workspace* ws_;
  };

  Activation activate() { return Activation(*this); }

 private:
  static std::size_t AlignUp(std::size_t value, std::size_t align) {
    return (value + (align - 1)) & ~(align - 1);
  }

  void Grow(std::size_t at_least) {
    // Growing the backing buffer reallocates and invalidates every pointer
    // we've already handed out. That's a load-bearing bug when it happens
    // mid-forward, so refuse to grow while activated — callers must
    // reserve() enough upfront, right after activate().
    if (in_use_) {
      std::fprintf(stderr,
                   "esm.cpp: Workspace overflow during an active forward "
                   "(need %zu bytes, capacity %zu). Call Workspace::reserve "
                   "with the worst-case size before allocating.\n",
                   at_least, buffer_.size());
      std::abort();
    }
    std::size_t new_size = buffer_.empty() ? 64 : buffer_.size() * 2;
    while (new_size < at_least) new_size *= 2;
    buffer_.resize(new_size);
  }

  std::vector<std::byte> buffer_;
  std::size_t offset_ = 0;
  bool in_use_ = false;
};

}  // namespace esm
