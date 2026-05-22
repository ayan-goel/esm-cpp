#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace esm {

// Minimal worker-pool with a single fan-out primitive: parallel_for.
// CLAUDE.md guidance: no std::async, no per-call thread allocation, no
// locks in the inner kernel loop. The pool's queue is touched only at
// parallel_for fan-out / fan-in.
//
// One process-global pool initialized lazily at first Model::load; size
// taken from ESM_NUM_THREADS (defaults to all logical cores via
// std::thread::hardware_concurrency() if unset).
class ThreadPool {
 public:
  explicit ThreadPool(std::size_t num_threads);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  std::size_t size() const { return workers_.size(); }

  // Apply body([chunk_begin, chunk_end)) over [begin, end) in disjoint
  // chunks of size >= grain, distributed across workers. Blocks until
  // every chunk has completed. The partition is deterministic given
  // (begin, end, grain, size()) — important for bit-identical numerics
  // across reruns at the same thread count.
  void parallel_for(int begin, int end, int grain,
                    const std::function<void(int, int)>& body);

  // Read ESM_NUM_THREADS and clamp to >= 1. Falls back to
  // std::thread::hardware_concurrency().
  static ThreadPool FromEnv();

 private:
  struct Task {
    int begin;
    int end;
    const std::function<void(int, int)>* body;
  };

  void WorkerMain();

  std::vector<std::thread> workers_;
  std::queue<Task> queue_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::atomic<int> pending_{0};
  bool stop_ = false;
};

// Process-global pool, lazily initialized at first call. Sized from
// ESM_NUM_THREADS (default: all logical cores, hardware_concurrency) at first
// construction. Subsequent env-var changes are not honored.
// NOTE: on Apple Silicon, including the E-cores measured fastest for the
// GEMM-bound forward — the grain-based parallel_for load-balances across
// heterogeneous P/E cores, so a P-core-only default regresses (Phase 10 T1:
// M3 Pro 650M uniform 12 cores 2826ms < 8 2919 < 6 P-only 3030).
ThreadPool& GlobalPool();

// True iff the current thread is a worker inside GlobalPool(). Use this
// to avoid nested parallel_for calls that would deadlock when every
// worker is already blocked waiting on its own inner dispatch.
bool InGlobalPoolWorker();

}  // namespace esm
