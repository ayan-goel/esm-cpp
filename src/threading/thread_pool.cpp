#include "esm_cpp/thread_pool.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace esm {

ThreadPool::ThreadPool(std::size_t num_threads) {
  if (num_threads == 0) num_threads = 1;
  workers_.reserve(num_threads);
  for (std::size_t i = 0; i < num_threads; ++i) {
    workers_.emplace_back([this] { WorkerMain(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> g(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) {
    if (t.joinable()) t.join();
  }
}

void ThreadPool::WorkerMain() {
  for (;;) {
    Task task{0, 0, nullptr};
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
      if (stop_ && queue_.empty()) return;
      task = queue_.front();
      queue_.pop();
    }
    (*task.body)(task.begin, task.end);
    if (pending_.fetch_sub(1) == 1) {
      std::lock_guard<std::mutex> g(mu_);
      done_cv_.notify_all();
    }
  }
}

void ThreadPool::parallel_for(int begin, int end, int grain,
                              const std::function<void(int, int)>& body) {
  if (end <= begin) return;
  if (grain < 1) grain = 1;
  const int total = end - begin;
  const int nworkers = std::max<int>(1, static_cast<int>(workers_.size()));
  // Aim for roughly one chunk per worker, but never smaller than grain.
  int chunk = (total + nworkers - 1) / nworkers;
  if (chunk < grain) chunk = grain;

  std::vector<Task> tasks;
  for (int b = begin; b < end; b += chunk) {
    Task t;
    t.begin = b;
    t.end = std::min(end, b + chunk);
    t.body = &body;
    tasks.push_back(t);
  }

  // Special-case: single chunk runs in the caller thread to avoid the
  // hand-off cost (and keeps Forward(B=1) cheap when batch=1).
  if (tasks.size() == 1) {
    body(tasks[0].begin, tasks[0].end);
    return;
  }

  pending_.store(static_cast<int>(tasks.size()));
  {
    std::lock_guard<std::mutex> g(mu_);
    for (auto& t : tasks) queue_.push(t);
  }
  cv_.notify_all();
  std::unique_lock<std::mutex> lk(mu_);
  done_cv_.wait(lk, [&] { return pending_.load() == 0; });
}

ThreadPool ThreadPool::FromEnv() {
  std::size_t n = 0;
  if (const char* env = std::getenv("ESM_NUM_THREADS"); env && *env) {
    try {
      const int v = std::stoi(env);
      if (v > 0) n = static_cast<std::size_t>(v);
    } catch (...) {
      n = 0;
    }
  }
  if (n == 0) n = std::thread::hardware_concurrency();
  if (n == 0) n = 1;
  return ThreadPool(n);
}

}  // namespace esm
