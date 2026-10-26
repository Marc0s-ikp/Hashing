#ifndef HASH_SET_STRIPED_H
#define HASH_SET_STRIPED_H

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wthread-safety-negative"
#endif

#include <cassert>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

#include "src/hash_set_base.h"

template <typename T>
class HashSetStriped : public HashSetBase<T> {
 public:
  explicit HashSetStriped(size_t initial_capacity)
    : table_(initial_capacity > 0 ? initial_capacity : 1),
      locks_(initial_capacity > 0 ? initial_capacity : 1),
      set_size_(0) {}

  bool Add(T elem) final {
    const size_t hash = std::hash<T>()(elem);
    size_t old_capacity = 0;
    bool should_resize = false;

    {
      // Acquire only the single stripe lock protecting |elem|'s bucket.
      // We scope this lock block so that if a resize is needed, this single
      // lock is released BEFORE calling Resize(). Holding a stripe lock while
      // attempting to acquire all locks in ascending order inside Resize()
      // would cause an ABBA deadlock if two threads on different stripes
      // triggered Resize() simultaneously.
      std::scoped_lock<std::mutex> lock(GetLock(hash));
      auto& bucket = table_[hash % table_.size()];
      if (std::find(bucket.begin(), bucket.end(), elem) != bucket.end()) {
        return false;
      }
      bucket.push_back(elem);
      set_size_.fetch_add(1, std::memory_order_relaxed);

      // Deviation from Textbook (Figures 13.2 & 13.7):
      // In the textbook, BaseHashSet::add() releases the stripe lock BEFORE
      // calling policy() and reading `oldCapacity = table.length` at the start
      // of resize(). In C++, reading `table_.size()` without holding at least
      // one stripe lock would be a data race against `table_ = std::move(...)`
      // in a concurrent Resize(). By evaluating Policy() and recording
      // |old_capacity| while still holding our stripe lock, we guarantee race-free
      // access to |table_.size()|.
      old_capacity = table_.size();
      should_resize = Policy();
    }

    if (should_resize) {
      Resize(old_capacity);
    }
    return true;

  }

  bool Remove(T elem) final {
    // Acquiring the element's stripe lock ensures exclusive access to the
    // target bucket and prevents a concurrent Resize() from replacing |table_|
    // while we search and erase |elem|.
    const size_t hash = std::hash<T>()(elem);
    std::scoped_lock<std::mutex> lock(GetLock(hash));
    auto& bucket = table_[hash % table_.size()];
    auto it = std::find(bucket.begin(), bucket.end(), elem);
    if (it == bucket.end()) {
      return false;
    }
    bucket.erase(it);
    set_size_.fetch_sub(1, std::memory_order_relaxed);
    return true;

  }

  [[nodiscard]] bool Contains(T elem) final {
    // Acquiring the element's stripe lock prevents concurrent modifications to
    // the bucket and prevents Resize() from invalidating |table_| during lookup.
    const size_t hash = std::hash<T>()(elem);
    std::scoped_lock<std::mutex> lock(GetLock(hash));
    const auto& bucket = table_[hash % table_.size()];
    return std::find(bucket.begin(), bucket.end(), elem) != bucket.end();
  }

  [[nodiscard]] size_t Size() const final {
    // Using std::atomic<size_t> allows Size() to read the current element count
    // safely in O(1) without locking all stripes.
    return set_size_.load(std::memory_order_relaxed);
  }

  private:
  // Custom RAII guard that acquires all mutexes in |locks_| in strict ascending
  // index order (0 to L - 1) on construction and releases them on destruction.
  class AllLocksGuard {
   public:
    explicit AllLocksGuard(std::vector<std::mutex>& locks)
      : locks_(locks) {
      for (auto& mtx : locks_) {
        mtx.lock();
      }
    }

    ~AllLocksGuard() {
      for (auto& mtx : locks_) {
        mtx.unlock();
      }
    }

    AllLocksGuard(const AllLocksGuard&) = delete;
    AllLocksGuard& operator=(const AllLocksGuard&) = delete;

   private:
    std::vector<std::mutex>& locks_;
  };

  // Precondition: The calling thread MUST hold at least one stripe lock in
  // |locks_| so that reading |table_.size()| is protected against concurrent
  // Resize() calls.
  [[nodiscard]] bool Policy() const {
    return set_size_.load(std::memory_order_relaxed) / table_.size() > 4;
  }

  // Stops the world by acquiring all L stripe locks, then doubles |table_|'s
  // capacity and rehashes all elements if another thread hasn't already done so.
  //
  // Exploitation of Non-Re-Entrancy & Deviation from Textbook (Figure 13.7):
  // The textbook's resize() calls initializeFrom(), which calls add(x) while all
  // locks are held—requiring re-entrant locks (Java ReentrantLock) and risking
  // nested resize() calls. Because our hash set does not support re-entrancy,
  // we use standard non-recursive std::mutexes and transfer elements directly
  // into |new_table| via push_back(). This avoids the overhead of recursive
  // mutexes, eliminates redundant lock acquisitions/releases for every element
  // during rehashing, and skips redundant std::find duplicate checks.

  void Resize(size_t old_capacity) {
    // Because we released our single stripe lock in Add() before acquiring all
    // locks here, another thread may have already resized |table_| in the
    // interim. Verify that the capacity has not changed before resizing.
    AllLocksGuard guard(locks_);
    if (table_.size() != old_capacity) {
      return;
    }
    size_t new_capacity = 2 * table_.size();
    std::vector<std::vector<T>> new_table(new_capacity);
    for (const auto& bucket : table_) {
      for (const auto& elem : bucket) {
        size_t new_bucket_idx = std::hash<T>()(elem) % new_capacity;
        new_table[new_bucket_idx].push_back(elem);
      }
    }
    table_ = std::move(new_table);
  }

  [[nodiscard]] std::mutex& GetLock(size_t hash) {
    return locks_[hash % locks_.size()];
  }

  std::vector<std::vector<T>> table_;
  std::vector<std::mutex> locks_;
  std::atomic<size_t> set_size_;
  
};


#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif  // HASH_SET_STRIPED_H
