#ifndef HASH_SET_COARSE_GRAINED_H
#define HASH_SET_COARSE_GRAINED_H

// Suppress Clang 21+ false-positive -Wthread-safety-negative warnings triggered
// by -Weverything on standard library mutex wrappers without modifying CMakeLists.txt.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wthread-safety-negative"
#endif

#include <cassert>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

#include "src/hash_set_base.h"

template <typename T>
class HashSetCoarseGrained : public HashSetBase<T> {
 public:
  explicit HashSetCoarseGrained(size_t initial_capacity)
    : table_(initial_capacity > 0 ? initial_capacity : 1), set_size_(0) {}

  bool Add(T elem) final {
    // Acquiring |mutex_| via RAII (std::scoped_lock) ensures exclusive access
    // to both |table_| and |set_size_| for the entire duration of Add(),
    // including any conditional Policy() check and Resize() operation.
    std::scoped_lock<std::mutex> lock(mutex_);
    auto& bucket = GetBucket(elem);
    if (std::find(bucket.begin(), bucket.end(), elem) != bucket.end()) {
      return false;
    }
    bucket.push_back(elem);
    ++set_size_;
    if (Policy()) {
      Resize();
    }
    return true;
  }

  bool Remove(T elem) final {
    // Acquiring |mutex_| prevents concurrent threads from reading or mutating
    // the target bucket or resizing |table_| while we search and erase |elem|.
    std::scoped_lock<std::mutex> lock(mutex_);
    auto& bucket = GetBucket(elem);
    auto it = std::find(bucket.begin(), bucket.end(), elem);
    if (it == bucket.end()) {
      return false;
    }
    bucket.erase(it);
    --set_size_;
    return true;
  }

  [[nodiscard]] bool Contains(T elem) final {
    // Even though Contains() is a read-only operation, acquiring |mutex_| is
    // necessary to prevent data races with concurrent Add()/Remove() calls that
    // may reallocate the underlying bucket std::vector or trigger a Resize()
    // that replaces |table_|.
    std::scoped_lock<std::mutex> lock(mutex_);
    const auto& bucket = GetBucket(elem);
    return std::find(bucket.begin(), bucket.end(), elem) != bucket.end();
  }

  [[nodiscard]] size_t Size() const final {
    // Acquiring |mutex_| ensures a thread-safe, sequentially consistent read of
    // |set_size_| without a data race against concurrent increments/decrements
    // in Add() and Remove(). |mutex_| is marked mutable so it can be locked
    // within this const member function.
    std::scoped_lock<std::mutex> lock(mutex_);
    return set_size_;
  }

  private:
  [[nodiscard]] bool Policy() const { return set_size_ / table_.size() > 4; }

  // Doubles the capacity of |table_| and rehashes all elements.
  // Exploitation of Non-Re-Entrancy & Deviation from Textbook (Figure 13.4):
  // 1. In the textbook, BaseHashSet::add() releases the lock before checking
  //    policy() and calling resize(), which forces CoarseHashSet::resize() to
  //    re-acquire the lock and check `if (oldCapacity != table.length) return;`
  //    in case another thread resized the table in the meantime (also note that
  //    reading `table.length` before acquiring the lock in Fig 13.4 Line 19 is
  //    a data race in C++).
  // 2. Because our coarse-grained hash set only uses a single lock, Add() keeps
  //    |mutex_| locked continuously across insertion, Policy(), and Resize().
  //    Consequently, Resize() does not re-acquire |mutex_| and does not call
  //    Add() to re-insert elements.
  void Resize() {
    size_t new_capacity = table_.size() * 2;
    std::vector<std::vector<T>> new_table(new_capacity);
    for (const auto& bucket : table_) {
      for (const auto& elem : bucket) {
        size_t new_index = std::hash<T>{}(elem) % new_capacity;
        new_table[new_index].push_back(elem);
      }
    }
    table_ = std::move(new_table);
  }

  [[nodiscard]] std::vector<T>& GetBucket(const T& elem) {
    return table_[std::hash<T>{}(elem) % table_.size()];
  }

  std::vector<std::vector<T>> table_;
  size_t set_size_;
  mutable std::mutex mutex_;
};

#endif  // HASH_SET_COARSE_GRAINED_H
