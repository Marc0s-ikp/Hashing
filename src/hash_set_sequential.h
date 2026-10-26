#ifndef HASH_SET_SEQUENTIAL_H
#define HASH_SET_SEQUENTIAL_H

#include <cassert>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "src/hash_set_base.h"

template <typename T>
class HashSetSequential : public HashSetBase<T> {
 public:
  explicit HashSetSequential(size_t initial_capacity)
    : table_(initial_capacity > 0 ? initial_capacity : 1), set_size_(0) {}

  bool Add(T elem) final {
    auto& bucket = GetBucket(elem);
    // Deviation from textbook Fig 13.2 (Line 27): In C++, std::vector::push_back
    // unconditionally appends elements without checking for duplicates. To
    // maintain strict mathematical set semantics (no duplicate keys), we first
    // scan the bucket via std::find and abort if |elem| is already present.
    if (std::find(bucket.begin(), bucket.end(), elem) != bucket.end()) {
      return false;  // Element already exists
    }
    bucket.push_back(elem);
    ++set_size_;
    if (Policy()) {
      Resize();
    }
    return true;
  }

  bool Remove(T elem) final {
    auto& bucket = GetBucket(elem);
    auto it = std::find(bucket.begin(), bucket.end(), elem);
    if (it == bucket.end()) {
      return false;  // Element not found
    }
    bucket.erase(it);
    --set_size_;
    return true;
  }

  [[nodiscard]] bool Contains(T elem) final {
    const auto& bucket = GetBucket(elem);
    return std::find(bucket.begin(), bucket.end(), elem) != bucket.end();
  }

  [[nodiscard]] size_t Size() const final {
    return set_size_;
  }
 
  private:
  // Returns true when the average bucket length exceeds the threshold of 4,
  // matching the policy in The Art of Multiprocessor Programming (Section 13.2.1).
  [[nodiscard]] bool Policy() const { return set_size_ / table_.size() > 4; }
  
  // Doubles the capacity of the hash table and rehashes all existing elements.
  // Instead of calling Add() (which would be re-entrant and redundant since
  // elements in the set are already guaranteed to be unique), we insert
  // elements directly into the new buckets.
  void Resize() {
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

  [[nodiscard]] std::vector<T>& GetBucket(const T& elem) {
    return table_[std::hash<T>()(elem) % table_.size()];
  }

  std::vector<std::vector<T>> table_;
  size_t set_size_;
};

#endif  // HASH_SET_SEQUENTIAL_H
