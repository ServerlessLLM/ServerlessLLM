// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

template <typename T>
class ConcurrentVector {
 private:
  std::vector<T> data;
  std::mutex mtx_;
  std::condition_variable cond_;
  size_t size_ = 0;
  size_t capacity_ = 0;
  std::string name_;
  std::unordered_set<size_t> keys_;

 public:
  ConcurrentVector() = default;
  ConcurrentVector(const ConcurrentVector&) = delete;
  ConcurrentVector& operator=(const ConcurrentVector&) = delete;
  void init(std::string name, size_t capacity);
  size_t capacity();
  bool find(size_t key);
  void enqueue(uint64_t key, T item);
  T dequeue(size_t pivot);
};

template <typename T>
void ConcurrentVector<T>::init(std::string name, size_t capacity) {
  std::lock_guard<std::mutex> lock(mtx_);
  capacity_ = capacity;
  name_ = std::move(name);
  data.resize(capacity_);
  size_ = 0;
}

template <typename T>
size_t ConcurrentVector<T>::capacity() {
  std::lock_guard<std::mutex> lock(mtx_);
  return capacity_;
}

template <typename T>
bool ConcurrentVector<T>::find(size_t key) {
  std::lock_guard<std::mutex> lock(mtx_);
  return keys_.find(key) != keys_.end();
}

template <typename T>
void ConcurrentVector<T>::enqueue(uint64_t key, T item) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (keys_.find(key) != keys_.end()) {
    return;
  }

  size_t idx = size_;
  size_++;
  data[idx] = std::move(item);
  keys_.insert(key);
  cond_.notify_all();
}

template <typename T>
T ConcurrentVector<T>::dequeue(size_t pivot) {
  std::unique_lock<std::mutex> lock(mtx_);
  cond_.wait(lock, [this, pivot] { return pivot < size_; });
  return data[pivot];
}
