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

#include <filesystem>
#include <mutex>
#include <unordered_set>
#include <vector>

// Forward declaration
class PinnedMemoryInstance;

// Factory functions for creating/opening pinned memory regions
std::unique_ptr<PinnedMemoryInstance> Create(std::string_view name,
                                             size_t size, void* base_addr = nullptr);
std::unique_ptr<PinnedMemoryInstance> Open(std::string_view name);

// RAII class for managing pinned memory
class PinnedMemoryInstance {
 public:
  // Move-only class
  PinnedMemoryInstance(const PinnedMemoryInstance&) = delete;
  PinnedMemoryInstance& operator=(const PinnedMemoryInstance&) = delete;
  PinnedMemoryInstance(PinnedMemoryInstance&&) = default;
  PinnedMemoryInstance& operator=(PinnedMemoryInstance&&) = default;

  ~PinnedMemoryInstance();

  void prefault_pages() {
    // Prefault pages to ensure they are allocated
    if (data_ != nullptr) {
      volatile char* p = static_cast<volatile char*>(data_);
      for (size_t i = 0; i < size_; i += 4096) {
        p[i] = 0;  // Touch each page
      }
    }
  }

  // Accessors
  void* data() const { return data_; }
  size_t size() const { return size_; }
  const std::string& name() const { return name_; }
  bool is_valid() const { return data_ != nullptr; }

  // Typed access
  template <typename T>
  T* data_as() const {
    return static_cast<T*>(data_);
  }

  template <typename T>
  size_t capacity() const {
    return size_ / sizeof(T);
  }

 private:
  friend std::unique_ptr<PinnedMemoryInstance> Create(std::string_view name,
                                                      size_t size);
  friend std::unique_ptr<PinnedMemoryInstance> Open(std::string_view name);

  PinnedMemoryInstance() = default;

  std::string name_;
  int fd_ = -1;
  void* data_ = nullptr;
  size_t size_ = 0;
  size_t mapped_size_ = 0;  // Actual mapped size (page-aligned)
  bool is_owner_ = false;
  bool is_huge_page_ = false;
};

class PinnedMemoryPool {
 public:
  PinnedMemoryPool(size_t total_size, size_t chunk_size);
  ~PinnedMemoryPool();

  int Allocate(size_t size, std::vector<char*>& buffers);
  int Deallocate(std::vector<char*>& buffers);
  size_t chunk_size() const { return chunk_size_; }

  // Forbid copy and assignment
  PinnedMemoryPool(const PinnedMemoryPool&) = delete;
  PinnedMemoryPool& operator=(const PinnedMemoryPool&) = delete;

 private:
  std::mutex mutex_;
  std::unordered_set<char*> free_list_;
  std::unordered_set<char*> pool_;
  size_t chunk_size_;
  std::vector<std::unique_ptr<PinnedMemoryInstance>> instances_;
};