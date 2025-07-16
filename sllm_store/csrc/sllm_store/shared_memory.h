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
class SharedMemoryInstance;

// Factory functions for creating/opening shared memory regions
std::unique_ptr<SharedMemoryInstance> Create(std::string_view name, size_t size,
                                             void* base_addr = nullptr);
std::unique_ptr<SharedMemoryInstance> Open(std::string_view name);

// RAII class for managing shared memory
class SharedMemoryInstance {
 public:
  // Move-only class
  SharedMemoryInstance(const SharedMemoryInstance&) = delete;
  SharedMemoryInstance& operator=(const SharedMemoryInstance&) = delete;
  SharedMemoryInstance(SharedMemoryInstance&&) = default;
  SharedMemoryInstance& operator=(SharedMemoryInstance&&) = default;

  ~SharedMemoryInstance();

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
  friend std::unique_ptr<SharedMemoryInstance> Create(std::string_view name,
                                                      size_t size,
                                                      void* base_addr);
  friend std::unique_ptr<SharedMemoryInstance> Open(std::string_view name);
  friend class MemoryRegistry;

  SharedMemoryInstance() = default;

  std::string name_;
  int fd_ = -1;
  void* data_ = nullptr;
  size_t size_ = 0;
  size_t mapped_size_ = 0;  // Actual mapped size (page-aligned)
  bool is_owner_ = false;
  bool is_huge_page_ = false;
};
