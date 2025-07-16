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

#include <atomic>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <unordered_map>

#include "shared_memory.h"

// Base allocator interface
class MemoryAllocator {
 public:
  virtual ~MemoryAllocator() = default;
  virtual void* allocate(size_t size, size_t alignment = 4096) = 0;
  virtual void deallocate(void* ptr) = 0;
  virtual bool is_shared() const = 0;
};

// Aligned allocator using aligned_alloc
class AlignedAllocator : public MemoryAllocator {
 public:
  void* allocate(size_t size, size_t alignment = 4096) override {
    return aligned_alloc(alignment, size);
  }

  void deallocate(void* ptr) override { free(ptr); }

  bool is_shared() const override { return false; }
};

// Shared memory allocator
class SharedMemoryAllocator : public MemoryAllocator {
 public:
  explicit SharedMemoryAllocator(std::string_view name_prefix)
      : name_prefix_(name_prefix), counter_(0) {}

  void* allocate(size_t size, size_t alignment = 4096) override {
    std::string name =
        std::string(name_prefix_) + "_" + std::to_string(counter_++);

    // Try to open existing shared memory first (like CUDA handle approach)
    auto shm = Open(name);
    if (!shm || !shm->is_valid()) {
      // If opening failed, create new shared memory
      shm = Create(name, size);
      if (!shm || !shm->is_valid()) {
        return nullptr;
      }
    }

    void* ptr = shm->data();
    // Store the shared memory instance for cleanup
    std::lock_guard<std::mutex> lock(mutex_);
    shared_memories_[ptr] = std::move(shm);
    return ptr;
  }

  void deallocate(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = shared_memories_.find(ptr);
    if (it != shared_memories_.end()) {
      shared_memories_.erase(it);
    }
  }

  std::unordered_map<int, std::string> GetSharedMemoryHandles(
      const std::unordered_map<int, void*>& memory_ptrs) {
    std::unordered_map<int, std::string> shm_handles;

    for (const auto& p : memory_ptrs) {
      int device = p.first;
      void* ptr = p.second;

      auto it = shared_memories_.find(ptr);
      if (it != shared_memories_.end()) {
        std::string shm_name = it->second->name();
        shm_handles[device] = shm_name;
      } else {
        throw std::runtime_error("Shared memory handle not found for pointer");
      }
    }
    return shm_handles;
  }

  bool is_shared() const override { return true; }

 private:
  std::string name_prefix_;
  std::atomic<size_t> counter_;
  std::mutex mutex_;
  std::unordered_map<void*, std::unique_ptr<SharedMemoryInstance>>
      shared_memories_;
};
