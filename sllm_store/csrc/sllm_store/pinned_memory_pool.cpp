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
#include "pinned_memory_pool.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

PinnedMemoryPool::PinnedMemoryPool(size_t total_size, size_t chunk_size)
    : chunk_size_(chunk_size) {
  size_t num_buffers = (total_size + chunk_size - 1) / chunk_size;
  if (num_buffers * chunk_size != total_size) {
    LOG(ERROR) << "PinnedMemoryPool size not multiple of chunk_size";
  }
  LOG(INFO) << "Creating PinnedMemoryPool with " << num_buffers
            << " buffers of " << chunk_size << " bytes";

  for (size_t i = 0; i < num_buffers; ++i) {
    char* buffer = static_cast<char*>(aligned_alloc(4096, chunk_size_));
    if (buffer == nullptr) {
      LOG(FATAL) << "Malloc failed";
    }

    cudaError_t err =
        cudaHostRegister(buffer, chunk_size_, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
      LOG(FATAL) << "cudaHostRegister failed: " << cudaGetErrorString(err);
    }
    pool_.insert(buffer);
    free_list_.insert(buffer);
  }
}

PinnedMemoryPool::~PinnedMemoryPool() {
  for (char* buffer : pool_) {
    cudaHostUnregister(buffer);
    free(buffer);
  }
}

int PinnedMemoryPool::Allocate(size_t size, std::vector<char*>& buffers) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (size == 0) {
    LOG(ERROR) << "PinnedMemoryPool Allocate size is zero";
    return -1;
  }

  int num_buffers_needed = (size + chunk_size_ - 1) / chunk_size_;
  if (num_buffers_needed > free_list_.size()) {
    LOG(ERROR) << "PinnedMemoryPool out of memory (" << free_list_.size()
               << " buffers available, " << num_buffers_needed
               << " buffers needed)";
    return num_buffers_needed - free_list_.size();
  }

  buffers.clear();
  buffers.resize(num_buffers_needed);
  auto it = free_list_.begin();
  for (size_t i = 0; i < num_buffers_needed; ++i) {
    buffers[i] = *it;
    it = free_list_.erase(it);
  }

  LOG(INFO) << "PinnedMemoryPool Allocate " << buffers.size() << " buffers"
            << " free buffers " << free_list_.size() << " total buffers "
            << pool_.size();

  return 0;  // Success
}

int PinnedMemoryPool::Deallocate(std::vector<char*>& buffers) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (char* buffer : buffers) {
    if (pool_.find(buffer) == pool_.end()) {
      LOG(ERROR) << "Buffer not found in pool";
      return -1;
    }
    if (free_list_.find(buffer) != free_list_.end()) {
      LOG(ERROR) << "Buffer already in free list";
      return -1;
    }
    free_list_.insert(buffer);
  }
  LOG(INFO) << "Deallocated " << buffers.size() << " buffers";
  return 0;  // Success
}
