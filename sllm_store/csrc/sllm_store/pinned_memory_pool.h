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

#include <mutex>
#include <unordered_set>
#include <vector>

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
};