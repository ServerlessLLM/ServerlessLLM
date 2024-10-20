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

#include <memory>
#include <vector>

#include "pinned_memory_pool.h"

class PinnedMemory {
 public:
  PinnedMemory() = default;
  ~PinnedMemory();

  // Disable copying and moving
  PinnedMemory(const PinnedMemory&) = delete;
  PinnedMemory& operator=(const PinnedMemory&) = delete;
  PinnedMemory(PinnedMemory&&) = delete;
  PinnedMemory& operator=(PinnedMemory&&) = delete;

  int Allocate(size_t size, std::shared_ptr<PinnedMemoryPool> mempool);
  std::vector<char*>& get();
  size_t num_chunks() const { return buffers_.size(); }
  size_t chunk_size() const { return mempool_->chunk_size(); }

 private:
  std::vector<char*> buffers_;
  std::shared_ptr<PinnedMemoryPool> mempool_;
};
