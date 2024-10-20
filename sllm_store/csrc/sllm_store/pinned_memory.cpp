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
#include "pinned_memory.h"

#include <glog/logging.h>

PinnedMemory::~PinnedMemory() {
  LOG(INFO) << "Deallocating " << buffers_.size() << " memory chunks";
  int ret = mempool_->Deallocate(buffers_);
  if (ret != 0) {
    LOG(ERROR) << "Error deallocating CPU memory";
  }
}

int PinnedMemory::Allocate(size_t size,
                           std::shared_ptr<PinnedMemoryPool> mempool) {
  if (buffers_.size() > 0) {
    LOG(ERROR) << "Memory already allocated";
    return 1;
  }

  mempool_ = mempool;
  return mempool_->Allocate(size, buffers_);
}

std::vector<char*>& PinnedMemory::get() { return buffers_; }
