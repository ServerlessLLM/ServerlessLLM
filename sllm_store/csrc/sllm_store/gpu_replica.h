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
#include <future>
#include <unordered_map>

#include "types_and_defs.h"

class GpuReplica {
  std::condition_variable cv_;
  MemoryState state_ = MemoryState::UNINITIALIZED;

  std::unordered_map<int, std::shared_ptr<BatchQueue>> gpu_loading_queue_;
  std::unordered_map<int, void*> device_ptrs_;

  std::unordered_map<std::string, size_t> tensor_offsets_;

  void Clear();
};