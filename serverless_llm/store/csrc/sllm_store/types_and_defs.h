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

#include <iostream>
#include <string>
#include <unordered_map>

#include "concurrent_queue.h"
#include "concurrent_vector.h"
#include "memory_state.h"

struct Batch {
  size_t chunk_id_ = 0;
  size_t size_ = 0;
};
typedef ConcurrentVector<Batch> BatchVector;

struct GpuBatch {
  size_t chunk_id_ = 0;
  size_t chunk_offset_ = 0;
  size_t size_ = 0;
  size_t gpu_offset_ = 0;
  size_t handle_idx_ = 0;
};
typedef ConcurrentQueue<GpuBatch> BatchQueue;

struct FileChunk {
  int fd_;
  size_t file_offset_;
  size_t size_;
  size_t chunk_id_;
  size_t chunk_offset_;
};

#define KB (1024LL)
#define MB (1024LL * KB)
#define GB (1024LL * MB)

// using DeviceMap = std::unordered_map<std::string, int>;
struct MemCopyChunk {
  size_t src_offset_ = 0;
  size_t size_ = 0;
  size_t dst_offset_ = 0;
  size_t handle_idx_ = 0;
};
using MemCopyChunkList = std::vector<MemCopyChunk>;

struct MemCopyHandle {
  std::string cuda_ipc_handle_;
};
using MemCopyHandleList = std::vector<MemCopyHandle>;

typedef std::unordered_map<std::string, MemCopyHandleList> MemCopyHandleListMap;
typedef std::unordered_map<std::string, MemCopyChunkList> MemCopyChunkListMap;
typedef std::unordered_map<int, std::vector<void*>> MemPtrListMap;

// device_id, chunk_offset, size, gpu_offset. handle_idx
typedef std::tuple<int, size_t, size_t, size_t, size_t> GpuChunk;
