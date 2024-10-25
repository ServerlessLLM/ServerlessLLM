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

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Third-party library headers
#include <cuda_runtime.h>
#include <glog/logging.h>

#include "error_handling.h"
#include "pinned_memory.h"
#include "types_and_defs.h"

struct GpuReplica {
  std::condition_variable cv_;
  MemoryState state_ = MemoryState::UNINITIALIZED;

  std::unordered_map<int, std::shared_ptr<BatchQueue>> gpu_loading_queue_;
  MemPtrListMap device_ptrs_;
};
using GpuReplicaPtr = std::shared_ptr<GpuReplica>;

class Model {
 public:
  Model(const std::filesystem::path& model_path) : model_path_(model_path) {}
  int Initialize(const std::filesystem::path storage_path);
  int AllocatePinnedMemory(std::shared_ptr<PinnedMemoryPool> pool);
  int ToHost(int num_threads);
  int ToGpu(const std::string& replica_uuid, const MemPtrListMap& device_ptrs,
            const std::unordered_map<int, MemCopyChunkList>& mem_copy_chunks,
            const std::unordered_map<int, MemCopyHandleList>& mem_copy_handles);
  int WaitInHost();
  int WaitInGpu(const std::string& replica_uuid);
  int FreeGpu(const std::string& replica_uuid);
  int FreeHost();
  int TryFreeHost();
  uint64_t GetModelSize() const { return model_size_; }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  MemoryState state_ = MemoryState::UNINITIALIZED;

  // Model path
  const std::string model_path_;

  // Model info needs to be initialized
  size_t model_size_;
  std::vector<size_t> partition_sizes_;
  std::vector<std::filesystem::path> partition_paths_;
  std::shared_ptr<PinnedMemory> pinned_mem_;

  std::unordered_map<std::string, GpuReplicaPtr> gpu_replicas_;

  std::shared_ptr<BatchVector> host_ptr_vector_;

  std::vector<std::tuple<int, size_t, size_t>> MapDataToChunks(
      size_t offset, size_t size, size_t chunk_size);
  int DispatchToGpu(
      const std::shared_ptr<GpuReplica>& gpu_replica,
      const std::unordered_map<int, MemCopyChunkList>& mem_copy_chunks,
      const std::unordered_map<int, MemCopyHandleList>& mem_copy_handles);
};
using ModelPtr = std::shared_ptr<Model>;