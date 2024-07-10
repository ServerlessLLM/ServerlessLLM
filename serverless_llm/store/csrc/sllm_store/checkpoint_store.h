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
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <queue>
#include <filesystem>

// Third-party library headers
#include <cuda_runtime.h>

// Own Headers
#include "cuda_memory.h"
// #include "cuda_memory_pool.h"
#include "pinned_memory.h"
#include "pinned_memory_pool.h"
#include "types_and_defs.h"
#include "model.h"

class CheckpointStore {
 public:
  CheckpointStore(const std::string& storage_path, size_t memory_pool_size,
                  int num_thread = 4, size_t chunk_size = 32L * 1024 * 1024,
                  bool registration_required = false);
  ~CheckpointStore();

  int RegisterModelInfo(const std::string& model_name);
  int LoadModelFromDisk(const std::string& model_name);
  int LoadModelFromDiskAsync(const std::string& model_name);
  int LoadModelFromMem(const std::string& model_name,
                       const std::string& replica_uuid,
                       const MemCopyHandleListMap& gpu_memory_handles,
                       const MemCopyChunkListMap& mem_copy_chunks);
  int LoadModelFromMemAsync(const std::string& model_name,
                            const std::string& replica_uuid,
                            const MemCopyHandleListMap& gpu_memory_handles,
                            const MemCopyChunkListMap& mem_copy_chunks);
  int WaitModelInGpu(const std::string& model_name,
                        const std::string& replica_uuid);
  int UnloadModelFromHost(const std::string& model_name);
  int ClearMem();
  void DeleteModelInfo(const std::string& model_name);
  void ClearModelMem(const std::string& model_name);
  void ClearModelGpuMem(const std::string& model_name,
                        const std::string& replica_uuid);

 private:
  // A GPU info struct
  struct GpuInfo {
    // uuid
    std::string uuid_;
    // int device_id_;
    size_t total_memory_ = 0;
    size_t free_memory_ = 0;
    cudaStream_t stream_;
  };

  const std::filesystem::path storage_path_;
  int num_gpus_;
  std::unordered_map<int, GpuInfo> gpu_info_map_;
  std::unordered_map<std::string, std::shared_ptr<Model>> model_map_;
  std::unordered_map<std::string, std::chrono::time_point<std::chrono::system_clock>>
      model_last_access_time_;
  std::mutex model_info_mutex_;
  const size_t memory_pool_size_;
  std::shared_ptr<PinnedMemoryPool> memory_pool_;
  int num_thread_;
  size_t chunk_size_;
  bool registration_required_;  // If true, the model must be registered before loading

  std::queue<std::future<int>> async_tasks_;

  size_t GetNumChunkFromTensorSize(size_t tensor_size);
  ModelPtr GetModelPtr(const std::string& model_name);
  GpuReplicaPtr NewGpuReplica(const std::shared_ptr<Model>& model,
                              const std::string& replica_uuid);
  int InitializeModel(const std::shared_ptr<Model>& model);
  int AllocatePinnedMemory(const std::shared_ptr<Model>& model);
//   int DispatchTensorToGpu(
//       const std::shared_ptr<Model>& model,
//       const std::shared_ptr<GpuReplica>& gpu_replica,
//       const std::unordered_map<int, MemCopyChunkList>& mem_copy_chunks);
  std::vector<std::tuple<int, size_t, size_t>> CalculateChunks(size_t offset,
                                                               size_t size);
  int AllocateCudaMemory(
      const std::shared_ptr<GpuReplica>& gpu_replica,
      std::vector<std::pair<int, uint64_t>> gpu_memory_sizes);
  ModelPtr GetModelByName(const std::string& model_name);
  MemPtrListMap GetDevicePtrsFromMemHandles(const MemCopyHandleListMap& memory_handles);
};