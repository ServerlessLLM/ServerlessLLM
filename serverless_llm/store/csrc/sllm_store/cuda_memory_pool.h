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

#include <cuda_runtime.h>

#include <mutex>
#include <vector>

class CudaMemoryPool {
 public:
  CudaMemoryPool(int device_count, size_t size_per_device);
  CudaMemoryPool(const CudaMemoryPool&) = delete;
  CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;
  ~CudaMemoryPool();

  int Allocate(size_t size, int device_id, void*& ptr,
               cudaIpcMemHandle_t& handle);
  int Deallocate(int device_id, void* ptr);

 private:
  std::mutex mutex_;
  int device_count_;
  size_t size_per_device_;
  std::vector<void*> pool_;
  std::vector<cudaIpcMemHandle_t> handles_;
  std::vector<bool> free_list_;
};
