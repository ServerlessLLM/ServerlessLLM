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

// #include "cuda_memory_pool.h"

class CudaMemory {
 public:
  CudaMemory();
  ~CudaMemory();

  // Disable copying and moving
  CudaMemory(const CudaMemory&) = delete;
  CudaMemory& operator=(const CudaMemory&) = delete;
  CudaMemory(CudaMemory&&) = delete;
  CudaMemory& operator=(CudaMemory&&) = delete;

  int Allocate(size_t size, int device);
  void* get() const;
  cudaIpcMemHandle_t getHandle() const;

 private:
  void* data_;
  cudaIpcMemHandle_t handle_;
  size_t size_;
  int device_;
};
