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
#include "cuda_memory.h"

#include <glog/logging.h>

CudaMemory::CudaMemory() : data_(nullptr), size_(0), device_(-1) {}

CudaMemory::~CudaMemory() {
  if (data_) {  // Ensure we have data to free
    cudaFree(data_);
  }
}

int CudaMemory::Allocate(size_t size, int device) {
  if (data_) {
    LOG(ERROR) << "Memory already allocated\n";
    return 1;  // Indicate error
  }

  // Check if device and size are valid
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (device >= deviceCount || size == 0) {
    LOG(ERROR) << "Invalid device or size\n";
    return 1;  // Indicate error
  }

  // Set device and allocate memory on it
  cudaSetDevice(device);
  cudaError_t status = cudaMalloc(&data_, size);
  if (status != cudaSuccess) {
    LOG(ERROR) << "Failed to allocate memory on device " << device << ": "
               << cudaGetErrorString(status) << "\n";
    return status;
  }
  device_ = device;
  size_ = size;

  // Get IPC handle
  status = cudaIpcGetMemHandle(&handle_, data_);
  if (status != cudaSuccess) {
    cudaFree(data_);
    data_ = nullptr;
    LOG(ERROR) << "Failed to get IPC handle: " << cudaGetErrorString(status)
               << "\n";
    return status;
  }

  return cudaSuccess;  // Indicate success
}

void* CudaMemory::get() const { return data_; }

cudaIpcMemHandle_t CudaMemory::getHandle() const { return handle_; }
