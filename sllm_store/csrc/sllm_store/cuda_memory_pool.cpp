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
#include "cuda_memory_pool.h"

#include <glog/logging.h>

#include <mutex>

CudaMemoryPool::CudaMemoryPool(int device_count, size_t size_per_device)
    : device_count_(device_count), size_per_device_(size_per_device) {
  pool_.resize(device_count);
  handles_.resize(device_count);
  free_list_.resize(device_count);
  LOG(INFO) << "Creating CudaMemoryPool with " << device_count
            << " devices, each with " << size_per_device << " bytes";
  for (int i = 0; i < device_count; ++i) {
    cudaError_t err = cudaSetDevice(i);
    if (err != cudaSuccess) {
      LOG(FATAL) << "Failed to set device: " << cudaGetErrorString(err);
    }
    void* ptr = nullptr;
    err = cudaMalloc(&ptr, size_per_device);
    if (err != cudaSuccess) {
      LOG(FATAL) << "Failed to allocate memory on device " << i << ": "
                 << cudaGetErrorString(err);
    }
    err = cudaIpcGetMemHandle(&handles_[i], ptr);
    if (err != cudaSuccess) {
      LOG(FATAL) << "Error getting GPU memory handle "
                 << cudaGetErrorString(err);
    }
    pool_[i] = ptr;
    free_list_[i] = true;
  }
}

CudaMemoryPool::~CudaMemoryPool() {
  for (int i = 0; i < device_count_; ++i) {
    cudaError_t err = cudaSetDevice(i);
    if (err != cudaSuccess) {
      LOG(FATAL) << "Failed to set device: " << cudaGetErrorString(err);
    }
    err = cudaFree(pool_[i]);
    if (err != cudaSuccess) {
      LOG(FATAL) << "Failed to free memory on device " << i << ": "
                 << cudaGetErrorString(err);
    }
  }
}

int CudaMemoryPool::Allocate(size_t size, int device_id, void*& ptr,
                             cudaIpcMemHandle_t& handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (device_id < 0 || device_id >= device_count_) {
    LOG(ERROR) << "Invalid device id " << device_id;
    return -1;
  } else if (size > size_per_device_) {
    LOG(ERROR) << "Requested size " << size << " exceeds size per device "
               << size_per_device_;
    return -1;
  } else if (!free_list_[device_id]) {
    LOG(ERROR) << "Device " << device_id << " is not free";
    return -1;
  }

  ptr = pool_[device_id];
  handle = handles_[device_id];
  free_list_[device_id] = false;
  return 0;
}

int CudaMemoryPool::Deallocate(int device_id, void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (device_id < 0 || device_id >= device_count_) {
    LOG(ERROR) << "Invalid device id " << device_id;
    return -1;
  } else if (free_list_[device_id]) {
    LOG(ERROR) << "Device " << device_id << " is already free";
    return -1;
  } else if (ptr != pool_[device_id]) {
    LOG(ERROR) << "Invalid pointer " << ptr;
    return -1;
  }

  free_list_[device_id] = true;
  return 0;
}
