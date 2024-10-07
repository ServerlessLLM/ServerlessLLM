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
#include "checkpoint_store.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <thread>

#include "error_handling.h"

CheckpointStore::CheckpointStore(const std::string& storage_path,
                                 size_t memory_pool_size, int num_thread,
                                 size_t chunk_size)
    : storage_path_(storage_path),
      memory_pool_size_(memory_pool_size),
      num_thread_(num_thread),
      chunk_size_(chunk_size) {
  // Get number of GPUs
  cudaGetDeviceCount(&num_gpus_);
  LOG(INFO) << "Number of GPUs: " << num_gpus_;

  LOG(INFO) << "I/O threads: " << num_thread
            << ", chunk size: " << chunk_size / MB << "MB";
  LOG(INFO) << "Storage path: " << storage_path_;

  for (size_t i = 0; i < num_gpus_; ++i) {
    cudaSetDevice(i);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, i);

    // Get GPU UUID
    char uuidStr[80];
    snprintf(
        uuidStr, sizeof(uuidStr),
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        (unsigned char)props.uuid.bytes[0], (unsigned char)props.uuid.bytes[1],
        (unsigned char)props.uuid.bytes[2], (unsigned char)props.uuid.bytes[3],
        (unsigned char)props.uuid.bytes[4], (unsigned char)props.uuid.bytes[5],
        (unsigned char)props.uuid.bytes[6], (unsigned char)props.uuid.bytes[7],
        (unsigned char)props.uuid.bytes[8], (unsigned char)props.uuid.bytes[9],
        (unsigned char)props.uuid.bytes[10],
        (unsigned char)props.uuid.bytes[11],
        (unsigned char)props.uuid.bytes[12],
        (unsigned char)props.uuid.bytes[13],
        (unsigned char)props.uuid.bytes[14],
        (unsigned char)props.uuid.bytes[15]);

    gpu_info_map_[i].uuid_ = std::string(uuidStr);
    LOG(INFO) << "GPU " << i << " UUID: " << gpu_info_map_[i].uuid_;

    // create stream
    cudaError_t err = cudaStreamCreate(&gpu_info_map_[i].stream_);
    if (err != cudaSuccess) {
      LOG(FATAL) << "cudaStreamCreate error: " << cudaGetErrorString(err);
    }
  }

  // Create a memory pool
  memory_pool_ =
      std::make_shared<PinnedMemoryPool>(memory_pool_size_, chunk_size_);
  LOG(INFO) << "Memory pool created with " << memory_pool_size_ / GB << "GB";
}

CheckpointStore::~CheckpointStore() { ClearMem(); }

int64_t CheckpointStore::RegisterModelInfo(const std::string& model_path) {
  std::unique_lock<std::mutex> lock_info(model_info_mutex_);
  if (model_map_.find(model_path) != model_map_.end()) {
    // LOG(WARNING) << "Model " << model_path << " is already regfistered";
    auto model = model_map_.at(model_path);
    return model->GetModelSize();
  }

  auto model = std::make_shared<Model>(model_path);

  int ret = model->Initialize(storage_path_);
  if (ret != 0) {
    LOG(ERROR) << "Failed to initialize model " << model_path;
    return ret;
  }

  model_map_[model_path] = model;

  LOG(INFO) << "Model " << model_path << " is registered";

  return model->GetModelSize();
}

int CheckpointStore::LoadModelFromDisk(const std::string& model_path) {
  std::unique_lock<std::mutex> lock_info(model_info_mutex_);
  auto model = GetModelPtr(model_path);
  if (model == nullptr) {
    LOG(ERROR) << "Model " << model_path << " is not registered";
    return -1;
  }
  lock_info.unlock();

  // Allocate memory
  lock_info.lock();
  int remaining_size = model->AllocatePinnedMemory(memory_pool_);
  if (remaining_size < 0) {
    LOG(ERROR) << "Failed to allocate memory for model " << model_path;
    return -1;
  } else if (remaining_size > 0) {
    int mem_chunks_needed = remaining_size;
    std::vector<std::pair<std::string,
                          std::chrono::time_point<std::chrono::system_clock>>>
        model_last_access_time(model_last_access_time_.begin(),
                               model_last_access_time_.end());
    std::sort(model_last_access_time.begin(), model_last_access_time.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    for (const auto& [model_path, last_access_time] : model_last_access_time) {
      auto& model = model_map_[model_path];
      int freed_chunks =
          model->TryFreeHost();  // try to free memory, non-blocking
      LOG(INFO) << "Free " << freed_chunks << " chunks, remaining "
                << mem_chunks_needed;
      if (freed_chunks > 0) {
        LOG(INFO) << "Model " << model_path << " is freed from memory";
        mem_chunks_needed -= freed_chunks;
        if (mem_chunks_needed <= 0) {
          break;
        }
      }
    }
    if (mem_chunks_needed > 0) {
      LOG(ERROR) << "Failed to free enough memory for model " << model_path;
      return -1;
    }
    ssize_t remaining_size = model->AllocatePinnedMemory(memory_pool_);
    if (remaining_size < 0) {
      LOG(ERROR) << "Failed to allocate memory for model " << model_path;
      return -1;
    }
  }
  model_last_access_time_[model_path] = std::chrono::system_clock::now();
  lock_info.unlock();

  int ret = model->ToHost(num_thread_);

  if (ret != 0) {
    LOG(ERROR) << "Failed to load model " << model_path << " to host";
    if (model->FreeHost() != 0) {
      LOG(ERROR) << "Failed to free memory for model " << model_path;
    }
    return ret;
  }

  return ret;
}

int CheckpointStore::LoadModelFromDiskAsync(const std::string& model_path) {
  std::unique_lock<std::mutex> lock_info(model_info_mutex_);
  async_tasks_.emplace(std::async(std::launch::async, [this, model_path]() {
    return LoadModelFromDisk(model_path);
  }));

  return 0;
}

int CheckpointStore::LoadModelFromMem(
    const std::string& model_path, const std::string& replica_uuid,
    const MemCopyHandleListMap& gpu_memory_handles,
    const MemCopyChunkListMap& mem_copy_chunks) {
  // Sanity check
  if (model_path.empty() || replica_uuid.empty()) {
    LOG(ERROR) << "Model name or replica uuid is empty";
    return -1;
  } else if (mem_copy_chunks.empty()) {
    LOG(ERROR) << "No memory copy chunk provided";
    return -1;
  } else if (gpu_memory_handles.size() != mem_copy_chunks.size()) {
    LOG(ERROR) << "Mismatch between memory handles "
               << gpu_memory_handles.size() << " and memory copy chunks "
               << mem_copy_chunks.size();
    return -1;
  }
  std::unique_lock<std::mutex> lock_info(model_info_mutex_);
  auto model = GetModelPtr(model_path);
  if (model == nullptr) {
    LOG(ERROR) << "Model " << model_path << " is not registered";
    return -1;
  }
  model_last_access_time_[model_path] = std::chrono::system_clock::now();
  lock_info.unlock();

  LOG(INFO) << "Loading model " << model_path;

  // Convert device uuid to device id
  std::unordered_map<int, MemCopyChunkList> converted_mem_copy_chunks;
  for (auto& [device_id, gpu_info] : gpu_info_map_) {
    if (mem_copy_chunks.find(gpu_info.uuid_) == mem_copy_chunks.end()) {
      continue;
    }
    converted_mem_copy_chunks[device_id] = mem_copy_chunks.at(gpu_info.uuid_);
  }

  std::unordered_map<int, MemCopyHandleList> converted_mem_copy_handles;
  for (auto& [device_id, gpu_info] : gpu_info_map_) {
    if (gpu_memory_handles.find(gpu_info.uuid_) == gpu_memory_handles.end()) {
      continue;
    }
    converted_mem_copy_handles[device_id] =
        gpu_memory_handles.at(gpu_info.uuid_);
  }
  // Convert memory handles to device pointers
  auto device_ptrs = GetDevicePtrsFromMemHandles(gpu_memory_handles);

  auto ret = model->ToGpu(replica_uuid, device_ptrs, converted_mem_copy_chunks,
                          converted_mem_copy_handles);

  // TODO: check if the model is loaded successfully
  if (ret != 0) {
    LOG(ERROR) << "Failed to load model " << model_path << " to GPU";
    if (model->FreeGpu(replica_uuid) != 0) {
      LOG(ERROR) << "Failed to free memory for model " << model_path;
    }
  }

  return ret;
}

int CheckpointStore::LoadModelFromMemAsync(
    const std::string& model_path, const std::string& replica_uuid,
    const std::unordered_map<std::string, MemCopyHandleList>&
        gpu_memory_handles,
    const std::unordered_map<std::string, MemCopyChunkList>& mem_copy_chunks) {
  std::unique_lock<std::mutex> lock_info(model_info_mutex_);
  async_tasks_.emplace(std::async(
      std::launch::async,
      [this, model_path, replica_uuid, gpu_memory_handles, mem_copy_chunks]() {
        return LoadModelFromMem(model_path, replica_uuid, gpu_memory_handles,
                                mem_copy_chunks);
      }));

  return 0;
}

int CheckpointStore::WaitModelInGpu(const std::string& model_path,
                                    const std::string& replica_uuid) {
  // check if the model is in memory
  std::shared_ptr<Model> model;
  std::unique_lock<std::mutex> lock_info(model_info_mutex_);
  if (model_map_.find(model_path) == model_map_.end()) {
    LOG(ERROR) << "Model " << model_path << " is not registered";
    return 1;
  }
  model = model_map_[model_path];
  lock_info.unlock();

  return model->WaitInGpu(replica_uuid);
}

int CheckpointStore::ClearMem() {
  std::unique_lock<std::mutex> lock_info(model_info_mutex_);
  for (auto& [model_path, model] : model_map_) {
    LOG(INFO) << "Unloading model " << model_path;
    int ret = model->FreeHost();
    if (ret != 0) {
      LOG(ERROR) << "Failed to free memory for model " << model_path;
    }
  }
  model_map_.clear();
  LOG(INFO) << "All models unloaded from memory\n";
  return 0;
}

int CheckpointStore::UnloadModelFromHost(const std::string& model_path) {
  std::unique_lock<std::mutex> lock_info(model_info_mutex_);
  if (model_map_.find(model_path) == model_map_.end()) {
    LOG(ERROR) << "Model " << model_path << " is not registered";
    return 1;
  }
  auto model = model_map_.at(model_path);
  lock_info.unlock();

  return model->FreeHost();
}

ModelPtr CheckpointStore::GetModelPtr(const std::string& model_path) {
  if (model_map_.find(model_path) == model_map_.end()) {
    LOG(ERROR) << "Model " << model_path << " is not registered";
    return nullptr;
  }
  return model_map_.at(model_path);
}

MemPtrListMap CheckpointStore::GetDevicePtrsFromMemHandles(
    const MemCopyHandleListMap& memory_handles) {
  MemPtrListMap gpu_ptrs;
  for (const auto& [device_id, gpu_info] : gpu_info_map_) {
    const std::string& uuid = gpu_info.uuid_;
    if (memory_handles.find(uuid) == memory_handles.end()) {
      continue;
    }
    auto& handle_list = memory_handles.at(uuid);
    for (const auto& handle : handle_list) {
      // Convert handle string to cuda handle
      cudaIpcMemHandle_t* cuda_handle =
          reinterpret_cast<cudaIpcMemHandle_t*>(const_cast<char*>(
              reinterpret_cast<const char*>(handle.cuda_ipc_handle_.data())));
      void* ptr = nullptr;

      cudaSetDevice(device_id);
      cudaError_t err = cudaIpcOpenMemHandle(&ptr, *cuda_handle,
                                             cudaIpcMemLazyEnablePeerAccess);
      if (err != cudaSuccess || ptr == nullptr) {
        LOG(ERROR) << "Failed to open cuda handle on device " << device_id
                   << " error: " << cudaGetErrorString(err);
        exit(1);
      }

      gpu_ptrs[device_id].push_back(ptr);
    }
  }
  return gpu_ptrs;
}