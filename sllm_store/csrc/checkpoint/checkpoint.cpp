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
#include "checkpoint.h"

#include <ATen/cuda/CUDABlas.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <fcntl.h>
#include <nvml.h>
#include <sys/stat.h>
#include <torch/extension.h>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "progress_bar.h"
#include "tensor_writer.h"

#define BUFFER_SIZE 1 << 30

std::unordered_map<std::string, uint64_t> SaveTensors(
    std::vector<std::string> tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path) {
  std::string tensor_filename = std::filesystem::path(path) / "tensor.data";
  // make a tensor writer
  TensorWriter writer(tensor_filename);
  // make a tensor index
  std::unordered_map<std::string, uint64_t> tensor_offsets;
  // Some tensors may share the same data, so we need to record the data to
  // avoid duplication
  std::unordered_map<const char*, std::string> data_record;

  int total = tensor_names.size();
  int count = 0;

  for (const auto& name : tensor_names) {
    const auto& [base, size] = tensor_data[name];
    const char* data_ptr = reinterpret_cast<const char*>(base);
    if (data_record.find(data_ptr) != data_record.end()) {
      tensor_offsets[name] = tensor_offsets[data_record[data_ptr]];
      continue;
    }
    data_record[data_ptr] = name;

    uint64_t offset = writer.writeRecord(data_ptr, size);
    tensor_offsets[name] = offset;

    // Update progress bar
    count++;
    float progress = static_cast<float>(count) / total;
    showProgressBar(progress, "Saving tensors: ");
  }

  return tensor_offsets;
}

// Function to print the binary array in hexadecimal format
void printBinaryArrayInHex(const unsigned char* data, size_t size) {
  std::cout << "Data in Hex: ";
  for (size_t i = 0; i < size; ++i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(data[i]) << " ";
  }
  std::cout << std::dec
            << std::endl;  // Switch back to decimal for any future output
}

// Mapping from string to at::ScalarType
at::ScalarType stringToScalarType(const std::string& dtype_str) {
  static const std::unordered_map<std::string, at::ScalarType> dtype_map = {
      {"torch.float16", torch::kFloat16},  {"torch.float32", torch::kFloat32},
      {"torch.float64", torch::kFloat64},  {"torch.int16", torch::kInt16},
      {"torch.int32", torch::kInt32},      {"torch.int64", torch::kInt64},
      {"torch.uint8", torch::kUInt8},      {"torch.int8", torch::kInt8},
      {"torch.bfloat16", torch::kBFloat16}};

  auto it = dtype_map.find(dtype_str);
  if (it != dtype_map.end()) {
    return it->second;
  } else {
    throw std::invalid_argument("Unknown dtype string: " + dtype_str);
  }
}

std::unordered_map<std::string, torch::Tensor> RestoreTensors(
    const std::unordered_map<
        std::string, std::tuple<std::vector<int64_t>, std::vector<int64_t>,
                                std::string>>& meta_state_dict,
    const std::unordered_map<int, void*>& memory_base_address,
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>&
        tensor_device_offsets) {
  std::unordered_map<std::string, torch::Tensor> state_dict;
  std::unordered_set<void*> handled_memory;
  for (const auto& [device, tensor_offset] : tensor_device_offsets) {
    for (const auto& p : tensor_offset) {
      std::string name = p.first;
      if (memory_base_address.find(device) != memory_base_address.end()) {
        void* base_address = memory_base_address.at(device);
        uint64_t offset = reinterpret_cast<uint64_t>(base_address) + p.second;
        torch::Device tensor_device = (device == -1)
                                          ? torch::Device(torch::kCPU)
                                          : torch::Device(torch::kCUDA, device);
        auto [sizes, strides, type_str] = meta_state_dict.at(name);
        at::ScalarType dtype = stringToScalarType(type_str);
        // std::cerr << name << " " << sizes << " " << strides << " " << dtype
        // << std::endl;
        if (p.second == 0 &&
            handled_memory.find(base_address) == handled_memory.end()) {
          // Use appropriate deallocator based on device type
          auto deallocator =
              (device == -1)
                  ? [](void* ptr) { /* Do nothing for shared memory */ }
                  : [](void* ptr) { cudaFree(ptr); };
          torch::Tensor real_tensor = torch::from_blob(
              reinterpret_cast<void*>(offset), c10::makeArrayRef(sizes),
              c10::makeArrayRef(strides), deallocator,
              torch::TensorOptions().device(tensor_device).dtype(dtype));
          state_dict[name] = real_tensor;
          handled_memory.insert(base_address);
          // std::cerr << "Tensor " << name << " is restored to device " <<
          // device << std::endl;
        } else {
          /*std::cerr << "RestoreTensors checkpoint 3b" << std::endl;
          std::cerr << "-------- RESTORE DEBUG INFO --------\n";
          std::cerr << "Tensor name: " << name << "\n";
          std::cerr << "Device: " << device << "\n";
          std::cerr << "Shared memory base: " << base_address << "\n";
          std::cerr << "Offset: " << p.second << "\n";
          std::cerr << "Final tensor address: " <<
          reinterpret_cast<void*>(offset) << "\n"; std::cerr << "Expected
          address range: [" << base_address << ", "
          << static_cast<void*>((char*)base_address + 268435456) << ")\n";
          std::cerr << "cudaPointerGetAttributes:\n";

          cudaPointerAttributes attr;
          cudaError_t err = cudaPointerGetAttributes(&attr,
          reinterpret_cast<void*>(offset)); if (err == cudaSuccess) { std::cerr
          << "  devicePointer: " << attr.devicePointer << "\n"; std::cerr << "
          hostPointer:   " << attr.hostPointer << "\n"; std::cerr << "
          memoryType:    " << attr.type << "\n"; } else { std::cerr << "  ERROR:
          " << cudaGetErrorString(err) << "\n";
          }
          std::cerr << "------------------------------------\n";*/

          torch::Tensor real_tensor = torch::from_blob(
              reinterpret_cast<void*>(offset), sizes, strides, [](void* ptr) {},
              torch::TensorOptions().device(tensor_device).dtype(dtype));
          state_dict[name] = real_tensor;
        }
      } else {
        std::cerr << "Cannot find device " << device << std::endl;
        exit(1);
      }
    }
  }
  return state_dict;
}

std::unordered_map<std::string, torch::Tensor> RestoreTensorsShm(
    const std::unordered_map<
        std::string, std::tuple<std::vector<int64_t>, std::vector<int64_t>,
                                std::string>>& meta_state_dict,
    const std::unordered_map<int, std::vector<void*>>& memory_base_addresses,
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>&
        tensor_device_offsets,
    size_t chunk_size) {
  std::unordered_map<std::string, torch::Tensor> state_dict;

  for (const auto& [device, tensor_offsets] : tensor_device_offsets) {
    if (memory_base_addresses.find(device) == memory_base_addresses.end()) {
      std::cerr << "Cannot find base addresses for device " << device
                << std::endl;
      exit(1);
    }
    const auto& base_ptrs = memory_base_addresses.at(device);

    for (const auto& [tensor_name, flat_offset] : tensor_offsets) {
      size_t base_index = flat_offset / chunk_size;
      size_t offset_within_chunk = flat_offset % chunk_size;

      if (base_index >= base_ptrs.size()) {
        std::cerr << "Invalid base_index " << base_index << " for tensor '"
                  << tensor_name << "' on device " << device << std::endl;
        exit(1);
      }

      void* base_address = base_ptrs[base_index];
      void* final_address = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(base_address) + offset_within_chunk);

      std::cerr << "Tensor: " << tensor_name << ", Device: " << device
                << ", Base index: " << base_index
                << ", Offset within chunk: " << offset_within_chunk
                << ", Chunk size: " << chunk_size << std::endl;

      torch::Device tensor_device = (device == -1)
                                        ? torch::Device(torch::kCPU)
                                        : torch::Device(torch::kCUDA, device);

      auto [sizes, strides, type_str] = meta_state_dict.at(tensor_name);
      at::ScalarType dtype = stringToScalarType(type_str);

      torch::Tensor real_tensor = torch::from_blob(
          final_address, sizes, strides, [](void* ptr) {},
          torch::TensorOptions().device(tensor_device).dtype(dtype));

      std::cout << "  First few values: "
                << real_tensor.cpu().flatten().slice(0, 0, 10) << std::endl;

      state_dict[tensor_name] = real_tensor;
    }
  }
  std::cerr << "Restored " << state_dict.size()
            << " tensors from shared memory." << std::endl;
  return state_dict;
}

std::unordered_map<std::string, int> GetGpuUUID() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);  // Get the number of CUDA devices
  std::unordered_map<std::string, int> uuidToDeviceIdMap;

  for (int devId = 0; devId < deviceCount; ++devId) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, devId);  // Get properties for each device

    // Convert UUID bytes to string with unsigned char casting
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

    uuidToDeviceIdMap[std::string(uuidStr)] = devId;
  }

  return uuidToDeviceIdMap;
}

std::unordered_map<int, void*> AllocateCudaMemory(
    const std::unordered_map<int, size_t>& tensor_sizes) {
  std::unordered_map<int, void*> memory_ptrs;
  for (const auto& p : tensor_sizes) {
    int device = p.first;
    size_t size = p.second;
    void* ptr = nullptr;
    cudaSetDevice(device);
    cudaMalloc(&ptr, size);
    memory_ptrs[device] = ptr;
  }
  return memory_ptrs;
}

std::unordered_map<int, std::string> GetCudaMemoryHandles(
    const std::unordered_map<int, void*>& memory_ptrs) {
  std::unordered_map<int, std::string> memory_handles;
  for (const auto& p : memory_ptrs) {
    int device = p.first;
    void* ptr = p.second;
    cudaIpcMemHandle_t handle;
    cudaSetDevice(device);
    cudaIpcGetMemHandle(&handle, ptr);
    memory_handles[device] = std::string(reinterpret_cast<const char*>(&handle),
                                         sizeof(cudaIpcMemHandle_t));
  }
  return memory_handles;
}

std::unordered_map<int, std::vector<std::string>> GetCudaMemoryHandles(
    const std::unordered_map<int, std::vector<void*>>& memory_ptrs) {
  std::unordered_map<int, std::vector<std::string>> memory_handles;
  for (const auto& p : memory_ptrs) {
    auto device = p.first;
    const auto& ptrs = p.second;
    cudaIpcMemHandle_t handle;
    cudaSetDevice(device);

    std::vector<std::string> handles;
    for (const auto& ptr : ptrs) {
      cudaIpcGetMemHandle(&handle, ptr);
      handles.push_back(std::string(reinterpret_cast<const char*>(&handle),
                                    sizeof(cudaIpcMemHandle_t)));
    }
    memory_handles[device] = handles;
  }
  return memory_handles;
}

std::unordered_map<int, std::string> GetDeviceUuidMap() {
  std::unordered_map<std::string, int> gpu_uuid = GetGpuUUID();
  std::unordered_map<int, std::string> device_uuid_map;
  for (const auto& p : gpu_uuid) {
    if (device_uuid_map.find(p.second) != device_uuid_map.end()) {
      std::cerr << "Duplicate device id: " << p.second << std::endl;
      exit(1);
    }
    device_uuid_map[p.second] = p.first;
  }
  return device_uuid_map;
}