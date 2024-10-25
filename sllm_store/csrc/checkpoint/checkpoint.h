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

#include <torch/extension.h>
#include <torch/script.h>  // One-stop header.

#include <string>
#include <unordered_map>

std::unordered_map<std::string, uint64_t> SaveTensors(
    std::vector<std::string> tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path);

std::unordered_map<std::string, torch::Tensor> RestoreTensors(
    const std::unordered_map<
        std::string, std::tuple<std::vector<int64_t>, std::vector<int64_t>,
                                std::string>>& meta_state_dict,
    const std::unordered_map<int, void*>& memory_base_address,
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>&
        tensor_device_offsets);

// {dev_id: ptr}
std::unordered_map<int, void*> AllocateCudaMemory(
    const std::unordered_map<int, size_t>& tensor_sizes);
std::unordered_map<int, std::string> GetCudaMemoryHandles(
    const std::unordered_map<int, void*>& memory_ptrs);
std::unordered_map<int, std::vector<std::string>> GetCudaMemoryHandles(
    const std::unordered_map<int, std::vector<void*>>& memory_ptrs);
std::unordered_map<int, std::string> GetDeviceUuidMap();

std::unordered_map<std::string, int> GetGpuUUID();