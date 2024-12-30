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
#include <torch/extension.h>

#include "checkpoint_store.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<CheckpointStore>(m, "CheckpointStore")
      .def(py::init<const std::string&, size_t, int, size_t>(),
           py::arg("storage_path"), py::arg("memory_pool_size"),
           py::arg("num_thread"), py::arg("chunk_size"))
      .def("register_model_info", &CheckpointStore::RegisterModelInfo,
           py::arg("model_path"),
           "Register the model information and return its size.")
      .def("load_model_from_disk", &CheckpointStore::LoadModelFromDisk,
           py::arg("model_path"), "Load a model from disk synchronously.")
      .def("load_model_from_disk_async",
           &CheckpointStore::LoadModelFromDiskAsync, py::arg("model_path"),
           "Load a model from disk asynchronously.")
      .def("load_model_from_mem", &CheckpointStore::LoadModelFromMem,
           py::arg("model_path"), py::arg("replica_uuid"),
           py::arg("gpu_memory_handles"), py::arg("mem_copy_chunks"),
           "Load a model from memory synchronously.")
      .def("load_model_from_mem_async", &CheckpointStore::LoadModelFromMemAsync,
           py::arg("model_path"), py::arg("replica_uuid"),
           py::arg("gpu_memory_handles"), py::arg("mem_copy_chunks"),
           "Load a model from memory asynchronously.")
      .def("wait_model_in_gpu", &CheckpointStore::WaitModelInGpu,
           py::arg("model_path"), py::arg("replica_uuid"),
           "Wait for a model to be available in GPU memory.")
      .def("unload_model_from_host", &CheckpointStore::UnloadModelFromHost,
           py::arg("model_path"), "Unload a model from the host memory.")
      .def("clear_mem", &CheckpointStore::ClearMem,
           "Clear all allocated memory.")
      .def("get_mem_pool_size", &CheckpointStore::GetMemPoolSize,
           "Get the memory pool size.")
      .def("get_chunk_size", &CheckpointStore::GetChunkSize,
           "Get the chunk size.")
      .def("__repr__",
           [](const CheckpointStore& cs) { return "<CheckpointStore>"; });
}