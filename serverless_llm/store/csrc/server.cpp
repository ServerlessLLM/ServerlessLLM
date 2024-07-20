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
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

#ifdef USE_HIP
  #include "checkpoint_store_hip.h"
#else
  #include "checkpoint_store.h"
#endif

#include "storage.grpc.pb.h"
#include "storage.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using storage::ClearMemRequest;
using storage::ClearMemResponse;
using storage::ConfirmModelRequest;
using storage::ConfirmModelResponse;
using storage::LoadModelRequest;
using storage::LoadModelResponse;
using storage::UnloadModelRequest;
using storage::UnloadModelResponse;

DEFINE_string(storage_path, "./models", "storage path");
DEFINE_int32(server_port, 8073, "Server port");

// system parameter
DEFINE_int32(num_thread, 4, "Number of I/O threads");
DEFINE_int32(chunk_size, 32, "Chunk size in MB");
DEFINE_int64(mem_pool_size, 32, "Memory pool size in GB");
DEFINE_int64(disk_size, 128, "Disk size in GB");

DEFINE_bool(registration_required, false,
            "Require registration before loading model");

// glog related
DECLARE_bool(logtostderr);
DECLARE_string(log_dir);

const int kMaxRetry = 5;

class CheckpointStoreServer final : public storage::Storage::Service {
 public:
  CheckpointStoreServer(const std::string& storage_path, size_t mem_pool_size,
                        size_t disk_size, int num_thread, int chunk_size,
                        bool registration_required)
      : registration_required_(registration_required) {
    if (mem_pool_size == 0) {
      LOG(FATAL) << "mem_pool_size is 0";
    }
    if (storage_path.empty()) {
      LOG(FATAL) << "storage_path is empty";
    }

    storage_ = std::make_unique<CheckpointStore>(storage_path, mem_pool_size,
                                                 num_thread, chunk_size);
  }

  Status LoadModelAsync(ServerContext* context, const LoadModelRequest* request,
                        LoadModelResponse* response) override {
    const std::string& model_name = request->model_name();
    if (model_name.empty()) {
      LOG(ERROR) << "model_name is empty";
      return Status::CANCELLED;
    }

    if (!registration_required_) {
      int64_t model_size = storage_->RegisterModelInfo(model_name);
      if (model_size < 0) {
        LOG(ERROR) << "RegisterModel failed";
        return Status::CANCELLED;
      }
    }

    auto device_type = request->target_device_type();

    if (device_type == storage::DEVICE_TYPE_CPU) {
      int ret = storage_->LoadModelFromDiskAsync(model_name);
      if (ret != 0) {
        LOG(ERROR) << "LoadModel failed";
        return Status::CANCELLED;
      }
    } else if (device_type == storage::DEVICE_TYPE_GPU) {
      const std::string& replica_uuid = request->replica_uuid();
      if (replica_uuid.empty()) {
        LOG(ERROR) << "replica_uuid is empty";
        return Status::CANCELLED;
      }
      std::unordered_map<std::string, MemCopyHandleList> gpu_memory_handles;
      for (const auto& [device_uuid, handle_list] : request->handles()) {
        for (const auto& handle : handle_list.handles()) {
          MemCopyHandle mem_copy_handle;
          mem_copy_handle.cuda_ipc_handle_ = handle.cuda_ipc_handle();
          gpu_memory_handles[device_uuid].push_back(mem_copy_handle);
        }
      }
      std::unordered_map<std::string, MemCopyChunkList> mem_copy_chunks;
      for (const auto& [device_uuid, chunk_list] : request->chunks()) {
        for (const auto& chunk : chunk_list.chunks()) {
          MemCopyChunk mem_copy_chunk;
          mem_copy_chunk.src_offset_ = chunk.src_offset();
          mem_copy_chunk.size_ = chunk.size();
          mem_copy_chunk.dst_offset_ = chunk.dst_offset();
          mem_copy_chunk.handle_idx_ = chunk.handle_idx();
          mem_copy_chunks[device_uuid].push_back(mem_copy_chunk);
        }
      }
      int ret = storage_->LoadModelFromMemAsync(
          model_name, replica_uuid, gpu_memory_handles, mem_copy_chunks);
      if (ret != 0) {
        LOG(ERROR) << "LoadModel failed";
        return Status::CANCELLED;
      }
    } else {
      LOG(ERROR) << "Unsupported device type: " << device_type;
      return Status::CANCELLED;
    }

    LOG(INFO) << "LoadModel: success " << request->model_name()
              << " with target " << device_type;

    return Status::OK;
  }

  Status ConfirmModel(ServerContext* context,
                      const ConfirmModelRequest* request,
                      ConfirmModelResponse* response) override {
    const std::string& model_name = request->model_name();
    const std::string& replica_uuid = request->replica_uuid();
    auto device_type = request->target_device_type();

    if (model_name.empty()) {
      LOG(ERROR) << "model_name is empty";
      return Status::CANCELLED;
    }

    LOG(INFO) << "Confirm model " << model_name << " replica " << replica_uuid;

    if (device_type != storage::DEVICE_TYPE_GPU) {
      LOG(ERROR) << "Unsupported device type: " << device_type;
      return Status::CANCELLED;
    }

    bool success = false;
    for (int i = 0; i < kMaxRetry; ++i) {
      int ret = storage_->WaitModelInGpu(model_name, replica_uuid);
      if (ret == 0) {
        success = true;
        break;
      }
      LOG(INFO) << "Confirm model " << model_name << " replica " << replica_uuid
                << " failed with retry " << i;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (!success) {
      LOG(ERROR) << "Confirm model " << model_name << " replica "
                 << replica_uuid << " failed";
      return Status::CANCELLED;
    }
    LOG(INFO) << "Confirm model " << model_name << " replica " << replica_uuid
              << " success";

    return Status::OK;
  }

  Status UnloadModel(ServerContext* context, const UnloadModelRequest* request,
                     UnloadModelResponse* response) override {
    const std::string& model_name = request->model_name();
    const std::string& replica_uuid = request->replica_uuid();

    if (model_name.empty()) {
      LOG(ERROR) << "model_name is empty";
      return Status::CANCELLED;
    }

    auto device_type = request->target_device_type();
    std::function<int(const std::string&)> unload_func;
    if (device_type == storage::DEVICE_TYPE_CPU) {
      unload_func = std::bind(&CheckpointStore::UnloadModelFromHost,
                              storage_.get(), std::placeholders::_1);
    } else {
      LOG(ERROR) << "Unsupported device type: " << device_type;
      return Status::CANCELLED;
    }

    LOG(INFO) << "UnloadModel: start " << model_name << " with target "
              << request->target_device_type();

    // retry 5 times
    bool success = false;
    for (int i = 0; i < kMaxRetry; ++i) {
      int ret = unload_func(model_name);
      if (ret == 0) {
        success = true;
        break;
      }
      LOG(INFO) << "UnloadModel failed for model " << model_name
                << " with retry " << i;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (!success) {
      LOG(ERROR) << "UnloadModel failed for model " << model_name;
      return Status::CANCELLED;
    }

    LOG(INFO) << "UnloadModel: success " << model_name << " with target "
              << request->target_device_type();

    return Status::OK;
  }

  Status ClearMem(ServerContext* context, const ClearMemRequest* request,
                  ClearMemResponse* response) override {
    LOG(INFO) << "ClearMem";
    int ret = storage_->ClearMem();
    if (ret != 0) {
      LOG(ERROR) << "ClearMem failed";
      return Status::CANCELLED;
    }

    return Status::OK;
  }

  Status RegisterModel(ServerContext* context,
                       const storage::RegisterModelRequest* request,
                       storage::RegisterModelResponse* response) override {
    const std::string& model_name = request->model_name();
    if (model_name.empty()) {
      LOG(ERROR) << "model_name is empty";
      return Status::CANCELLED;
    }

    int64_t model_size = storage_->RegisterModelInfo(model_name);
    if (model_size < 0) {
      LOG(ERROR) << "RegisterModel failed";
      return Status::CANCELLED;
    }
    response->set_model_size(model_size);

    return Status::OK;
  }

  Status GetServerConfig(ServerContext* context,
                         const storage::GetServerConfigRequest* request,
                         storage::GetServerConfigResponse* response) override {
    response->set_chunk_size(storage_->GetChunkSize());

    return Status::OK;
  }

 private:
  std::unique_ptr<CheckpointStore> storage_;

  std::mutex disk_mutex_;
  std::mutex queue_mutex_;
  std::queue<std::condition_variable*> wait_queue_;

  bool registration_required_ = false;
};

void RunServer(const std::string& server_address,
               const std::string& storage_path, size_t mem_pool_size,
               size_t disk_size, int num_thread, size_t chunk_size,
               bool registration_required) {
  CheckpointStoreServer service(storage_path, mem_pool_size, disk_size,
                                num_thread, chunk_size, registration_required);

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address;
  server->Wait();
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::string server_address = "0.0.0.0:" + std::to_string(FLAGS_server_port);
  std::string storage_path = FLAGS_storage_path;
  size_t mem_pool_size = (size_t)FLAGS_mem_pool_size * 1024 * 1024 * 1024;
  size_t disk_size = (size_t)FLAGS_disk_size * 1024 * 1024 * 1024;
  int num_thread = FLAGS_num_thread;
  size_t chunk_size = (size_t)FLAGS_chunk_size * 1024 * 1024;
  bool registration_required = FLAGS_registration_required;

  if (storage_path.back() != '/') {
    storage_path += "/";
  }

  const std::string home_dir = std::getenv("HOME") ? std::getenv("HOME") : "";
  const std::string kLogDir = home_dir + "/.checkpoint_store/logs";

  try {
    // Create the log directory if it does not exist
    if (!std::filesystem::exists(kLogDir)) {
      std::filesystem::create_directories(kLogDir);
      // std::cout << "Log directory created successfully.\n";
      LOG(INFO) << "Log directory created successfully.";
    } else {
      // std::cout << "Log directory already exists.\n";
      LOG(INFO) << "Log directory already exists.";
    }
  } catch (const std::filesystem::filesystem_error& e) {
    // std::cerr << "Filesystem error while creating the log directory: " <<
    // e.what() << '\n';
    LOG(ERROR) << "Filesystem error while creating the log directory: "
               << e.what();
    return -1;
  }

  FLAGS_log_dir = kLogDir;
  FLAGS_alsologtostderr = 1;

  google::InitGoogleLogging(argv[0]);

  google::InstallFailureSignalHandler();

  RunServer(server_address, storage_path, mem_pool_size, disk_size, num_thread,
            chunk_size, registration_required);
  google::ShutdownGoogleLogging();

  return 0;
}