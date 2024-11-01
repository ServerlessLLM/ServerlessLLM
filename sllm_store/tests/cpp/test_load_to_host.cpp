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
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef USE_HIP
#include "checkpoint_store_hip.h"
#else
#include "checkpoint_store.h"
#endif

bool WriteBytesToFile(const std::string& file_path,
                      const std::vector<uint8_t>& data) {
  // Ensure the directory exists
  std::filesystem::create_directories(
      std::filesystem::path(file_path).parent_path());

  std::ofstream file(file_path, std::ios::binary);
  if (!file) {
    return false;  // Failed to open file
  }
  file.write(reinterpret_cast<const char*>(data.data()), data.size());
  return file.good();  // Return true if the write was successful
}

class CheckpointStoreTest : public ::testing::Test {
 protected:
  static CheckpointStore* storage;
  static size_t mem_pool_size;
  static std::string storage_path;
  static int num_thread;
  static size_t chunk_size;

  static void SetUpTestSuite() {
    mem_pool_size = 4L * 1024 * 1024 * 1024;  // 4GB
    storage_path = "./test_models";
    num_thread = 4;
    chunk_size = 4 * 1024 * 1024;  // 4MB

    storage = new CheckpointStore(storage_path, mem_pool_size, num_thread,
                                  chunk_size);
  }

  static void TearDownTestSuite() { delete storage; }
};

CheckpointStore* CheckpointStoreTest::storage = nullptr;
size_t CheckpointStoreTest::mem_pool_size = 0;
std::string CheckpointStoreTest::storage_path = "";
int CheckpointStoreTest::num_thread = 0;
size_t CheckpointStoreTest::chunk_size = 0;

TEST_F(CheckpointStoreTest, LoadModelFromDisk) {
  std::string model_path = "facebook/opt-1.3b";
  size_t model_size = 256 * 1024 * 1024;  // 256MB

  std::vector<uint8_t> model_data(model_size, 0xFF);
  std::string data_path = storage_path + "/" + model_path + "/tensor.data_0";

  bool write_success = WriteBytesToFile(data_path, model_data);
  ASSERT_TRUE(write_success) << "Failed to write test data to file";

  // Register model
  size_t registered_model_size = storage->RegisterModelInfo(model_path);
  EXPECT_EQ(registered_model_size, model_size);

  // Load model from disk
  EXPECT_EQ(storage->LoadModelFromDisk(model_path), 0);

  // Clear all models
  EXPECT_EQ(storage->ClearMem(), 0);

  // Remove the model file
  std::filesystem::remove_all(storage_path + "/*");
}
