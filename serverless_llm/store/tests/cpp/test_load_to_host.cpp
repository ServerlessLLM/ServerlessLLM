#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <vector>

#include "checkpoint_store.h"

bool WriteBytesToFile(const std::string& file_path, const std::vector<uint8_t>& data) {
    // Ensure the directory exists
    std::filesystem::create_directories(std::filesystem::path(file_path).parent_path());

    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        return false; // Failed to open file
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    return file.good(); // Return true if the write was successful
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
  std::string model_name = "facebook/opt-1.3b";
  size_t model_size = 256 * 1024 * 1024;  // 256MB

  std::vector<uint8_t> model_data(model_size, 0xFF);
  std::string data_path = storage_path + "/" + model_name + "/tensor.data_0";
  
  bool write_success = WriteBytesToFile(data_path, model_data);
  ASSERT_TRUE(write_success) << "Failed to write test data to file";

  // Register model
  size_t registered_model_size = storage->RegisterModelInfo(model_name);
  EXPECT_EQ(registered_model_size, model_size);

  // Load model from disk
  EXPECT_EQ(storage->LoadModelFromDisk(model_name), 0);

  // Clear all models
  EXPECT_EQ(storage->ClearMem(), 0);

  // Remove the model file
  std::filesystem::remove_all(storage_path + "/*");
}
