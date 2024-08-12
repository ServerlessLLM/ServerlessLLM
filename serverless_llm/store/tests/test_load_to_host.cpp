#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>

#include "checkpoint_store.h"

class CheckpointStoreTest : public ::testing::Test {
 protected:
  static CheckpointStore* storage;
  static size_t mem_pool_size;
  static std::string storage_path;
  static int num_thread;
  static size_t chunk_size;

  static void SetUpTestCase() {
    mem_pool_size = 4L * 1024 * 1024 * 1024;  // 1GB
    storage_path = "./models";
    num_thread = 4;
    chunk_size = 4 * 1024 * 1024;  // 4MB

    storage = new CheckpointStore(storage_path, mem_pool_size, num_thread,
                                  chunk_size);
  }

  static void TearDownTestCase() { delete storage; }
};

CheckpointStore* CheckpointStoreTest::storage = nullptr;
size_t CheckpointStoreTest::mem_pool_size = 0;
std::string CheckpointStoreTest::storage_path = "";
int CheckpointStoreTest::num_thread = 0;
size_t CheckpointStoreTest::chunk_size = 0;

TEST_F(CheckpointStoreTest, LoadModelFromDisk) {
  std::string model_name = "facebook/opt-1.3b";

  // Register model
  EXPECT_GE(storage->RegisterModelInfo(model_name), 0);

  // Load model from disk
  EXPECT_EQ(storage->LoadModelFromDisk(model_name), 0);

  // Clear all models
  EXPECT_EQ(storage->ClearMem(), 0);
}