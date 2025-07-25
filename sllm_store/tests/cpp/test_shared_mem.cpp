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

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef USE_HIP
#include "checkpoint_store_hip.h"
#else
#include "checkpoint_store.h"
#endif

#include "memory_allocators.h"
#include "shared_memory.h"

class SharedMemoryTest : public ::testing::Test {
 protected:
  static size_t chunk_size;

  static void SetUpTestSuite() {
    chunk_size = 4 * 1024 * 1024;  // 4MB chunks
  }
};

size_t SharedMemoryTest::chunk_size =
    4 * 1024 * 1024;  // Default chunk size of 4MB

TEST_F(SharedMemoryTest, AllocateSharedMemoryBasic) {
  // Test basic allocation for a single device
  std::unordered_map<int, size_t> tensor_sizes = {{0, 8 * 1024 * 1024}};  // 8MB

  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);

  EXPECT_EQ(memory_ptrs.size(), 1);
  EXPECT_NE(memory_ptrs.find(0), memory_ptrs.end());
  EXPECT_NE(memory_ptrs[0], nullptr);

  // Test that we can write to the allocated memory
  void* ptr = memory_ptrs[0];
  char* char_ptr = static_cast<char*>(ptr);
  memset(char_ptr, 0xAB, 1024);  // Write 1KB of test data

  LOG(INFO) << "Memory has been written";

  // Verify the data was written correctly
  EXPECT_EQ(static_cast<unsigned char>(char_ptr[0]), 0xAB);
  EXPECT_EQ(static_cast<unsigned char>(char_ptr[1023]), 0xAB);

  LOG(INFO) << "Memory verification passed";
}

TEST_F(SharedMemoryTest, AllocateSharedMemoryMultipleDevices) {
  // Test allocation for multiple devices with different sizes
  std::unordered_map<int, size_t> tensor_sizes = {
      {0, 4 * 1024 * 1024},  // 4MB for device 0
      {1, 8 * 1024 * 1024},  // 8MB for device 1
      {2, 12 * 1024 * 1024}  // 12MB for device 2
  };

  LOG(INFO) << "Tensor sizes for multiple devices set";

  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);

  LOG(INFO) << "Memory pointers for multiple devices received";

  EXPECT_EQ(memory_ptrs.size(), 3);

  // Verify all devices have allocated memory
  for (int device_id = 0; device_id < 3; ++device_id) {
    EXPECT_NE(memory_ptrs.find(device_id), memory_ptrs.end());
    EXPECT_NE(memory_ptrs[device_id], nullptr);
  }

  // Test writing different patterns to each device's memory
  std::vector<unsigned char> patterns = {0xAA, 0xBB, 0xCC};
  for (int device_id = 0; device_id < 3; ++device_id) {
    char* ptr = static_cast<char*>(memory_ptrs[device_id]);
    memset(ptr, patterns[device_id], 1024);

    // Verify the pattern was written correctly
    EXPECT_EQ(static_cast<unsigned char>(ptr[0]), patterns[device_id]);
    EXPECT_EQ(static_cast<unsigned char>(ptr[1023]), patterns[device_id]);
  }
}

TEST_F(SharedMemoryTest, AllocateSharedMemoryChunkAlignment) {
  // Test that memory allocation respects chunk size alignment
  std::unordered_map<int, size_t> tensor_sizes = {
      {0, chunk_size / 2},       // Half chunk size
      {1, chunk_size + 1024},    // Just over one chunk
      {2, 3 * chunk_size + 512}  // 3.5 chunks
  };

  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);

  EXPECT_EQ(memory_ptrs.size(), 3);

  // Verify all allocations succeeded
  for (int device_id = 0; device_id < 3; ++device_id) {
    EXPECT_NE(memory_ptrs.find(device_id), memory_ptrs.end());
    EXPECT_NE(memory_ptrs[device_id], nullptr);
  }

  // The actual allocated sizes should be rounded up to chunk boundaries
  // Device 0: 0.5 chunks -> 1 chunk (4MB)
  // Device 1: 1+ chunks -> 2 chunks (8MB)
  // Device 2: 3.5 chunks -> 4 chunks (16MB)
}

TEST_F(SharedMemoryTest, GetSharedMemoryHandlesBasic) {
  // Test getting handles for allocated shared memory
  std::unordered_map<int, size_t> tensor_sizes = {{0, 8 * 1024 * 1024}};

  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);
  ASSERT_EQ(memory_ptrs.size(), 1);
  ASSERT_NE(memory_ptrs[0], nullptr);

  auto handles = GetSharedMemoryHandles(memory_ptrs);

  EXPECT_EQ(handles.size(), 1);
  EXPECT_NE(handles.find(0), handles.end());
  EXPECT_FALSE(handles[0].empty());

  // Handle should start with "tensor_device_" prefix
  EXPECT_TRUE(handles[0].find("tensor_device_") != std::string::npos);
}

TEST_F(SharedMemoryTest, GetSharedMemoryHandlesMultipleDevices) {
  // Test getting handles for multiple devices
  std::unordered_map<int, size_t> tensor_sizes = {
      {0, 4 * 1024 * 1024}, {1, 8 * 1024 * 1024}, {2, 12 * 1024 * 1024}};

  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);
  ASSERT_EQ(memory_ptrs.size(), 3);

  auto handles = GetSharedMemoryHandles(memory_ptrs);

  EXPECT_EQ(handles.size(), 3);

  // Verify all devices have valid handles
  for (int device_id = 0; device_id < 3; ++device_id) {
    EXPECT_NE(handles.find(device_id), handles.end());
    EXPECT_FALSE(handles[device_id].empty());
    EXPECT_TRUE(handles[device_id].find("tensor_device_") != std::string::npos);
  }

  // All handles should be different (each device gets its own shared memory)
  EXPECT_NE(handles[0], handles[1]);
  EXPECT_NE(handles[1], handles[2]);
  EXPECT_NE(handles[0], handles[2]);
}

TEST_F(SharedMemoryTest, AllocateAndGetHandlesWorkflow) {
  // Test the complete workflow: allocate -> write data -> get handles -> verify
  std::unordered_map<int, size_t> tensor_sizes = {
      {0, 2 * 1024 * 1024},  // 2MB
      {1, 6 * 1024 * 1024}   // 6MB
  };

  // Step 1: Allocate shared memory
  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);
  ASSERT_EQ(memory_ptrs.size(), 2);

  // Step 2: Write test data to each device's memory
  const std::string test_data_0 = "Device 0 test data";
  const std::string test_data_1 = "Device 1 test data with more content";

  strcpy(static_cast<char*>(memory_ptrs[0]), test_data_0.c_str());
  strcpy(static_cast<char*>(memory_ptrs[1]), test_data_1.c_str());

  // Step 3: Get shared memory handles
  auto handles = GetSharedMemoryHandles(memory_ptrs);
  ASSERT_EQ(handles.size(), 2);

  // Step 4: Verify data integrity after getting handles
  EXPECT_STREQ(static_cast<char*>(memory_ptrs[0]), test_data_0.c_str());
  EXPECT_STREQ(static_cast<char*>(memory_ptrs[1]), test_data_1.c_str());

  // Step 5: Verify handles are valid
  EXPECT_FALSE(handles[0].empty());
  EXPECT_FALSE(handles[1].empty());
}

TEST_F(SharedMemoryTest, EmptyTensorSizes) {
  // Test edge case: empty tensor sizes map
  std::unordered_map<int, size_t> tensor_sizes = {};

  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);
  EXPECT_TRUE(memory_ptrs.empty());

  auto handles = GetSharedMemoryHandles(memory_ptrs);
  EXPECT_TRUE(handles.empty());
}

TEST_F(SharedMemoryTest, ZeroSizeAllocation) {
  // Test edge case: zero size allocation
  std::unordered_map<int, size_t> tensor_sizes = {{0, 0}};

  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);

  // Should not allocate chunks
  EXPECT_EQ(memory_ptrs.size(), 0);
}

TEST_F(SharedMemoryTest, LargeAllocation) {
  // Test allocation of larger memory sizes
  std::unordered_map<int, size_t> tensor_sizes = {
      {0, 64 * 1024 * 1024}  // 64MB
  };

  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);
  ASSERT_EQ(memory_ptrs.size(), 1);
  ASSERT_NE(memory_ptrs[0], nullptr);

  // Test writing to the entire allocated space
  char* ptr = static_cast<char*>(memory_ptrs[0]);

  // Write pattern to beginning
  memset(ptr, 0xDE, 1024);

  // Write pattern to middle
  memset(ptr + 32 * 1024 * 1024, 0xAD, 1024);

  // Write pattern to near end
  memset(ptr + 63 * 1024 * 1024, 0xBE, 1024);

  // Verify patterns
  EXPECT_EQ(static_cast<unsigned char>(ptr[0]), 0xDE);
  EXPECT_EQ(static_cast<unsigned char>(ptr[32 * 1024 * 1024]), 0xAD);
  EXPECT_EQ(static_cast<unsigned char>(ptr[63 * 1024 * 1024]), 0xBE);

  auto handles = GetSharedMemoryHandles(memory_ptrs);
  EXPECT_EQ(handles.size(), 1);
  EXPECT_FALSE(handles[0].empty());
}

TEST_F(SharedMemoryTest, ConcurrentAllocations) {
  // Test multiple concurrent allocations to verify thread safety
  const int num_threads = 4;
  std::vector<std::thread> threads;
  std::vector<std::unordered_map<int, void*>> results(num_threads);
  std::vector<std::unordered_map<int, std::string>> handle_results(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([i, &results, &handle_results]() {
      std::unordered_map<int, size_t> tensor_sizes = {
          {i, (i + 1) * 2 * 1024 * 1024}  // Different sizes for each thread
      };

      results[i] =
          AllocateSharedMemory(tensor_sizes, SharedMemoryTest::chunk_size);
      if (!results[i].empty()) {
        handle_results[i] = GetSharedMemoryHandles(results[i]);
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify all allocations succeeded
  for (int i = 0; i < num_threads; ++i) {
    EXPECT_EQ(results[i].size(), 1);
    EXPECT_NE(results[i][i], nullptr);
    EXPECT_EQ(handle_results[i].size(), 1);
    EXPECT_FALSE(handle_results[i][i].empty());
  }

  // Verify all handles are unique
  std::set<std::string> unique_handles;
  for (int i = 0; i < num_threads; ++i) {
    unique_handles.insert(handle_results[i][i]);
  }
  EXPECT_EQ(unique_handles.size(), num_threads);
}

TEST_F(SharedMemoryTest, ExplicitCleanupAfterAllocation) {
  // Test explicit cleanup to verify no double-free issues
  std::unordered_map<int, size_t> tensor_sizes = {{0, 4 * 1024 * 1024}};

  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);
  ASSERT_EQ(memory_ptrs.size(), 1);
  ASSERT_NE(memory_ptrs[0], nullptr);

  // Write some data to verify memory is accessible
  memset(memory_ptrs[0], 0xCC, 1024);

  // Explicitly trigger cleanup (this should be safe to call multiple times)
  MemoryRegistry::Instance().CleanupAll();
  MemoryRegistry::Instance().CleanupAll();  // Should be idempotent
}

TEST_F(SharedMemoryTest, MemoryReuseAfterCleanup) {
  // Test that we can allocate, cleanup, and allocate again
  std::unordered_map<int, size_t> tensor_sizes = {{0, 4 * 1024 * 1024}};

  // First allocation
  auto memory_ptrs1 = AllocateSharedMemory(tensor_sizes, chunk_size);
  ASSERT_EQ(memory_ptrs1.size(), 1);
  ASSERT_NE(memory_ptrs1[0], nullptr);
  memset(memory_ptrs1[0], 0xAA, 1024);

  // Cleanup
  MemoryRegistry::Instance().CleanupAll();

  // Second allocation - should work without issues
  auto memory_ptrs2 = AllocateSharedMemory(tensor_sizes, chunk_size);
  ASSERT_EQ(memory_ptrs2.size(), 1);
  ASSERT_NE(memory_ptrs2[0], nullptr);
  memset(memory_ptrs2[0], 0xBB, 1024);

  // Verify the second allocation is independent
  EXPECT_EQ(static_cast<unsigned char*>(memory_ptrs2[0])[0], 0xBB);
}

TEST_F(SharedMemoryTest, MultipleAllocationsWithoutCleanup) {
  // Test multiple allocations to detect potential memory leaks
  // This simulates what happens when tests run without explicit cleanup

  for (int i = 0; i < 5; ++i) {
    std::unordered_map<int, size_t> tensor_sizes = {
        {i, (i + 1) * 1024 * 1024}  // 1MB, 2MB, 3MB, 4MB, 5MB
    };

    auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);
    ASSERT_EQ(memory_ptrs.size(), 1);
    ASSERT_NE(memory_ptrs[i], nullptr);

    // Write unique pattern for each iteration
    unsigned char pattern = 0x10 + i;
    memset(memory_ptrs[i], pattern, 1024);
    EXPECT_EQ(static_cast<unsigned char*>(memory_ptrs[i])[0], pattern);
  }

  // All allocations should succeed without crashes
  // Cleanup will happen automatically via exit handlers
}
