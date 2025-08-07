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
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
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

  for (int device_id = 0; device_id < 3; ++device_id) {
    EXPECT_NE(memory_ptrs.find(device_id), memory_ptrs.end());
    EXPECT_NE(memory_ptrs[device_id], nullptr);
  }

  std::vector<unsigned char> patterns = {0xAA, 0xBB, 0xCC};
  for (int device_id = 0; device_id < 3; ++device_id) {
    char* ptr = static_cast<char*>(memory_ptrs[device_id]);
    memset(ptr, patterns[device_id], 1024);

    EXPECT_EQ(static_cast<unsigned char>(ptr[0]), patterns[device_id]);
    EXPECT_EQ(static_cast<unsigned char>(ptr[1023]), patterns[device_id]);
  }
}

TEST_F(SharedMemoryTest, AllocateSharedMemoryChunkAlignment) {
  std::unordered_map<int, size_t> tensor_sizes = {
      {0, chunk_size / 2}, {1, chunk_size + 1024}, {2, 3 * chunk_size + 512}};

  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);

  EXPECT_EQ(memory_ptrs.size(), 3);

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

  for (int device_id = 0; device_id < 3; ++device_id) {
    EXPECT_NE(handles.find(device_id), handles.end());
    EXPECT_FALSE(handles[device_id].empty());
    EXPECT_TRUE(handles[device_id].find("tensor_device_") != std::string::npos);
  }

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

  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);
  ASSERT_EQ(memory_ptrs.size(), 2);

  const std::string test_data_0 = "Device 0 test data";
  const std::string test_data_1 = "Device 1 test data with more content";

  strcpy(static_cast<char*>(memory_ptrs[0]), test_data_0.c_str());
  strcpy(static_cast<char*>(memory_ptrs[1]), test_data_1.c_str());

  auto handles = GetSharedMemoryHandles(memory_ptrs);
  ASSERT_EQ(handles.size(), 2);

  EXPECT_STREQ(static_cast<char*>(memory_ptrs[0]), test_data_0.c_str());
  EXPECT_STREQ(static_cast<char*>(memory_ptrs[1]), test_data_1.c_str());

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

  memset(ptr, 0xDE, 1024);
  memset(ptr + 32 * 1024 * 1024, 0xAD, 1024);
  memset(ptr + 63 * 1024 * 1024, 0xBE, 1024);

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

  for (auto& thread : threads) {
    thread.join();
  }

  for (int i = 0; i < num_threads; ++i) {
    EXPECT_EQ(results[i].size(), 1);
    EXPECT_NE(results[i][i], nullptr);
    EXPECT_EQ(handle_results[i].size(), 1);
    EXPECT_FALSE(handle_results[i][i].empty());
  }

  std::set<std::string> unique_handles;
  for (int i = 0; i < num_threads; ++i) {
    unique_handles.insert(handle_results[i][i]);
  }
  EXPECT_EQ(unique_handles.size(), num_threads);
}

TEST_F(SharedMemoryTest, SubprocessBasicReadWrite) {
  // Test basic shared memory read/write across subprocess boundary
  std::unordered_map<int, size_t> tensor_sizes = {{0, 4 * 1024 * 1024}};  // 4MB

  auto memory_ptrs = AllocateSharedMemory(tensor_sizes, chunk_size);
  ASSERT_EQ(memory_ptrs.size(), 1);
  ASSERT_NE(memory_ptrs[0], nullptr);

  auto handles = GetSharedMemoryHandles(memory_ptrs);
  ASSERT_EQ(handles.size(), 1);
  ASSERT_FALSE(handles[0].empty());

  // Write test data in parent process
  const std::string parent_data = "Parent process test data";
  strcpy(static_cast<char*>(memory_ptrs[0]), parent_data.c_str());

  LOG(INFO) << "Parent wrote data to shared memory";

  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process read data written by parent
    char* child_ptr = static_cast<char*>(memory_ptrs[0]);
    std::string read_data(child_ptr);

    if (read_data == parent_data) {
      const std::string child_data = "Child process response";
      strcpy(child_ptr, child_data.c_str());

      _exit(0);
    } else {
      _exit(1);
    }
  } else {
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);

    std::string response_data(static_cast<char*>(memory_ptrs[0]));
    EXPECT_EQ(response_data, "Child process response");
  }
}

TEST_F(SharedMemoryTest, SubprocessMultipleDevicesReadWrite) {
  std::unordered_map<int, size_t> sizes = {
      {0, 2 * 1024 * 1024}, {1, 4 * 1024 * 1024}, {2, 6 * 1024 * 1024}};

  auto ptrs = AllocateSharedMemory(sizes, chunk_size);
  ASSERT_EQ(ptrs.size(), 3);

  auto handles = GetSharedMemoryHandles(ptrs);
  ASSERT_EQ(handles.size(), 3);

  std::vector<std::string> parent_data = {
      "Device 0 parent data", "Device 1 parent data with more content",
      "Device 2 parent data with even more content here"};

  for (int i = 0; i < 3; ++i) {
    strcpy(static_cast<char*>(ptrs[i]), parent_data[i].c_str());
  }

  LOG(INFO) << "Parent wrote data to all devices";

  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    bool all_reads_successful = true;
    std::vector<std::string> child_responses = {"Device 0 child response",
                                                "Device 1 child response",
                                                "Device 2 child response"};

    for (int i = 0; i < 3; ++i) {
      char* ptr = static_cast<char*>(ptrs[i]);
      std::string read_data(ptr);

      if (read_data != parent_data[i]) {
        all_reads_successful = false;
        break;
      }

      strcpy(ptr, child_responses[i].c_str());
    }

    _exit(all_reads_successful ? 0 : 1);
  } else {
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);

    std::vector<std::string> expected_responses = {"Device 0 child response",
                                                   "Device 1 child response",
                                                   "Device 2 child response"};

    for (int i = 0; i < 3; ++i) {
      std::string response_data(static_cast<char*>(ptrs[i]));
      EXPECT_EQ(response_data, expected_responses[i]);
    }

    LOG(INFO) << "Parent verified all child responses";
  }
}

TEST_F(SharedMemoryTest, SubprocessConcurrentAccess) {
  std::unordered_map<int, size_t> sizes = {{0, 16 * 1024 * 1024}};

  auto ptrs = AllocateSharedMemory(sizes, chunk_size);
  ASSERT_EQ(ptrs.size(), 1);
  ASSERT_NE(ptrs[0], nullptr);

  struct SharedData {
    std::atomic<int> counter;
    int results[4];
  };

  SharedData* shared_data = static_cast<SharedData*>(ptrs[0]);
  shared_data->counter.store(0);
  memset(shared_data->results, 0, sizeof(shared_data->results));

  LOG(INFO) << "Parent initialized shared data structure";

  const int num_children = 4;
  std::vector<pid_t> child_pids;

  for (int child_id = 0; child_id < num_children; ++child_id) {
    pid_t pid = fork();
    ASSERT_NE(pid, -1);

    if (pid == 0) {
      SharedData* child_shared = static_cast<SharedData*>(ptrs[0]);

      int my_count = child_shared->counter.fetch_add(1) + 1;
      child_shared->results[child_id] = my_count * 10 + child_id;

      usleep(1000 * child_id);

      _exit(0);
    } else {
      child_pids.push_back(pid);
    }
  }

  for (pid_t child_pid : child_pids) {
    int status;
    waitpid(child_pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
  }

  EXPECT_EQ(shared_data->counter.load(), num_children);

  std::set<int> unique_results;
  for (int i = 0; i < num_children; ++i) {
    EXPECT_NE(shared_data->results[i], 0);
    unique_results.insert(shared_data->results[i]);
  }
  EXPECT_EQ(unique_results.size(), num_children);

  LOG(INFO) << "Parent verified concurrent child process results";
}

TEST_F(SharedMemoryTest, SubprocessHandleInheritance) {
  std::unordered_map<int, size_t> sizes = {{0, 8 * 1024 * 1024}};

  auto ptrs = AllocateSharedMemory(sizes, chunk_size);
  ASSERT_EQ(ptrs.size(), 1);

  auto handles = GetSharedMemoryHandles(ptrs);
  ASSERT_EQ(handles.size(), 1);
  ASSERT_FALSE(handles[0].empty());

  const std::string test_message = "Handle inheritance test";
  strcpy(static_cast<char*>(ptrs[0]), test_message.c_str());

  LOG(INFO) << "Parent wrote test message and obtained handle: " << handles[0];

  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    auto child_handles = GetSharedMemoryHandles(ptrs);

    if (child_handles.size() == 1 && child_handles[0] == handles[0] &&
        !child_handles[0].empty()) {
      char* ptr = static_cast<char*>(ptrs[0]);
      std::string read_message(ptr);

      if (read_message == test_message) {
        const std::string child_confirmation = "Child confirmed handle access";
        strcpy(ptr, child_confirmation.c_str());
        _exit(0);
      }
    }
    _exit(1);
  } else {
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);

    std::string confirmation(static_cast<char*>(ptrs[0]));
    EXPECT_EQ(confirmation, "Child confirmed handle access");

    LOG(INFO) << "Parent verified child handle inheritance worked";
  }
}