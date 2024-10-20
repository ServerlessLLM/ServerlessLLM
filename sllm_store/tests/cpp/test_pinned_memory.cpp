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
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "pinned_memory.h"
#include "pinned_memory_pool.h"

class PinnedMemoryPoolTest : public ::testing::Test {
 protected:
  static size_t mem_pool_size;
  static size_t chunk_size;

  static void SetUpTestSuite() {
    mem_pool_size = 1L * 1024 * 1024 * 1024;  // 1GB
    chunk_size = 4 * 1024 * 1024;             // 4MB
  }
};

size_t PinnedMemoryPoolTest::mem_pool_size = 0;
size_t PinnedMemoryPoolTest::chunk_size = 0;

TEST_F(PinnedMemoryPoolTest, InitializePinnedMemoryPool) {
  auto pinned_memory_pool =
      std::make_unique<PinnedMemoryPool>(mem_pool_size, chunk_size);
  EXPECT_EQ(pinned_memory_pool->chunk_size(), chunk_size);
}

TEST_F(PinnedMemoryPoolTest, RegularAllocateDeallocate) {
  size_t size = 1024;
  auto pinned_memory_pool =
      std::make_unique<PinnedMemoryPool>(mem_pool_size, chunk_size);

  std::vector<char*> buffers;
  EXPECT_EQ(pinned_memory_pool->Allocate(size, buffers), 0);
  EXPECT_EQ(buffers.size(), 1);
  ASSERT_NE(buffers[0], nullptr);
  EXPECT_EQ(pinned_memory_pool->Deallocate(buffers), 0);
}

TEST_F(PinnedMemoryPoolTest, IrregularAllocateDeallocate) {
  size_t size = 17 * 1024 * 1024;  // 17MB
  auto pinned_memory_pool =
      std::make_unique<PinnedMemoryPool>(mem_pool_size, chunk_size);

  std::vector<char*> buffers;
  EXPECT_EQ(pinned_memory_pool->Allocate(size, buffers), 0);
  EXPECT_EQ(buffers.size(), 5);
  for (auto buffer : buffers) {
    ASSERT_NE(buffer, nullptr);
  }
  EXPECT_EQ(pinned_memory_pool->Deallocate(buffers), 0);
}

TEST_F(PinnedMemoryPoolTest, OutOfMemoryAllocate) {
  size_t size = 1024 * 1024 * 1024;  // 1GB
  auto pinned_memory_pool =
      std::make_unique<PinnedMemoryPool>(mem_pool_size, chunk_size);

  std::vector<char*> buffers;
  EXPECT_EQ(pinned_memory_pool->Allocate(size, buffers), 0);
  EXPECT_EQ(buffers.size(), 256);
  for (auto buffer : buffers) {
    ASSERT_NE(buffer, nullptr);
  }
  // When out of memory, Allocate() should return the number of buffers needed
  EXPECT_EQ(pinned_memory_pool->Allocate(size, buffers), 256);
  EXPECT_EQ(pinned_memory_pool->Deallocate(buffers), 0);

  // Allocate again
  EXPECT_EQ(pinned_memory_pool->Allocate(size, buffers), 0);
  EXPECT_EQ(buffers.size(), 256);
  for (auto buffer : buffers) {
    ASSERT_NE(buffer, nullptr);
  }
  EXPECT_EQ(pinned_memory_pool->Deallocate(buffers), 0);
}

TEST_F(PinnedMemoryPoolTest, ZeroSizeAllocate) {
  size_t size = 0;
  auto pinned_memory_pool =
      std::make_unique<PinnedMemoryPool>(mem_pool_size, chunk_size);

  std::vector<char*> buffers;
  EXPECT_NE(pinned_memory_pool->Allocate(size, buffers), 0);
}

TEST_F(PinnedMemoryPoolTest, DeallocateNonExistentBuffer) {
  size_t size = 1024;
  auto pinned_memory_pool =
      std::make_unique<PinnedMemoryPool>(mem_pool_size, chunk_size);

  std::vector<char*> buffers;
  EXPECT_EQ(pinned_memory_pool->Allocate(size, buffers), 0);
  EXPECT_EQ(pinned_memory_pool->Deallocate(buffers), 0);
  EXPECT_NE(pinned_memory_pool->Deallocate(buffers), 0);
}

TEST_F(PinnedMemoryPoolTest, ConcurrentAllocateDeallocate) {
  size_t size = 1024;
  auto pinned_memory_pool =
      std::make_unique<PinnedMemoryPool>(mem_pool_size, chunk_size);

  std::vector<std::future<int>> futures;
  for (int i = 0; i < 100; ++i) {
    futures.push_back(
        std::async(std::launch::async, [size, &pinned_memory_pool]() -> int {
          std::vector<char*> buffers;
          if (pinned_memory_pool->Allocate(size, buffers) == 0) {
            if (buffers.size() != 1 || buffers[0] == nullptr) {
              return -1;
            }
            // sleep for a random time
            std::this_thread::sleep_for(
                std::chrono::milliseconds(rand() % 100));
            return pinned_memory_pool->Deallocate(buffers);
          } else {
            return -1;
          }
        }));
  }

  for (auto& future : futures) {
    EXPECT_EQ(future.get(), 0);
  }
}

class PinnedMemoryTest : public ::testing::Test {
 protected:
  static size_t mem_pool_size;
  static size_t chunk_size;

  static void SetUpTestSuite() {
    mem_pool_size = 1L * 1024 * 1024 * 1024;  // 1GB
    chunk_size = 4 * 1024 * 1024;             // 4MB
  }
};

size_t PinnedMemoryTest::mem_pool_size = 0;
size_t PinnedMemoryTest::chunk_size = 0;

TEST_F(PinnedMemoryTest, AllocatePinMemory) {
  size_t size = 1024;
  auto pinned_memory_pool =
      std::make_shared<PinnedMemoryPool>(mem_pool_size, chunk_size);
  std::unique_ptr<PinnedMemory> pinned_memory =
      std::make_unique<PinnedMemory>();
  EXPECT_EQ(pinned_memory->Allocate(size, pinned_memory_pool), 0);
  EXPECT_EQ(pinned_memory->num_chunks(), 1);
  EXPECT_EQ(pinned_memory->chunk_size(), chunk_size);
  auto buffers = pinned_memory->get();
  EXPECT_EQ(buffers.size(), 1);
  ASSERT_NE(buffers[0], nullptr);
}

TEST_F(PinnedMemoryTest, DeallocatePinMemory) {
  size_t size = 1024 * 1024 * 1024;  // 1GB
  auto pinned_memory_pool =
      std::make_shared<PinnedMemoryPool>(mem_pool_size, chunk_size);
  //   PinnedMemory should deallocate memory when it goes out of scope
  for (int i = 0; i < 100; ++i) {
    std::unique_ptr<PinnedMemory> pinned_memory =
        std::make_unique<PinnedMemory>();
    EXPECT_EQ(pinned_memory->Allocate(size, pinned_memory_pool), 0);
    EXPECT_EQ(pinned_memory->num_chunks(), 256);
    EXPECT_EQ(pinned_memory->chunk_size(), chunk_size);
    auto buffers = pinned_memory->get();
    EXPECT_EQ(buffers.size(), 256);
    for (auto buffer : buffers) {
      ASSERT_NE(buffer, nullptr);
    }
  }
}