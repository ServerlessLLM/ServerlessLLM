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

#include <memory>
#include <vector>

#include "pinned_memory.h"
#include "pinned_memory_pool.h"

class TemplatedPinnedMemoryPoolTest : public ::testing::Test {
 protected:
  static size_t mem_pool_size;
  static size_t chunk_size;

  static void SetUpTestSuite() {
    mem_pool_size = 64 * 1024 * 1024;  // 64MB
    chunk_size = 4 * 1024 * 1024;      // 4MB
  }
};

size_t TemplatedPinnedMemoryPoolTest::mem_pool_size = 0;
size_t TemplatedPinnedMemoryPoolTest::chunk_size = 0;

TEST_F(TemplatedPinnedMemoryPoolTest, AlignedAllocatorBasic) {
  auto pool =
      std::make_shared<AlignedPinnedMemoryPool>(mem_pool_size, chunk_size);

  std::vector<char*> buffers;
  EXPECT_EQ(pool->Allocate(8 * 1024 * 1024, buffers), 0);  // 8MB -> 2 chunks
  EXPECT_EQ(buffers.size(), 2);

  for (auto buffer : buffers) {
    ASSERT_NE(buffer, nullptr);
    // Write some data to test the buffer
    memset(buffer, 0xAB, chunk_size);
  }

  EXPECT_EQ(pool->Deallocate(buffers), 0);
}

TEST_F(TemplatedPinnedMemoryPoolTest, SharedMemoryAllocatorBasic) {
  auto pool = std::make_shared<SharedPinnedMemoryPool>(mem_pool_size,
                                                       chunk_size, "test_pool");

  std::vector<char*> buffers;
  EXPECT_EQ(pool->Allocate(8 * 1024 * 1024, buffers), 0);  // 8MB -> 2 chunks
  EXPECT_EQ(buffers.size(), 2);

  for (auto buffer : buffers) {
    ASSERT_NE(buffer, nullptr);
    // Write some data to test the buffer
    memset(buffer, 0xCD, chunk_size);
  }

  EXPECT_EQ(pool->Deallocate(buffers), 0);
}

TEST_F(TemplatedPinnedMemoryPoolTest, PinnedMemoryWithAlignedPool) {
  auto pool =
      std::make_shared<AlignedPinnedMemoryPool>(mem_pool_size, chunk_size);

  PinnedMemory memory;
  EXPECT_EQ(memory.Allocate(12 * 1024 * 1024, pool), 0);  // 12MB -> 3 chunks
  EXPECT_EQ(memory.num_chunks(), 3);
  EXPECT_EQ(memory.chunk_size(), chunk_size);

  auto& buffers = memory.get();
  EXPECT_EQ(buffers.size(), 3);

  for (auto buffer : buffers) {
    ASSERT_NE(buffer, nullptr);
    // Test buffer accessibility
    memset(buffer, 0xEF, chunk_size);
  }
  // Memory automatically freed when 'memory' goes out of scope
}

TEST_F(TemplatedPinnedMemoryPoolTest, PinnedMemoryWithSharedPool) {
  auto pool = std::make_shared<SharedPinnedMemoryPool>(
      mem_pool_size, chunk_size, "test_shared");

  PinnedMemory memory;
  EXPECT_EQ(memory.Allocate(12 * 1024 * 1024, pool), 0);  // 12MB -> 3 chunks
  EXPECT_EQ(memory.num_chunks(), 3);
  EXPECT_EQ(memory.chunk_size(), chunk_size);

  auto& buffers = memory.get();
  EXPECT_EQ(buffers.size(), 3);

  for (auto buffer : buffers) {
    ASSERT_NE(buffer, nullptr);
    // Test buffer accessibility
    memset(buffer, 0x12, chunk_size);
  }
  // Memory automatically freed when 'memory' goes out of scope
}

TEST_F(TemplatedPinnedMemoryPoolTest, CustomTemplateInstantiation) {
  // Demonstrate custom template usage
  auto custom_pool = std::make_shared<PinnedMemoryPool<AlignedAllocator>>(
      mem_pool_size, chunk_size);

  std::vector<char*> buffers;
  EXPECT_EQ(custom_pool->Allocate(4 * 1024 * 1024, buffers),
            0);  // 4MB -> 1 chunk
  EXPECT_EQ(buffers.size(), 1);
  ASSERT_NE(buffers[0], nullptr);

  EXPECT_EQ(custom_pool->Deallocate(buffers), 0);
}

TEST_F(TemplatedPinnedMemoryPoolTest, MixedUsage) {
  // Show that different allocator types can coexist
  auto aligned_pool =
      std::make_shared<AlignedPinnedMemoryPool>(mem_pool_size, chunk_size);
  auto shared_pool = std::make_shared<SharedPinnedMemoryPool>(
      mem_pool_size, chunk_size, "mixed_test");

  PinnedMemory aligned_memory;
  PinnedMemory shared_memory;

  EXPECT_EQ(aligned_memory.Allocate(8 * 1024 * 1024, aligned_pool), 0);
  EXPECT_EQ(shared_memory.Allocate(8 * 1024 * 1024, shared_pool), 0);

  EXPECT_EQ(aligned_memory.num_chunks(), 2);
  EXPECT_EQ(shared_memory.num_chunks(), 2);

  // Both can be used simultaneously
  auto& aligned_buffers = aligned_memory.get();
  auto& shared_buffers = shared_memory.get();

  EXPECT_EQ(aligned_buffers.size(), 2);
  EXPECT_EQ(shared_buffers.size(), 2);

  // Both types work correctly
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_NE(aligned_buffers[i], nullptr);
    ASSERT_NE(shared_buffers[i], nullptr);

    memset(aligned_buffers[i], 0x34, chunk_size);
    memset(shared_buffers[i], 0x56, chunk_size);
  }
}
