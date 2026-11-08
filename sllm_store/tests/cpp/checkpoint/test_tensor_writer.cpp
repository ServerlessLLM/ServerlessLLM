// Copyright (c) ServerlessLLM Team 2026
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "aligned_buffer.h"
#include "tensor_writer.h"

namespace {
enum class WriteFault { None, Interrupt, ShortWrite, NoProgress, NoSpace };
WriteFault write_fault = WriteFault::None;
}  // namespace

extern "C" ssize_t __real_pwrite(int, const void*, size_t, off_t);

extern "C" ssize_t __wrap_pwrite(int fd, const void* data, size_t size,
                                 off_t offset) {
  const auto fault = write_fault;
  write_fault = WriteFault::None;
  switch (fault) {
    case WriteFault::Interrupt:
      errno = EINTR;
      return -1;
    case WriteFault::ShortWrite:
      return __real_pwrite(fd, data, std::min(size, kAlignment), offset);
    case WriteFault::NoProgress:
      return 0;
    case WriteFault::NoSpace:
      errno = ENOSPC;
      return -1;
    default:
      return __real_pwrite(fd, data, size, offset);
  }
}

namespace {
class CheckpointWriterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    char path[] = "/tmp/sllm-checkpoint-test-XXXXXX";
    const char* result = mkdtemp(path);
    ASSERT_NE(result, nullptr);
    directory_ = result;
    write_fault = WriteFault::None;
  }

  void TearDown() override {
    write_fault = WriteFault::None;
    std::filesystem::remove_all(directory_);
  }

  std::string base() const { return (directory_ / "tensor.data").string(); }
  std::string partition(int index = 0) const {
    return base() + "_" + std::to_string(index);
  }

  std::vector<char> read(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
  }

  void expectContents(const std::string& path, const std::vector<char>& data) {
    const auto saved = read(path);
    const size_t padded_size =
        (data.size() + kAlignment - 1) / kAlignment * kAlignment;
    ASSERT_EQ(saved.size(), padded_size);
    EXPECT_TRUE(std::equal(data.begin(), data.end(), saved.begin()));
    EXPECT_TRUE(std::all_of(saved.begin() + data.size(), saved.end(),
                            [](char value) { return value == 0; }));
  }

  std::filesystem::path directory_;
};

TEST_F(CheckpointWriterTest, SavesUnalignedRecordsAndZeroFillsTheTail) {
  for (size_t size : {1, 2, 8, 512, 520, 4094, 4096, 4098}) {
    SCOPED_TRACE(size);
    const std::vector<char> data(size, 'x');
    TensorWriter writer(base(), kPartitionMaxSize, kAlignment);
    EXPECT_EQ(writer.writeRecord(data.data(), data.size()), 0);
    writer.finish();
    writer.finish();  // Finalization is idempotent.
    expectContents(partition(), data);
    EXPECT_THROW(writer.writeRecord(data.data(), 1), std::logic_error);
  }
}

TEST_F(CheckpointWriterTest, KeepsSmallRecordOffsetsAndZeroesRecordPadding) {
  TensorWriter writer(base(), kPartitionMaxSize, kAlignment);
  const char first[] = {'a', 'b'};
  const char second[] = {'c', 'd', 'e'};
  EXPECT_EQ(writer.writeRecord(first, sizeof(first)), 0);
  EXPECT_EQ(writer.writeRecord(second, sizeof(second)), 8);
  writer.finish();
  expectContents(partition(), {'a', 'b', 0, 0, 0, 0, 0, 0, 'c', 'd', 'e'});
}

TEST_F(CheckpointWriterTest, PreservesTheTailAfterAFullBuffer) {
  TensorWriter writer(base(), kPartitionMaxSize, kAlignment);
  std::vector<char> data(kAlignment, 'a');
  EXPECT_EQ(writer.writeRecord(data.data(), data.size()), 0);
  EXPECT_EQ(writer.writeRecord("b", 1), kAlignment);
  writer.finish();
  data.push_back('b');
  expectContents(partition(), data);
}

TEST_F(CheckpointWriterTest, IncludesPhysicalPaddingInNextPartitionOffset) {
  TensorWriter writer(base(), kAlignment, kAlignment);
  const std::vector<char> first(kAlignment - 8, 'a');
  const std::vector<char> second(16, 'b');
  EXPECT_EQ(writer.writeRecord(first.data(), first.size()), 0);
  EXPECT_EQ(writer.writeRecord(second.data(), second.size()), kAlignment);
  writer.finish();
  expectContents(partition(0), first);
  expectContents(partition(1), second);

  // Model::ToHost concatenates physical partitions. Both halves of a read
  // crossing the boundary must satisfy O_DIRECT, including the target address.
  void* memory = aligned_alloc(kAlignment, 2 * kAlignment);
  ASSERT_NE(memory, nullptr);
  const int first_fd = open(partition(0).c_str(), O_RDONLY | O_DIRECT);
  const int second_fd = open(partition(1).c_str(), O_RDONLY | O_DIRECT);
  ASSERT_GE(first_fd, 0);
  ASSERT_GE(second_fd, 0);
  const ssize_t count = pread(first_fd, memory, 2 * kAlignment, 0);
  EXPECT_EQ(count, kAlignment);
  EXPECT_EQ(
      pread(second_fd, static_cast<char*>(memory) + kAlignment, kAlignment, 0),
      kAlignment);
  EXPECT_TRUE(
      std::equal(first.begin(), first.end(), static_cast<char*>(memory)));
  EXPECT_TRUE(std::equal(second.begin(), second.end(),
                         static_cast<char*>(memory) + kAlignment));
  close(first_fd);
  close(second_fd);
  free(memory);
}

TEST_F(CheckpointWriterTest, TruncatesAnExistingPartition) {
  {
    TensorWriter writer(base(), kPartitionMaxSize, kAlignment);
    const std::vector<char> data(2 * kAlignment, 'a');
    writer.writeRecord(data.data(), data.size());
    writer.finish();
  }
  TensorWriter writer(base(), kPartitionMaxSize, kAlignment);
  writer.writeRecord("b", 1);
  writer.finish();
  expectContents(partition(), {'b'});
}

TEST_F(CheckpointWriterTest, ReportsOpenFailures) {
  std::filesystem::create_directory(partition());
  TensorWriter writer(base(), kPartitionMaxSize, kAlignment);
  EXPECT_THROW(writer.writeRecord("x", 1), std::system_error);
}

TEST_F(CheckpointWriterTest, ReportsFinalWriteFailures) {
  TensorWriter writer(base(), kPartitionMaxSize, kAlignment);
  writer.writeRecord("x", 1);
  write_fault = WriteFault::NoSpace;
  EXPECT_THROW(writer.finish(), std::system_error);
}

TEST_F(CheckpointWriterTest, ReportsFullBufferAndPartitionRolloverFailures) {
  const std::vector<char> data(kAlignment, 'a');
  {
    TensorWriter writer(base(), kPartitionMaxSize, kAlignment);
    write_fault = WriteFault::NoSpace;
    EXPECT_THROW(writer.writeRecord(data.data(), data.size()),
                 std::system_error);
  }
  TensorWriter writer(base(), kAlignment, kAlignment);
  writer.writeRecord(data.data(), kAlignment - 8);
  write_fault = WriteFault::NoSpace;
  EXPECT_THROW(writer.writeRecord(data.data(), 16), std::system_error);
  EXPECT_FALSE(std::filesystem::exists(partition(1)));
}

TEST_F(CheckpointWriterTest, RetriesInterruptedAndShortWrites) {
  for (auto fault : {WriteFault::Interrupt, WriteFault::ShortWrite}) {
    const std::vector<char> data(2 * kAlignment - 8, 'x');
    TensorWriter writer(base(), kPartitionMaxSize, 2 * kAlignment);
    writer.writeRecord(data.data(), data.size());
    write_fault = fault;
    writer.finish();
    expectContents(partition(), data);
  }
}

TEST_F(CheckpointWriterTest, ReportsWritesThatMakeNoProgress) {
  TensorWriter writer(base(), kPartitionMaxSize, kAlignment);
  writer.writeRecord("x", 1);
  write_fault = WriteFault::NoProgress;
  EXPECT_THROW(writer.finish(), std::runtime_error);
}
}  // namespace
