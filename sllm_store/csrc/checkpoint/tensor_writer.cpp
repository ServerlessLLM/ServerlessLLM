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
#include "tensor_writer.h"

#include <stdexcept>

TensorWriter::TensorWriter(const std::string& filename,
                           size_t partition_max_size, size_t buffer_size)
    : filename_(filename),
      partition_max_size_(partition_max_size),
      buffer_size_(buffer_size) {
  if (partition_max_size_ == 0) {
    throw std::invalid_argument("Partition size must be positive");
  }
}

TensorWriter::~TensorWriter() {}

uint64_t TensorWriter::writeRecord(const char* data, size_t size) {
  if (finished_) {
    throw std::logic_error("Cannot write to a finished tensor writer");
  }
  const size_t padding = (size % 8) ? (8 - size % 8) : 0;
  if (partition_idx_ == -1 ||
      partition_size_ + size + padding > partition_max_size_) {
    finishPartition();
    // create a new partition
    partition_idx_++;
    partition_size_ = 0;
    std::string partition_filename =
        filename_ + "_" + std::to_string(partition_idx_);
    buffer_ = std::make_unique<AlignedBuffer>(partition_filename, buffer_size_);
  }

  uint64_t start_offset = offset_;
  // make sure the data is 64-bit aligned
  size_t written = buffer_->writeData(data, size);
  if (padding) {
    written += buffer_->writePadding(padding);
  }
  offset_ += written;
  partition_size_ += written;
  // std::cerr << "writeRecord: " << partition_idx_ << " " << partition_size_ <<
  // " " << kPartitionMaxSize << std::endl;

  return start_offset;
}

void TensorWriter::finishPartition() {
  if (buffer_) {
    // The reader treats partitions as concatenated physical files, so the
    // next tensor offset must include the previous partition's aligned tail.
    const size_t physical_size = buffer_->finish();
    offset_ += physical_size - partition_size_;
    buffer_.reset();
  }
}

void TensorWriter::finish() {
  finishPartition();
  finished_ = true;
}
