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
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "aligned_buffer.h"

const size_t kPartitionMaxSize = 10L << 30;  // 10GB

// A tensor writer that writes the raw tensor data to a file in raw binary.
class TensorWriter final {
 public:
  explicit TensorWriter(const std::string& filename);
  ~TensorWriter();

  uint64_t writeRecord(const char* data, size_t size);

 private:
  size_t offset_ = 0;
  int partition_idx_ = -1;
  size_t partition_size_ = 0;
  std::string filename_;
  std::unique_ptr<AlignedBuffer> buffer_;
};