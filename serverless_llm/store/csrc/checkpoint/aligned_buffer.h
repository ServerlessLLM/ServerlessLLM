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

#include <string>

const size_t kAlignment = 4096;       // 4k
const size_t kBufferSize = 1L << 30;  // 1GB

// A write buffer that writes to a file (4k aligned).
class AlignedBuffer {
 public:
  explicit AlignedBuffer(const std::string& filename);
  ~AlignedBuffer();

  size_t writeData(const void* data, size_t size);
  size_t writePadding(size_t padding_size);

 private:
  int fd_;
  size_t buf_size_;
  size_t buf_pos_;
  size_t file_offset_;
  void* buffer_;
};