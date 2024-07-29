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
#include "aligned_buffer.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

AlignedBuffer::AlignedBuffer(const std::string& filename)
    : fd_(-1), buf_size_(kBufferSize), buf_pos_(0), file_offset_(0) {
  fd_ = open(filename.c_str(), O_WRONLY | O_CREAT | O_DIRECT, 0644);
  if (fd_ < 0) {
    std::cerr << "Failed to open file " << filename << std::endl;
  }
  buffer_ = aligned_alloc(kAlignment, kBufferSize);
}

AlignedBuffer::~AlignedBuffer() {
  // if the buffer is not written to file, write it to file
  if (buffer_ && buf_pos_) {
    pwrite(fd_, buffer_, buf_pos_, file_offset_);
  }
  if (buffer_) {
    free(buffer_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

size_t AlignedBuffer::writeData(const void* data, size_t size) {
  size_t written = 0;
  while (written < size) {
    // if data size is larger than buffer size and buffer is empty
    // write data directly to file
    if (size - written > buf_size_ && buf_pos_ == 0) {
      size_t direct_write_size = (size - written) / kAlignment * kAlignment;
      // allocate aligned memory
      void* direct_write_buf = aligned_alloc(kAlignment, direct_write_size);
      if (!direct_write_buf) {
        char err_msg[256];
        strerror_r(errno, err_msg, sizeof(err_msg));
        std::cerr << "Failed to allocate aligned memory: " << err_msg
                  << std::endl;
        std::cerr << "kAlignment: " << kAlignment
                  << " direct_write_size: " << direct_write_size;
        exit(1);
      }
      memcpy(direct_write_buf, (char*)data + written, direct_write_size);
      ssize_t ret =
          pwrite(fd_, direct_write_buf, direct_write_size, file_offset_);
      if (ret < 0 || ret != direct_write_size) {
        std::cerr << "Failed to write to file, ret: " << ret
                  << " errno: " << errno << std::endl;
        return written;
      }
      written += direct_write_size;
      file_offset_ += direct_write_size;
      free(direct_write_buf);
    }
    size_t to_write = std::min(size - written, buf_size_ - buf_pos_);
    memcpy(static_cast<char*>(buffer_) + buf_pos_, (char*)data + written,
           to_write);
    buf_pos_ += to_write;
    written += to_write;
    if (buf_pos_ == buf_size_) {
      ssize_t ret = pwrite(fd_, buffer_, buf_size_, file_offset_);
      if (ret < 0 || ret != buf_size_) {
        std::cerr << "Failed to write to file, ret: " << ret
                  << " errno: " << errno << std::endl;
        return written;
      }
      buf_pos_ = 0;
      file_offset_ += buf_size_;
    }
  }
  return written;
}

size_t AlignedBuffer::writePadding(size_t padding_size) {
  if (padding_size >= 8) {
    std::cerr << "Padding size should be less than 8 bytes" << std::endl;
    return 0;
  }
  buf_pos_ += padding_size;
  if (buf_pos_ > buf_size_) {
    std::cerr << "Padding size is too large" << std::endl;
    return 0;
  }
  if (buf_pos_ == buf_size_) {
    pwrite(fd_, buffer_, buf_pos_, file_offset_);
    buf_pos_ = 0;
    file_offset_ += buf_size_;
  }
  return padding_size;
}